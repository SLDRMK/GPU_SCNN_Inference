#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <cstring>

// 卷积常量内存（适配本模型固定形状）
__constant__ float c_conv1_w[6 * 1 * 5 * 5];
__constant__ float c_conv1_b[6];
__constant__ float c_conv2_w[16 * 6 * 5 * 5];
__constant__ float c_conv2_b[16];

// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================


// ===================================================================================
// CUDA Kernel Functions
// ===================================================================================

// IF神经元核函数
__global__ void if_neuron_kernel(const float* __restrict__ input, float* __restrict__ output, float* __restrict__ voltage, 
                                int size, float threshold, float v_reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        voltage[idx] += input[idx];
        output[idx] = (voltage[idx] >= threshold) ? 1.0f : 0.0f;
        if (voltage[idx] >= threshold) {
            voltage[idx] = v_reset;
        }
    }
}

// 卷积核函数
__global__ void conv2d_kernel(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ weight, const float* __restrict__ bias,
                             int batch_size, int in_channels, int out_channels,
                             int input_h, int input_w, int kernel_size, int output_h, int output_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_channels * output_h * output_w;
    
    if (idx < total_output) {
        int n = idx / (out_channels * output_h * output_w);
        int oc = (idx % (out_channels * output_h * output_w)) / (output_h * output_w);
        int oh = (idx % (output_h * output_w)) / output_w;
        int ow = idx % output_w;
        
        float sum = __ldg(&bias[oc]);
        
        for (int ic = 0; ic < in_channels; ic++) {
            const int input_c_base = n * in_channels * input_h * input_w + ic * input_h * input_w;
            const int weight_c_base = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size;
            if (kernel_size == 5) {
                #pragma unroll
                for (int kh = 0; kh < 5; kh++) {
                    const int ih = oh + kh;
                    if (ih < 0 || ih >= input_h) continue;
                    #pragma unroll
                    for (int kw = 0; kw < 5; kw++) {
                        const int iw = ow + kw;
                        if (iw < 0 || iw >= input_w) continue;
                        const int input_idx = input_c_base + ih * input_w + iw;
                        const int weight_idx = weight_c_base + kh * 5 + kw;
                        sum = fmaf(__ldg(&input[input_idx]), __ldg(&weight[weight_idx]), sum);
                    }
                }
            } else {
                #pragma unroll 2
                for (int kh = 0; kh < kernel_size; kh++) {
                    const int ih = oh + kh;
                    if (ih < 0 || ih >= input_h) continue;
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int iw = ow + kw;
                        if (iw < 0 || iw >= input_w) continue;
                        const int input_idx = input_c_base + ih * input_w + iw;
                        const int weight_idx = weight_c_base + kh * kernel_size + kw;
                        sum = fmaf(__ldg(&input[input_idx]), __ldg(&weight[weight_idx]), sum);
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

// 最大池化核函数
__global__ void maxpool2d_kernel(const float* __restrict__ input, float* __restrict__ output,
                                int batch_size, int channels, int input_h, int input_w,
                                int kernel_size, int stride, int output_h, int output_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * channels * output_h * output_w;
    
    if (idx < total_output) {
        int n = idx / (channels * output_h * output_w);
        int c = (idx % (channels * output_h * output_w)) / (output_h * output_w);
        int oh = (idx % (output_h * output_w)) / output_w;
        int ow = idx % output_w;
        
        if (kernel_size == 2 && stride == 2) {
            int ih = oh << 1;
            int iw = ow << 1;
            int base = n * channels * input_h * input_w + c * input_h * input_w + ih * input_w + iw;
            float v00 = __ldg(&input[base]);
            float v01 = __ldg(&input[base + 1]);
            float v10 = __ldg(&input[base + input_w]);
            float v11 = __ldg(&input[base + input_w + 1]);
            float m0 = fmaxf(v00, v01);
            float m1 = fmaxf(v10, v11);
            output[idx] = fmaxf(m0, m1);
        } else {
            float max_val = -1e30f;
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih2 = oh * stride + kh;
                    int iw2 = ow * stride + kw;
                    if (ih2 < input_h && iw2 < input_w) {
                        int input_idx = n * channels * input_h * input_w + 
                                      c * input_h * input_w + ih2 * input_w + iw2;
                        max_val = fmaxf(max_val, __ldg(&input[input_idx]));
                    }
                }
            }
            output[idx] = max_val;
        }
    }
}

// 全连接层核函数
__global__ void linear_kernel(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ weight, const float* __restrict__ bias,
                             int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_features;
    
    if (idx < total_output) {
        int n = idx / out_features;
        int out_idx = idx % out_features;
        
        float sum = __ldg(&bias[out_idx]);
        const float* in_ptr = input + n * in_features;
        const float* w_ptr = weight + out_idx * in_features;
        if ((in_features & 3) == 0) {
            const float4* in4 = reinterpret_cast<const float4*>(in_ptr);
            const float4* w4  = reinterpret_cast<const float4*>(w_ptr);
            int iters = in_features >> 2;
            #pragma unroll 4
            for (int i = 0; i < iters; ++i) {
                float4 iv = in4[i];
                float4 wv = w4[i];
                sum = fmaf(iv.x, wv.x, sum);
                sum = fmaf(iv.y, wv.y, sum);
                sum = fmaf(iv.z, wv.z, sum);
                sum = fmaf(iv.w, wv.w, sum);
            }
        } else {
            for (int i = 0; i < in_features; i++) {
                sum = fmaf(__ldg(&in_ptr[i]), __ldg(&w_ptr[i]), sum);
            }
        }
        output[idx] = sum;
    }
}

// 融合第三层全连接与脉冲累加的核函数
__global__ void linear_accumulate_kernel(const float* __restrict__ input, float* __restrict__ output,
                                        const float* __restrict__ weight, const float* __restrict__ bias,
                                        float* __restrict__ spike_sum,
                                        int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_features;
    if (idx < total_output) {
        int n = idx / out_features;
        int out_idx = idx % out_features;
        float sum = __ldg(&bias[out_idx]);
        const float* in_ptr = input + n * in_features;
        const float* w_ptr = weight + out_idx * in_features;
        if ((in_features & 3) == 0) {
            const float4* in4 = reinterpret_cast<const float4*>(in_ptr);
            const float4* w4  = reinterpret_cast<const float4*>(w_ptr);
            int iters = in_features >> 2;
            #pragma unroll 4
            for (int i = 0; i < iters; ++i) {
                float4 iv = in4[i];
                float4 wv = w4[i];
                sum = fmaf(iv.x, wv.x, sum);
                sum = fmaf(iv.y, wv.y, sum);
                sum = fmaf(iv.z, wv.z, sum);
                sum = fmaf(iv.w, wv.w, sum);
            }
        } else {
            for (int i = 0; i < in_features; i++) {
                sum = fmaf(__ldg(&in_ptr[i]), __ldg(&w_ptr[i]), sum);
            }
        }
        output[idx] = sum;
        spike_sum[idx] += sum;
    }
}

// 卷积 + IF 融合核函数（valid卷积快速路径，通用回退）
__global__ void conv2d_if_kernel(const float* __restrict__ input, float* __restrict__ output_spike,
                                float* __restrict__ voltage,
                                const float* __restrict__ weight, const float* __restrict__ bias,
                                int batch_size, int in_channels, int out_channels,
                                int input_h, int input_w, int kernel_size, int output_h, int output_w,
                                float threshold, float v_reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_channels * output_h * output_w;
    if (idx < total_output) {
        int n = idx / (out_channels * output_h * output_w);
        int oc = (idx % (out_channels * output_h * output_w)) / (output_h * output_w);
        int oh = (idx % (output_h * output_w)) / output_w;
        int ow = idx % output_w;

        float sum = __ldg(&bias[oc]);
        bool valid_no_pad = (output_h == (input_h - kernel_size + 1)) && (output_w == (input_w - kernel_size + 1));
        for (int ic = 0; ic < in_channels; ic++) {
            const int input_c_base = n * in_channels * input_h * input_w + ic * input_h * input_w;
            const int weight_c_base = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size;
            if (kernel_size == 5 && valid_no_pad) {
                #pragma unroll
                for (int kh = 0; kh < 5; kh++) {
                    const int ih = oh + kh;
                    #pragma unroll
                    for (int kw = 0; kw < 5; kw++) {
                        const int iw = ow + kw;
                        const int input_idx = input_c_base + ih * input_w + iw;
                        const int weight_idx = weight_c_base + kh * 5 + kw;
                        sum = fmaf(__ldg(&input[input_idx]), __ldg(&weight[weight_idx]), sum);
                    }
                }
            } else {
                for (int kh = 0; kh < kernel_size; kh++) {
                    const int ih = oh + kh;
                    if (ih < 0 || ih >= input_h) continue;
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int iw = ow + kw;
                        if (iw < 0 || iw >= input_w) continue;
                        const int input_idx = input_c_base + ih * input_w + iw;
                        const int weight_idx = weight_c_base + kh * kernel_size + kw;
                        sum = fmaf(__ldg(&input[input_idx]), __ldg(&weight[weight_idx]), sum);
                    }
                }
            }
        }

        float v = voltage[idx];
        v += sum;
        float spk = (v >= threshold) ? 1.0f : 0.0f;
        if (spk >= 1.0f) v = v_reset;
        voltage[idx] = v;
        output_spike[idx] = spk;
    }
}

// 使用常量内存权重/偏置的 5x5 valid 卷积 + IF（共享内存 tile）
__global__ void conv2d_if_smem5x5_tiled_kernel_constw(const float* __restrict__ input, float* __restrict__ output_spike,
                                                     float* __restrict__ voltage,
                                                     int batch_size, int in_channels, int out_channels,
                                                     int input_h, int input_w, int output_h, int output_w,
                                                     float threshold, float v_reset,
                                                     int which_layer /*1 or 2*/) {
    const int TILE_H = 16;
    const int TILE_W = 16;
    const int sm_h = TILE_H + 4;
    const int sm_w = TILE_W + 4;
    extern __shared__ float smem[];

    const int tiles_per_row = (output_w + TILE_W - 1) / TILE_W;
    const int tiles_per_col = (output_h + TILE_H - 1) / TILE_H;
    const int tiles_per_fm = tiles_per_row * tiles_per_col;

    int tile_id = blockIdx.x;
    int fm_id = tile_id / tiles_per_fm;
    int tile_within = tile_id % tiles_per_fm;
    int tile_oh = (tile_within / tiles_per_row) * TILE_H;
    int tile_ow = (tile_within % tiles_per_row) * TILE_W;
    int n = fm_id / out_channels;
    int oc = fm_id % out_channels;
    if (n >= batch_size) return;

    int t = threadIdx.x;
    int ty = t / TILE_W;
    int tx = t % TILE_W;

    float sum = (which_layer == 1) ? __ldg(&c_conv1_b[oc]) : __ldg(&c_conv2_b[oc]);

    for (int ic = 0; ic < in_channels; ++ic) {
        const int input_c_base = n * in_channels * input_h * input_w + ic * input_h * input_w;
        for (int load_idx = threadIdx.x; load_idx < sm_h * sm_w; load_idx += blockDim.x) {
            int sy = load_idx / sm_w;
            int sx = load_idx % sm_w;
            int ih = tile_oh + sy;
            int iw = tile_ow + sx;
            float val = 0.0f;
            if (ih < input_h && iw < input_w) {
                int gidx = input_c_base + ih * input_w + iw;
                val = __ldg(&input[gidx]);
            }
            smem[sy * sm_w + sx] = val;
        }
        __syncthreads();

        if (ty < TILE_H && tx < TILE_W) {
            int oh = tile_oh + ty;
            int ow = tile_ow + tx;
            if (oh < output_h && ow < output_w) {
                #pragma unroll
                for (int kh = 0; kh < 5; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < 5; ++kw) {
                        float a = smem[(ty + kh) * sm_w + (tx + kw)];
                        float w;
                        if (which_layer == 1) {
                            int widx = oc * in_channels * 25 + ic * 25 + kh * 5 + kw;
                            w = c_conv1_w[widx];
                        } else {
                            int widx = oc * in_channels * 25 + ic * 25 + kh * 5 + kw;
                            w = c_conv2_w[widx];
                        }
                        sum = fmaf(a, w, sum);
                    }
                }
            }
        }
        __syncthreads();
    }

    int oh = tile_oh + ty;
    int ow = tile_ow + tx;
    if (ty < TILE_H && tx < TILE_W && oh < output_h && ow < output_w) {
        int out_idx = n * out_channels * output_h * output_w + oc * output_h * output_w + oh * output_w + ow;
        float v = voltage[out_idx] + sum;
        float spk = (v >= threshold) ? 1.0f : 0.0f;
        if (spk >= 1.0f) v = v_reset;
        voltage[out_idx] = v;
        output_spike[out_idx] = spk;
    }
}
// 全连接 + IF 融合核函数（向量化+IF）
__global__ void linear_if_kernel(const float* __restrict__ input, float* __restrict__ output_spike,
                                float* __restrict__ voltage,
                                const float* __restrict__ weight, const float* __restrict__ bias,
                                int batch_size, int in_features, int out_features,
                                float threshold, float v_reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_features;
    if (idx < total_output) {
        int n = idx / out_features;
        int out_idx = idx % out_features;
        float sum = __ldg(&bias[out_idx]);
        const float* in_ptr = input + n * in_features;
        const float* w_ptr = weight + out_idx * in_features;
        if ((in_features & 3) == 0) {
            const float4* in4 = reinterpret_cast<const float4*>(in_ptr);
            const float4* w4  = reinterpret_cast<const float4*>(w_ptr);
            int iters = in_features >> 2;
            #pragma unroll 4
            for (int i = 0; i < iters; ++i) {
                float4 iv = in4[i];
                float4 wv = w4[i];
                sum = fmaf(iv.x, wv.x, sum);
                sum = fmaf(iv.y, wv.y, sum);
                sum = fmaf(iv.z, wv.z, sum);
                sum = fmaf(iv.w, wv.w, sum);
            }
        } else {
            for (int i = 0; i < in_features; i++) {
                sum = fmaf(__ldg(&in_ptr[i]), __ldg(&w_ptr[i]), sum);
            }
        }
        float v = voltage[idx];
        v += sum;
        float spk = (v >= threshold) ? 1.0f : 0.0f;
        if (spk >= 1.0f) v = v_reset;
        voltage[idx] = v;
        output_spike[idx] = spk;
    }
}

// 共享内存输入缓存的 FC + IF 融合核（每块处理同一 n 的一组输出）
__global__ void linear_if_tiled_smem_kernel(const float* __restrict__ input, float* __restrict__ output_spike,
                                           float* __restrict__ voltage,
                                           const float* __restrict__ weight, const float* __restrict__ bias,
                                           int batch_size, int in_features, int out_features,
                                           float threshold, float v_reset) {
    // blockIdx.y: 样本 n, blockIdx.x: 输出tile编号，threadIdx.x: tile内输出偏移
    int n = blockIdx.y;
    int tile_base = blockIdx.x * blockDim.x;
    int out_idx = tile_base + threadIdx.x;
    if (n >= batch_size || out_idx >= out_features) return;

    extern __shared__ float sm_in[]; // in_features
    // 协作加载输入向量
    const float* in_ptr = input + n * in_features;
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        sm_in[i] = __ldg(&in_ptr[i]);
    }
    __syncthreads();

    float sum = __ldg(&bias[out_idx]);
    const float* w_ptr = weight + out_idx * in_features;
    if ((in_features & 3) == 0) {
        const float4* in4 = reinterpret_cast<const float4*>(sm_in);
        const float4* w4  = reinterpret_cast<const float4*>(w_ptr);
        int iters = in_features >> 2;
        #pragma unroll 4
        for (int i = 0; i < iters; ++i) {
            float4 iv = in4[i];
            float4 wv = w4[i];
            sum = fmaf(iv.x, wv.x, sum);
            sum = fmaf(iv.y, wv.y, sum);
            sum = fmaf(iv.z, wv.z, sum);
            sum = fmaf(iv.w, wv.w, sum);
        }
    } else {
        for (int i = 0; i < in_features; ++i) {
            sum = fmaf(sm_in[i], __ldg(&w_ptr[i]), sum);
        }
    }

    int out_linear_idx = n * out_features + out_idx;
    float v = voltage[out_linear_idx] + sum;
    float spk = (v >= threshold) ? 1.0f : 0.0f;
    if (spk >= 1.0f) v = v_reset;
    voltage[out_linear_idx] = v;
    output_spike[out_linear_idx] = spk;
}

// 使用共享内存的 5x5 valid 卷积 + IF 融合（16x16 tile，1D block/thread 映射）
__global__ void conv2d_if_smem5x5_tiled_kernel(const float* __restrict__ input, float* __restrict__ output_spike,
                                              float* __restrict__ voltage,
                                              const float* __restrict__ weight, const float* __restrict__ bias,
                                              int batch_size, int in_channels, int out_channels,
                                              int input_h, int input_w, int output_h, int output_w,
                                              float threshold, float v_reset) {
    const int TILE_H = 16;
    const int TILE_W = 16;
    const int sm_h = TILE_H + 4; // + (K-1)
    const int sm_w = TILE_W + 4;

    extern __shared__ float smem[]; // sm_h * sm_w

    // 计算 tile 编号与对应的 (n, oc, oh0, ow0)
    const int tiles_per_row = (output_w + TILE_W - 1) / TILE_W;
    const int tiles_per_col = (output_h + TILE_H - 1) / TILE_H;
    const int tiles_per_fm = tiles_per_row * tiles_per_col;

    int tile_id = blockIdx.x;
    int fm_id = tile_id / tiles_per_fm; // 0 .. batch*out_channels-1
    int tile_within = tile_id % tiles_per_fm;
    int tile_oh = (tile_within / tiles_per_row) * TILE_H;
    int tile_ow = (tile_within % tiles_per_row) * TILE_W;
    int n = fm_id / out_channels;
    int oc = fm_id % out_channels;

    if (n >= batch_size) return;

    // 线程在 tile 内的坐标
    int t = threadIdx.x; // 0..255
    int ty = t / TILE_W;
    int tx = t % TILE_W;

    // 初始化 sum 为 bias
    float sum = __ldg(&bias[oc]);

    // 对每个输入通道分阶段加载共享内存并累加
    for (int ic = 0; ic < in_channels; ++ic) {
        const int input_c_base = n * in_channels * input_h * input_w + ic * input_h * input_w;
        // 协作加载 smem (sm_h * sm_w)
        for (int load_idx = threadIdx.x; load_idx < sm_h * sm_w; load_idx += blockDim.x) {
            int sy = load_idx / sm_w;
            int sx = load_idx % sm_w;
            int ih = tile_oh + sy;
            int iw = tile_ow + sx;
            float val = 0.0f;
            if (ih < input_h && iw < input_w) {
                int gidx = input_c_base + ih * input_w + iw;
                val = __ldg(&input[gidx]);
            }
            smem[sy * sm_w + sx] = val;
        }
        __syncthreads();

        // 仅对有效输出位置进行卷积累加
        int oh = tile_oh + ty;
        int ow = tile_ow + tx;
        if (ty < TILE_H && tx < TILE_W && oh < output_h && ow < output_w) {
            const int weight_c_base = oc * in_channels * 25 + ic * 25;
            #pragma unroll
            for (int kh = 0; kh < 5; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 5; ++kw) {
                    float a = smem[(ty + kh) * sm_w + (tx + kw)];
                    float w = __ldg(&weight[weight_c_base + kh * 5 + kw]);
                    sum = fmaf(a, w, sum);
                }
            }
        }
        __syncthreads();
    }

    // 写回 IF 输出与电压
    int oh = tile_oh + ty;
    int ow = tile_ow + tx;
    if (ty < TILE_H && tx < TILE_W && oh < output_h && ow < output_w) {
        int out_idx = n * out_channels * output_h * output_w + oc * output_h * output_w + oh * output_w + ow;
        float v = voltage[out_idx] + sum;
        float spk = (v >= threshold) ? 1.0f : 0.0f;
        if (spk >= 1.0f) v = v_reset;
        voltage[out_idx] = v;
        output_spike[out_idx] = spk;
    }
}

// 脉冲累加核函数
__global__ void spike_accumulate_kernel(float* current_spikes, float* spike_sum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        spike_sum[idx] += current_spikes[idx];
    }
}

// 脉冲频率计算核函数（使用预计算倒数，避免除法）
__global__ void spike_frequency_kernel(const float* __restrict__ spike_sum, float* __restrict__ output, int size, float invT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = spike_sum[idx] * invT;
    }
}

// 前向声明：卷积 + IF + 2x2池化 融合核（供后文调用）
__global__ void conv2d_if_pool2x2_kernel(const float* __restrict__ input,
                                        float* __restrict__ pooled_out,
                                        float* __restrict__ voltage,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        int batch_size, int in_channels, int out_channels,
                                        int input_h, int input_w,
                                        int kernel_size,
                                        int conv_out_h, int conv_out_w,
                                        int pool_out_h, int pool_out_w,
                                        float threshold, float v_reset);

// 使用常量内存权重/偏置的 卷积 + IF + 2x2池化 融合核（which_layer: 1=conv1, 2=conv2）
__global__ void conv2d_if_pool2x2_kernel_constw(const float* __restrict__ input,
                                               float* __restrict__ pooled_out,
                                               float* __restrict__ voltage,
                                               int batch_size, int in_channels, int out_channels,
                                               int input_h, int input_w,
                                               int kernel_size,
                                               int conv_out_h, int conv_out_w,
                                               int pool_out_h, int pool_out_w,
                                               float threshold, float v_reset,
                                               int which_layer);

// U8 variants forward declarations
__global__ void conv1_if_pool_from_sum_u8_kernel(const float* conv_sum,
                                                unsigned char* pooled_out_u8,
                                                float* voltage,
                                                int batch_size,
                                                int conv_out_h, int conv_out_w,
                                                int pool_out_h, int pool_out_w,
                                                float threshold, float v_reset);
__global__ void conv2d_if_pool2x2_u8_kernel(const unsigned char* __restrict__ input_u8,
                                           float* __restrict__ pooled_out,
                                           float* __restrict__ voltage,
                                           const float* __restrict__ weight,
                                           const float* __restrict__ bias,
                                           int batch_size, int in_channels, int out_channels,
                                           int input_h, int input_w,
                                           int kernel_size,
                                           int conv_out_h, int conv_out_w,
                                           int pool_out_h, int pool_out_w,
                                           float threshold, float v_reset);

// 共享内存 5x5 卷积 + IF + 2x2 池化（用于 Conv2），以 pooled 输出 tile 为单位
__global__ void conv2d_if_pool2x2_smem5x5_tiled_kernel(const float* __restrict__ input,
                                                      float* __restrict__ pooled_out,
                                                      float* __restrict__ voltage,
                                                      const float* __restrict__ weight,
                                                      const float* __restrict__ bias,
                                                      int batch_size, int in_channels, int out_channels,
                                                      int input_h, int input_w,
                                                      int conv_out_h, int conv_out_w,
                                                      int pool_out_h, int pool_out_w,
                                                      float threshold, float v_reset) {
    const int TILE_PH = 16; // pooled tile 高度
    const int TILE_PW = 16; // pooled tile 宽度
    const int TILE_CH = TILE_PH * 2; // 对应 conv 输出 tile 高度
    const int TILE_CW = TILE_PW * 2; // 对应 conv 输出 tile 宽度
    const int SM_H = TILE_CH + 4; // + (K-1)
    const int SM_W = TILE_CW + 4;

    extern __shared__ float smem[]; // SM_H * SM_W

    // block 按 (n, oc, pooled tile) 映射
    int tiles_per_row = (pool_out_w + TILE_PW - 1) / TILE_PW;
    int tiles_per_col = (pool_out_h + TILE_PH - 1) / TILE_PH;
    int tiles_per_fm = tiles_per_row * tiles_per_col;

    int tile_id = blockIdx.x;
    int fm_id = tile_id / tiles_per_fm; // 0 .. batch*out_channels-1
    int tile_within = tile_id % tiles_per_fm;
    int ph0 = (tile_within / tiles_per_row) * TILE_PH; // pooled 起点 h
    int pw0 = (tile_within % tiles_per_row) * TILE_PW; // pooled 起点 w
    int n = fm_id / out_channels;
    int oc = fm_id % out_channels;
    if (n >= batch_size) return;

    int t = threadIdx.x;
    int ty = t / TILE_PW; // 在 pooled tile 中的 y（最多 16）
    int tx = t % TILE_PW; // 在 pooled tile 中的 x（最多 16）

    // 累加 2x2 卷积分支的最大脉冲
    float max_spk = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        int input_c_base = n * in_channels * input_h * input_w + ic * input_h * input_w;

        // 协作加载对应的 conv 输入 tile 至共享内存（覆盖 2x2 卷积窗口导致的边界）
        for (int load_idx = threadIdx.x; load_idx < SM_H * SM_W; load_idx += blockDim.x) {
            int sy = load_idx / SM_W;
            int sx = load_idx % SM_W;
            int ih = ph0 * 2 + sy; // pooled->conv
            int iw = pw0 * 2 + sx;
            float v = 0.0f;
            if (ih < input_h && iw < input_w) {
                int gidx = input_c_base + ih * input_w + iw;
                v = __ldg(&input[gidx]);
            }
            smem[sy * SM_W + sx] = v;
        }
        __syncthreads();

        if (ty < TILE_PH && tx < TILE_PW) {
            int ph = ph0 + ty;
            int pw = pw0 + tx;
            if (ph < pool_out_h && pw < pool_out_w) {
                // 2x2 分支
                float local_max = 0.0f;
                #pragma unroll
                for (int dy = 0; dy < 2; ++dy) {
                    #pragma unroll
                    for (int dx = 0; dx < 2; ++dx) {
                        int oh = ty * 2 + dy; // conv tile 内坐标
                        int ow = tx * 2 + dx;
                        float sum = __ldg(&bias[oc]);
                        int w_base = oc * in_channels * 25 + ic * 25;
                        #pragma unroll
                        for (int kh = 0; kh < 5; ++kh) {
                            #pragma unroll
                            for (int kw = 0; kw < 5; ++kw) {
                                float a = smem[(oh + kh) * SM_W + (ow + kw)];
                                float w = __ldg(&weight[w_base + kh * 5 + kw]);
                                sum = fmaf(a, w, sum);
                            }
                        }
                        int oh_g = ph * 2 + dy;
                        int ow_g = pw * 2 + dx;
                        int conv_linear = n * out_channels * conv_out_h * conv_out_w + oc * conv_out_h * conv_out_w + oh_g * conv_out_w + ow_g;
                        float v = voltage[conv_linear] + sum;
                        float spk = (v >= threshold) ? 1.0f : 0.0f;
                        if (spk >= 1.0f) v = v_reset;
                        voltage[conv_linear] = v;
                        local_max = fmaxf(local_max, spk);
                    }
                }
                // 累加该 ic 分支的最大值
                max_spk = fmaxf(max_spk, local_max);
            }
        }
        __syncthreads();
    }

    // 写回 pooled 输出
    if (ty < TILE_PH && tx < TILE_PW) {
        int ph = ph0 + ty;
        int pw = pw0 + tx;
        if (ph < pool_out_h && pw < pool_out_w) {
            int out_idx = n * out_channels * pool_out_h * pool_out_w + oc * pool_out_h * pool_out_w + ph * pool_out_w + pw;
            pooled_out[out_idx] = max_spk;
        }
    }
}

// 预计算 conv1 线性和（含 bias）的前向声明
__global__ void conv1_linear_sum_kernel(const float* input,
                                       float* conv_sum,
                                       const float* weight,
                                       const float* bias,
                                       int batch_size,
                                       int input_h, int input_w,
                                       int conv_out_h, int conv_out_w);

// 从预计算和执行 IF+2x2 池化的前向声明
__global__ void conv1_if_pool_from_sum_kernel(const float* conv_sum,
                                             float* pooled_out,
                                             float* voltage,
                                             int batch_size,
                                             int conv_out_h, int conv_out_w,
                                             int pool_out_h, int pool_out_w,
                                             float threshold, float v_reset);

// ===================================================================================
// CUDA Kernel Functions
// ===================================================================================

std::vector<int> scnn_inference(
    const std::vector<std::vector<float>>& images,
    // Device pointers for parameters
    float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b,
    float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
    float* d_fc3_w,   float* d_fc3_b,
    int max_batch_in   // YOU CAN ADD MORE PARAMETERS HERE!!!
    )
{
    std::vector<int> predictions;
    const int num_images = images.size();
    predictions.reserve(num_images);

    // SNN-specific parameter, must match training
    const int T = 8;
    
    // 网络参数
    const int input_h = 28, input_w = 28;
    const int conv1_out_channels = 6;
    const int conv1_out_h = 24, conv1_out_w = 24; // (28-5+1)
    const int pool1_out_h = 12, pool1_out_w = 12; // 24/2
    const int conv2_out_channels = 16;
    const int conv2_out_h = 8, conv2_out_w = 8; // (12-5+1)
    const int pool2_out_h = 4, pool2_out_w = 4; // 8/2
    const int fc1_out_features = 120;
    const int fc2_out_features = 84;
    const int fc3_out_features = 10;
    
    // 设定最大批大小并分配设备内存（按最大批大小复用缓冲）
    // 从命令行传入的 max_batch_in 与样本总数取 min，避免越界
    const int max_batch = std::min(num_images, std::max(1, max_batch_in));
    float *d_input[2], *d_conv1_out, *d_pool1_out, *d_conv2_out, *d_pool2_out;
    float *d_fc1_out, *d_fc2_out, *d_fc3_out, *d_spike_sum, *d_final_out[2];
    unsigned char* d_pool1_out_u8; // 第一层池化输出 U8（0/1）
    float *d_voltage1, *d_voltage2, *d_voltage3, *d_voltage4;
    float *d_conv1_sum; // 预计算 conv1 线性和
    
    size_t input_size = max_batch * input_h * input_w * sizeof(float);
    size_t conv1_size = max_batch * conv1_out_channels * conv1_out_h * conv1_out_w * sizeof(float);
    size_t pool1_size = max_batch * conv1_out_channels * pool1_out_h * pool1_out_w * sizeof(float);
    size_t conv2_size = max_batch * conv2_out_channels * conv2_out_h * conv2_out_w * sizeof(float);
    size_t pool2_size = max_batch * conv2_out_channels * pool2_out_h * pool2_out_w * sizeof(float);
    size_t fc1_size = max_batch * fc1_out_features * sizeof(float);
    size_t fc2_size = max_batch * fc2_out_features * sizeof(float);
    size_t fc3_size = max_batch * fc3_out_features * sizeof(float);
    
    checkCudaErrors(cudaMalloc(&d_input[0], input_size));
    checkCudaErrors(cudaMalloc(&d_input[1], input_size));
    checkCudaErrors(cudaMalloc(&d_conv1_out, conv1_size));
    checkCudaErrors(cudaMalloc(&d_pool1_out, pool1_size));
    checkCudaErrors(cudaMalloc(&d_pool1_out_u8, max_batch * conv1_out_channels * pool1_out_h * pool1_out_w * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_conv2_out, conv2_size));
    checkCudaErrors(cudaMalloc(&d_pool2_out, pool2_size));
    checkCudaErrors(cudaMalloc(&d_fc1_out, fc1_size));
    checkCudaErrors(cudaMalloc(&d_fc2_out, fc2_size));
    checkCudaErrors(cudaMalloc(&d_fc3_out, fc3_size));
    checkCudaErrors(cudaMalloc(&d_spike_sum, fc3_size));
    checkCudaErrors(cudaMalloc(&d_final_out[0], fc3_size));
    checkCudaErrors(cudaMalloc(&d_final_out[1], fc3_size));
    checkCudaErrors(cudaMalloc(&d_conv1_sum, max_batch * conv1_out_channels * conv1_out_h * conv1_out_w * sizeof(float)));
    
    // 分配电压状态内存（按最大批大小），每批重置
    checkCudaErrors(cudaMalloc(&d_voltage1, conv1_size));
    checkCudaErrors(cudaMalloc(&d_voltage2, conv2_size));
    checkCudaErrors(cudaMalloc(&d_voltage3, fc1_size));
    checkCudaErrors(cudaMalloc(&d_voltage4, fc2_size));
    
    // 设置CUDA块大小（卷积/FC可分开控制），默认卷积=384，FC=256；
    // 环境变量：BLOCK_SIZE（全局），BLOCK_SIZE_CONV，BLOCK_SIZE_FC（覆盖）
    auto parse_block = [](const char* env_name, int def)->int{
        int v = def;
        if (const char* e = std::getenv(env_name)) {
            int t = std::atoi(e);
            if (t >= 64 && t <= 1024) v = ((t + 31) / 32) * 32;
        }
        return v;
    };
    int global_bs = parse_block("BLOCK_SIZE", 0);
    // 默认使用经过实测较优的块大小：Conv=224, FC=192（环境变量仍可覆盖）
    int conv_bs = global_bs ? global_bs : parse_block("BLOCK_SIZE_CONV", 224);
    int fc_bs   = global_bs ? global_bs : parse_block("BLOCK_SIZE_FC",   192);
    dim3 block_size_conv(conv_bs);
    dim3 block_size_fc(fc_bs);
    
    // 是否启用常量内存卷积权重（默认关闭，避免索引分歧导致的常量缓存串行）
    const bool use_const_conv = false;
    if (use_const_conv) {
        checkCudaErrors(cudaMemcpyToSymbol(c_conv1_w, d_conv1_w, 6 * 1 * 5 * 5 * sizeof(float), 0, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(c_conv1_b, d_conv1_b, 6 * sizeof(float), 0, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(c_conv2_w, d_conv2_w, 16 * 6 * 5 * 5 * sizeof(float), 0, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(c_conv2_b, d_conv2_b, 16 * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    }

    // 设置共享内存偏好
    checkCudaErrors(cudaFuncSetCacheConfig(conv2d_if_smem5x5_tiled_kernel, cudaFuncCachePreferShared));
    checkCudaErrors(cudaFuncSetCacheConfig(conv2d_if_smem5x5_tiled_kernel_constw, cudaFuncCachePreferShared));
    checkCudaErrors(cudaFuncSetCacheConfig(conv2d_if_pool2x2_kernel_constw, cudaFuncCachePreferShared));
    
    // 分批推理（为减少多次 kernel 启动开销，引入每批次 CUDA Graph 捕获）
    // 创建流与图（双缓冲）
    cudaStream_t compute_stream, copy_stream;
    checkCudaErrors(cudaStreamCreate(&compute_stream));
    checkCudaErrors(cudaStreamCreate(&copy_stream));
    cudaGraph_t graph[2] = {nullptr, nullptr};
    cudaGraphExec_t graph_exec[2] = {nullptr, nullptr};
    int graph_batch_size[2] = {-1, -1};
    cudaEvent_t h2d_done[2];
    checkCudaErrors(cudaEventCreateWithFlags(&h2d_done[0], cudaEventDisableTiming));
    checkCudaErrors(cudaEventCreateWithFlags(&h2d_done[1], cudaEventDisableTiming));

    // 固定内存 host 缓冲
    float* h_batch_in[2];
    float* h_batch_out[2];
    checkCudaErrors(cudaMallocHost(&h_batch_in[0], input_size));
    checkCudaErrors(cudaMallocHost(&h_batch_in[1], input_size));
    checkCudaErrors(cudaMallocHost(&h_batch_out[0], fc3_size));
    checkCudaErrors(cudaMallocHost(&h_batch_out[1], fc3_size));

    int num_batches = (num_images + max_batch - 1) / max_batch;
    // 预取第0批
    if (num_batches > 0) {
        int bs = std::min(max_batch, num_images);
        for (int i = 0; i < bs; ++i) {
            const auto& img = images[i];
            std::memcpy(h_batch_in[0] + i * (input_h * input_w), img.data(), input_h * input_w * sizeof(float));
        }
        checkCudaErrors(cudaMemcpyAsync(d_input[0], h_batch_in[0], bs * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
        checkCudaErrors(cudaEventRecord(h2d_done[0], copy_stream));
    }

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        int cur = batch_idx & 1;
        int nxt = (batch_idx + 1) & 1;
        int batch_start = batch_idx * max_batch;
        int current_batch = std::min(max_batch, num_images - batch_start);

        // 等待当前缓冲 H2D 完成
        checkCudaErrors(cudaStreamWaitEvent(compute_stream, h2d_done[cur], 0));

        // 重置电压与累加缓冲
        checkCudaErrors(cudaMemsetAsync(d_voltage1, 0, current_batch * conv1_out_channels * conv1_out_h * conv1_out_w * sizeof(float), compute_stream));
        checkCudaErrors(cudaMemsetAsync(d_voltage2, 0, current_batch * conv2_out_channels * conv2_out_h * conv2_out_w * sizeof(float), compute_stream));
        checkCudaErrors(cudaMemsetAsync(d_voltage3, 0, current_batch * fc1_out_features * sizeof(float), compute_stream));
        checkCudaErrors(cudaMemsetAsync(d_voltage4, 0, current_batch * fc2_out_features * sizeof(float), compute_stream));
        checkCudaErrors(cudaMemsetAsync(d_spike_sum, 0, current_batch * fc3_out_features * sizeof(float), compute_stream));

        // 设置网格
        dim3 conv1_grid((current_batch * conv1_out_channels * pool1_out_h * pool1_out_w + block_size_conv.x - 1) / block_size_conv.x);
        // Conv2 共享内存 + 线程粗化实现：一个 block 处理一个样本，固定 256 线程
        dim3 conv2_grid(current_batch);
        dim3 block_size_conv2(256);
        dim3 fc1_grid((current_batch * fc1_out_features + block_size_fc.x - 1) / block_size_fc.x);
        dim3 fc2_grid((current_batch * fc2_out_features + block_size_fc.x - 1) / block_size_fc.x);
        dim3 fc3_grid((current_batch * fc3_out_features + block_size_fc.x - 1) / block_size_fc.x);
        dim3 conv1_sum_grid((current_batch * conv1_out_channels * conv1_out_h * conv1_out_w + block_size_conv.x - 1) / block_size_conv.x);

        // 构建/复用图（按 cur 缓冲）
        if (graph_exec[cur] == nullptr || graph_batch_size[cur] != current_batch) {
            if (graph_exec[cur] != nullptr) {
                checkCudaErrors(cudaGraphExecDestroy(graph_exec[cur]));
                checkCudaErrors(cudaGraphDestroy(graph[cur]));
                graph_exec[cur] = nullptr;
                graph[cur] = nullptr;
            }
            checkCudaErrors(cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeGlobal));

            conv1_linear_sum_kernel<<<conv1_sum_grid, block_size_conv, 0, compute_stream>>>(
                d_input[cur],
                d_conv1_sum,
                d_conv1_w, d_conv1_b,
                current_batch,
                input_h, input_w,
                conv1_out_h, conv1_out_w);

            for (int t = 0; t < T; ++t) {
                // 第一层输出写 U8（0/1），降低第二层带宽并启用稀疏乘加
                conv1_if_pool_from_sum_u8_kernel<<<conv1_grid, block_size_conv, 0, compute_stream>>>(
                    d_conv1_sum, d_pool1_out_u8, d_voltage1,
                    current_batch,
                    conv1_out_h, conv1_out_w,
                    pool1_out_h, pool1_out_w,
                    1.0f, 0.0f);

                // Conv2 使用 U8 稀疏输入 + 共享内存加速
                conv2d_if_pool2x2_u8_kernel<<<conv2_grid, block_size_conv2, 0, compute_stream>>>(
                    d_pool1_out_u8, d_pool2_out, d_voltage2,
                    d_conv2_w, d_conv2_b,
                    current_batch, conv1_out_channels, conv2_out_channels,
                    pool1_out_h, pool1_out_w,
                    5,
                    conv2_out_h, conv2_out_w,
                    pool2_out_h, pool2_out_w,
                    1.0f, 0.0f);

                linear_if_kernel<<<fc1_grid, block_size_fc, 0, compute_stream>>>(
                    d_pool2_out, d_fc1_out, d_voltage3,
                    d_fc1_w, d_fc1_b,
                    current_batch, conv2_out_channels * pool2_out_h * pool2_out_w, fc1_out_features,
                    1.0f, 0.0f);

                linear_if_kernel<<<fc2_grid, block_size_fc, 0, compute_stream>>>(
                    d_fc1_out, d_fc2_out, d_voltage4,
                    d_fc2_w, d_fc2_b,
                    current_batch, fc1_out_features, fc2_out_features,
                    1.0f, 0.0f);

                linear_accumulate_kernel<<<fc3_grid, block_size_fc, 0, compute_stream>>>(
                    d_fc2_out, d_fc3_out, d_fc3_w, d_fc3_b,
                    d_spike_sum,
                    current_batch, fc2_out_features, fc3_out_features);
            }

            spike_frequency_kernel<<<fc3_grid, block_size_fc, 0, compute_stream>>>(
                d_spike_sum, d_final_out[cur],
                current_batch * fc3_out_features, 1.0f / static_cast<float>(T));

            checkCudaErrors(cudaStreamEndCapture(compute_stream, &graph[cur]));
            checkCudaErrors(cudaGraphInstantiate(&graph_exec[cur], graph[cur], nullptr, nullptr, 0));
            graph_batch_size[cur] = current_batch;
        }

        // 启动计算
        checkCudaErrors(cudaGraphLaunch(graph_exec[cur], compute_stream));

        // 预取下一批 H2D
        if (batch_idx + 1 < num_batches) {
            int next_start = (batch_idx + 1) * max_batch;
            int next_batch = std::min(max_batch, num_images - next_start);
            for (int i = 0; i < next_batch; ++i) {
                const auto& img = images[next_start + i];
                std::memcpy(h_batch_in[nxt] + i * (input_h * input_w), img.data(), input_h * input_w * sizeof(float));
            }
            checkCudaErrors(cudaMemcpyAsync(d_input[nxt], h_batch_in[nxt], next_batch * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
            checkCudaErrors(cudaEventRecord(h2d_done[nxt], copy_stream));
        }

        // 等待计算完成并回传结果
        checkCudaErrors(cudaStreamSynchronize(compute_stream));
        std::vector<float> batch_final_output(static_cast<size_t>(current_batch) * fc3_out_features);
        checkCudaErrors(cudaMemcpyAsync(batch_final_output.data(), d_final_out[cur], current_batch * fc3_out_features * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
        checkCudaErrors(cudaStreamSynchronize(copy_stream));
        for (int i = 0; i < current_batch; ++i) {
            int max_idx = 0;
            float max_val = batch_final_output[i * fc3_out_features];
            for (int j = 1; j < fc3_out_features; ++j) {
                float v = batch_final_output[i * fc3_out_features + j];
                if (v > max_val) { max_val = v; max_idx = j; }
            }
            predictions.push_back(max_idx);
        }
    }

    // 清理资源
    if (graph_exec[0] != nullptr) checkCudaErrors(cudaGraphExecDestroy(graph_exec[0]));
    if (graph_exec[1] != nullptr) checkCudaErrors(cudaGraphExecDestroy(graph_exec[1]));
    if (graph[0] != nullptr) checkCudaErrors(cudaGraphDestroy(graph[0]));
    if (graph[1] != nullptr) checkCudaErrors(cudaGraphDestroy(graph[1]));
    checkCudaErrors(cudaEventDestroy(h2d_done[0]));
    checkCudaErrors(cudaEventDestroy(h2d_done[1]));
    checkCudaErrors(cudaStreamDestroy(copy_stream));
    checkCudaErrors(cudaStreamDestroy(compute_stream));
    checkCudaErrors(cudaFreeHost(h_batch_in[0]));
    checkCudaErrors(cudaFreeHost(h_batch_in[1]));
    checkCudaErrors(cudaFreeHost(h_batch_out[0]));
    checkCudaErrors(cudaFreeHost(h_batch_out[1]));
    
    // 释放设备内存
    checkCudaErrors(cudaFree(d_input[0]));
    checkCudaErrors(cudaFree(d_input[1]));
    checkCudaErrors(cudaFree(d_conv1_out));
    checkCudaErrors(cudaFree(d_pool1_out));
    checkCudaErrors(cudaFree(d_pool1_out_u8));
    checkCudaErrors(cudaFree(d_conv2_out));
    checkCudaErrors(cudaFree(d_pool2_out));
    checkCudaErrors(cudaFree(d_fc1_out));
    checkCudaErrors(cudaFree(d_fc2_out));
    checkCudaErrors(cudaFree(d_fc3_out));
    checkCudaErrors(cudaFree(d_spike_sum));
    checkCudaErrors(cudaFree(d_final_out[0]));
    checkCudaErrors(cudaFree(d_final_out[1]));
    checkCudaErrors(cudaFree(d_conv1_sum));
    checkCudaErrors(cudaFree(d_voltage1));
    checkCudaErrors(cudaFree(d_voltage2));
    checkCudaErrors(cudaFree(d_voltage3));
    checkCudaErrors(cudaFree(d_voltage4));
    
    return predictions;
}

// 卷积 + IF + 2x2池化（stride=2）融合核（valid 卷积），避免中间特征图往返
__global__ void conv2d_if_pool2x2_kernel(const float* __restrict__ input,
                                        float* __restrict__ pooled_out,
                                        float* __restrict__ voltage,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        int batch_size, int in_channels, int out_channels,
                                        int input_h, int input_w,
                                        int kernel_size,
                                        int conv_out_h, int conv_out_w,
                                        int pool_out_h, int pool_out_w,
                                        float threshold, float v_reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * pool_out_h * pool_out_w;
    if (idx >= total) return;

    int n = idx / (out_channels * pool_out_h * pool_out_w);
    int oc = (idx % (out_channels * pool_out_h * pool_out_w)) / (pool_out_h * pool_out_w);
    int ohp = (idx % (pool_out_h * pool_out_w)) / pool_out_w; // pooled h
    int owp = idx % pool_out_w; // pooled w

    int oh0 = (ohp << 1);
    int ow0 = (owp << 1);

    float max_spk = 0.0f;

    // 遍历 2x2 的四个卷积位置
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int oh = oh0 + dy;
            int ow = ow0 + dx;
            float sum = __ldg(&bias[oc]);

            for (int ic = 0; ic < in_channels; ++ic) {
                int input_c_base = n * in_channels * input_h * input_w + ic * input_h * input_w;
                int weight_c_base = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size;
                if (kernel_size == 5) {
                    #pragma unroll
                    for (int kh = 0; kh < 5; ++kh) {
                        int ih = oh + kh;
                        #pragma unroll
                        for (int kw = 0; kw < 5; ++kw) {
                            int iw = ow + kw;
                            int in_idx = input_c_base + ih * input_w + iw;
                            int w_idx  = weight_c_base + kh * 5 + kw;
                            sum = fmaf(__ldg(&input[in_idx]), __ldg(&weight[w_idx]), sum);
                        }
                    }
                } else {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        int ih = oh + kh;
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int iw = ow + kw;
                            int in_idx = input_c_base + ih * input_w + iw;
                            int w_idx  = weight_c_base + kh * kernel_size + kw;
                            sum = fmaf(__ldg(&input[in_idx]), __ldg(&weight[w_idx]), sum);
                        }
                    }
                }
            }

            // 更新该卷积位置对应的电压与脉冲
            int conv_linear = n * out_channels * conv_out_h * conv_out_w + oc * conv_out_h * conv_out_w + oh * conv_out_w + ow;
            float v = voltage[conv_linear] + sum;
            float spk = (v >= threshold) ? 1.0f : 0.0f;
            if (spk >= 1.0f) v = v_reset;
            voltage[conv_linear] = v;
            max_spk = fmaxf(max_spk, spk);
        }
    }

    // 写回池化后的输出（脉冲 0/1）
    pooled_out[idx] = max_spk;
}

// 使用常量内存权重/偏置的 卷积 + IF + 2x2池化 融合核
__global__ void conv2d_if_pool2x2_kernel_constw(const float* __restrict__ input,
                                               float* __restrict__ pooled_out,
                                               float* __restrict__ voltage,
                                               int batch_size, int in_channels, int out_channels,
                                               int input_h, int input_w,
                                               int kernel_size,
                                               int conv_out_h, int conv_out_w,
                                               int pool_out_h, int pool_out_w,
                                               float threshold, float v_reset,
                                               int which_layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * pool_out_h * pool_out_w;
    if (idx >= total) return;

    int n = idx / (out_channels * pool_out_h * pool_out_w);
    int oc = (idx % (out_channels * pool_out_h * pool_out_w)) / (pool_out_h * pool_out_w);
    int ohp = (idx % (pool_out_h * pool_out_w)) / pool_out_w; // pooled h
    int owp = idx % pool_out_w; // pooled w

    int oh0 = (ohp << 1);
    int ow0 = (owp << 1);

    float max_spk = 0.0f;

    // 遍历 2x2 的四个卷积位置
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int oh = oh0 + dy;
            int ow = ow0 + dx;
            float sum = (which_layer == 1) ? __ldg(&c_conv1_b[oc]) : __ldg(&c_conv2_b[oc]);

            for (int ic = 0; ic < in_channels; ++ic) {
                int input_c_base = n * in_channels * input_h * input_w + ic * input_h * input_w;
                if (kernel_size == 5) {
                    #pragma unroll
                    for (int kh = 0; kh < 5; ++kh) {
                        int ih = oh + kh;
                        #pragma unroll
                        for (int kw = 0; kw < 5; ++kw) {
                            int iw = ow + kw;
                            int in_idx = input_c_base + ih * input_w + iw;
                            int w_off  = oc * in_channels * 25 + ic * 25 + kh * 5 + kw;
                            float w = (which_layer == 1) ? __ldg(&c_conv1_w[w_off]) : __ldg(&c_conv2_w[w_off]);
                            sum = fmaf(__ldg(&input[in_idx]), w, sum);
                        }
                    }
                } else {
                    // kernel_size != 5 的通用路径（仍从常量内存按 5x5 偏置访问会出错，故仅支持 5x5）
                    // 这里保留结构，但不展开，若出现将导致结果未定义
                }
            }

            // 更新该卷积位置对应的电压与脉冲
            int conv_linear = n * out_channels * conv_out_h * conv_out_w + oc * conv_out_h * conv_out_w + oh * conv_out_w + ow;
            float v = voltage[conv_linear] + sum;
            float spk = (v >= threshold) ? 1.0f : 0.0f;
            if (spk >= 1.0f) v = v_reset;
            voltage[conv_linear] = v;
            max_spk = fmaxf(max_spk, spk);
        }
    }

    // 写回池化后的输出（脉冲 0/1）
    pooled_out[idx] = max_spk;
}

// 仅用于第一层：预计算 conv1 的线性卷积和（含 bias），不包括 IF 与池化
// 复用通用 conv2d_kernel 亦可，但单独声明可读性更好
__global__ void conv1_linear_sum_kernel(const float* __restrict__ input,
                                       float* __restrict__ conv_sum,
                                       const float* __restrict__ weight,
                                       const float* __restrict__ bias,
                                       int batch_size,
                                       int input_h, int input_w,
                                       int conv_out_h, int conv_out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * 6 * conv_out_h * conv_out_w; // out_channels=6, kernel=5x5, in_channels=1
    if (idx >= total) return;

    int n = idx / (6 * conv_out_h * conv_out_w);
    int oc = (idx % (6 * conv_out_h * conv_out_w)) / (conv_out_h * conv_out_w);
    int oh = (idx % (conv_out_h * conv_out_w)) / conv_out_w;
    int ow = idx % conv_out_w;

    float sum = __ldg(&bias[oc]);
    // in_channels == 1
    int input_c_base = n * input_h * input_w;
    int weight_c_base = oc * 25; // 5x5
    #pragma unroll
    for (int kh = 0; kh < 5; ++kh) {
        int ih = oh + kh;
        #pragma unroll
        for (int kw = 0; kw < 5; ++kw) {
            int iw = ow + kw;
            int in_idx = input_c_base + ih * input_w + iw;
            int w_idx  = weight_c_base + kh * 5 + kw;
            sum = fmaf(__ldg(&input[in_idx]), __ldg(&weight[w_idx]), sum);
        }
    }
    conv_sum[idx] = sum;
}

// 使用预计算 conv1 和的 IF+2x2池化核：每步仅做电压更新与阈值判定+池化
__global__ void conv1_if_pool_from_sum_kernel(const float* __restrict__ conv_sum,
                                             float* __restrict__ pooled_out,
                                             float* __restrict__ voltage,
                                             int batch_size,
                                             int conv_out_h, int conv_out_w,
                                             int pool_out_h, int pool_out_w,
                                             float threshold, float v_reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * 6 * pool_out_h * pool_out_w; // out_channels=6
    if (idx >= total) return;

    int n = idx / (6 * pool_out_h * pool_out_w);
    int oc = (idx % (6 * pool_out_h * pool_out_w)) / (pool_out_h * pool_out_w);
    int ohp = (idx % (pool_out_h * pool_out_w)) / pool_out_w;
    int owp = idx % pool_out_w;

    int oh0 = (ohp << 1);
    int ow0 = (owp << 1);

    float max_spk = 0.0f;

    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int oh = oh0 + dy;
            int ow = ow0 + dx;
            int conv_linear = n * 6 * conv_out_h * conv_out_w + oc * conv_out_h * conv_out_w + oh * conv_out_w + ow;
            float v = voltage[conv_linear] + __ldg(&conv_sum[conv_linear]);
            float spk = (v >= threshold) ? 1.0f : 0.0f;
            if (spk >= 1.0f) v = v_reset;
            voltage[conv_linear] = v;
            max_spk = fmaxf(max_spk, spk);
        }
    }
    pooled_out[idx] = max_spk;
}

// 写 U8 的变体：把池化脉冲写为 0/1 的 unsigned char，供下一层使用
__global__ void conv1_if_pool_from_sum_u8_kernel(const float* __restrict__ conv_sum,
                                                unsigned char* __restrict__ pooled_out_u8,
                                                float* __restrict__ voltage,
                                                int batch_size,
                                                int conv_out_h, int conv_out_w,
                                                int pool_out_h, int pool_out_w,
                                                float threshold, float v_reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * 6 * pool_out_h * pool_out_w; // out_channels=6
    if (idx >= total) return;

    int n = idx / (6 * pool_out_h * pool_out_w);
    int oc = (idx % (6 * pool_out_h * pool_out_w)) / (pool_out_h * pool_out_w);
    int ohp = (idx % (pool_out_h * pool_out_w)) / pool_out_w;
    int owp = idx % pool_out_w;

    int oh0 = (ohp << 1);
    int ow0 = (owp << 1);

    float max_spk = 0.0f;

    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int oh = oh0 + dy;
            int ow = ow0 + dx;
            int conv_linear = n * 6 * conv_out_h * conv_out_w + oc * conv_out_h * conv_out_w + oh * conv_out_w + ow;
            float v = voltage[conv_linear] + __ldg(&conv_sum[conv_linear]);
            float spk = (v >= threshold) ? 1.0f : 0.0f;
            if (spk >= 1.0f) v = v_reset;
            voltage[conv_linear] = v;
            max_spk = fmaxf(max_spk, spk);
        }
    }
    pooled_out_u8[idx] = static_cast<unsigned char>(max_spk >= 1.0f);
}

// 第二层：U8 输入（0/1）卷积 + IF + 2x2 池化（共享内存 + 线程粗化版本）
// 专门针对本模型的 Conv2 形状：in_channels=6, out_channels=16, input=12x12, kernel=5, conv_out=8x8, pool_out=4x4
// 一个 block 处理一个样本 n 的全部 16 个通道的 4x4 pooled 输出，每个线程计算一个 (oc, ph, pw) 的 2x2 conv patch
__global__ void conv2d_if_pool2x2_u8_kernel(const unsigned char* __restrict__ input_u8,
                                           float* __restrict__ pooled_out,
                                           float* __restrict__ voltage,
                                           const float* __restrict__ weight,
                                           const float* __restrict__ bias,
                                           int batch_size, int in_channels, int out_channels,
                                           int input_h, int input_w,
                                           int kernel_size,
                                           int conv_out_h, int conv_out_w,
                                           int pool_out_h, int pool_out_w,
                                           float threshold, float v_reset) {
    // 本核函数假定固定的 Conv2 形状，若不满足则直接返回（当前模型不会触发）
    if (in_channels != 6 || out_channels != 16 || input_h != 12 || input_w != 12 ||
        kernel_size != 5 || conv_out_h != 8 || conv_out_w != 8 ||
        pool_out_h != 4 || pool_out_w != 4) {
        return;
    }

    const int C2_IN  = 6;
    const int H2_IN  = 12;
    const int W2_IN  = 12;
    const int C2_OUT = 16;
    const int K      = 5;
    const int H2_OUT = 8;
    const int W2_OUT = 8;
    const int PH     = 4;
    const int PW     = 4;

    // 一个 block 处理一个样本
    int n = blockIdx.x;
    if (n >= batch_size) return;

    // 共享内存缓存整张输入特征图（6x12x12），从 U8 转为 float（0/1）
    __shared__ float s_in[C2_IN * H2_IN * W2_IN];

    int tid = threadIdx.x;
    int total_in = C2_IN * H2_IN * W2_IN; // 864
    int in_base = n * total_in;

    // 协作加载 U8 输入到共享内存（转为 float）
    for (int i = tid; i < total_in; i += blockDim.x) {
        unsigned char s = __ldg(&input_u8[in_base + i]);
        s_in[i] = static_cast<float>(s);
    }
    __syncthreads();

    // 需要的线程数：C2_OUT * PH * PW = 16 * 4 * 4 = 256
    int work_items = C2_OUT * PH * PW;
    if (tid >= work_items) return;

    int oc  = tid / (PH * PW);          // 0..15
    int tmp = tid % (PH * PW);          // 0..15
    int ph  = tmp / PW;                 // 0..3
    int pw  = tmp % PW;                 // 0..3

    int h_base = ph * 2;                // 对应 conv 输出的 2x2 patch 左上角
    int w_base = pw * 2;

    float max_spk = 0.0f;

    // 遍历该 pooled 单元对应的 2x2 卷积位置
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int oh = h_base + dy;
            int ow = w_base + dx;

            float sum = __ldg(&bias[oc]);

            // 卷积累加：6 个输入通道，5x5 kernel
            for (int ic = 0; ic < C2_IN; ++ic) {
                int s_in_offset      = ic * H2_IN * W2_IN;
                int weight_base_c_ic = oc * (C2_IN * K * K) + ic * (K * K);

                #pragma unroll
                for (int kh = 0; kh < K; ++kh) {
                    int h_in = oh + kh;
                    #pragma unroll
                    for (int kw = 0; kw < K; ++kw) {
                        int w_in = ow + kw;

                        int in_idx  = s_in_offset + h_in * W2_IN + w_in;
                        int w_idx   = weight_base_c_ic + kh * K + kw;
                        float a     = s_in[in_idx];                  // 0 或 1
                        float w_val = __ldg(&weight[w_idx]);
                        sum = fmaf(a, w_val, sum);
                    }
                }
            }

            // IF 电压与脉冲发放
            int conv_linear = n * out_channels * conv_out_h * conv_out_w
                            + oc * conv_out_h * conv_out_w
                            + oh * conv_out_w + ow;
            float v = voltage[conv_linear] + sum;
            float spk = (v >= threshold) ? 1.0f : 0.0f;
            if (spk >= 1.0f) v = v_reset;
            voltage[conv_linear] = v;
            max_spk = fmaxf(max_spk, spk);
        }
    }

    // 写回池化后的输出（脉冲 0/1）
    int out_idx = n * out_channels * pool_out_h * pool_out_w
                + oc * pool_out_h * pool_out_w
                + ph * pool_out_w + pw;
    pooled_out[out_idx] = max_spk;
}

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" << std::endl;
        return 1;
    }
	std::string dir = argv[1];
	
    // Load test data
    auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty()) return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");
    
    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float)));

    // --- 2. Copy constant parameters from host to device ---
    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================

    // --- 3. Perform inference ---
    // 可选命令行参数：argv[2] 作为 max_batch，默认使用实测最快的 640
    int max_batch_cli = 640;
    if (argc >= 3) {
        int v = std::atoi(argv[2]);
        if (v > 0) max_batch_cli = v;
    }
    // Pass device pointers to the inference function
    std::vector<int> predictions = scnn_inference(images,
        d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b,
        max_batch_cli
        // YOU CAN ADD MORE PARAMETERS HERE!!!
        );
    
// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // --- 4. Free all allocated GPU memory ---
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));
    
    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();
    
    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    
    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================
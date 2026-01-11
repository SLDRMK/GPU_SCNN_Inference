# program1_scnn_train.py

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR


from spikingjelly.activation_based import neuron, functional, layer


LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 40
# 推理侧已将时间步设为 4，这里保持一致
T_TIMESTEPS = 2


class SCNN(nn.Module):
    def __init__(self, T: int):
        super(SCNN, self).__init__()
        self.T = T

        self.conv1 = layer.Conv2d(1, 16, 5)
        self.if1 = neuron.IFNode()
        self.pool1 = layer.MaxPool2d(2, 2)

        self.conv2 = layer.Conv2d(16, 32, 5)
        self.if2 = neuron.IFNode()
        self.pool2 = layer.MaxPool2d(2, 2)

        self.flatten = layer.Flatten()
        
        self.fc1 = layer.Linear(32 * 4 * 4, 128)
        self.if3 = neuron.IFNode()

        self.fc2 = layer.Linear(128, 64)
        self.if4 = neuron.IFNode()

        self.fc3 = layer.Linear(64, 10)

    def forward(self, x: torch.Tensor):
        outputs = []
        for t in range(self.T):
            y = self.conv1(x)
            y = self.if1(y)
            y = self.pool1(y)
            y = self.conv2(y)
            y = self.if2(y)
            y = self.pool2(y)
            y = self.flatten(y)
            y = self.fc1(y)
            y = self.if3(y)
            y = self.fc2(y)
            y = self.if4(y)
            y = self.fc3(y)
            outputs.append(y)
        
        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(0)

script_dir = os.path.dirname(os.path.abspath(__file__))


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# 如果本地已将原始文件放在 data/fashion，则在离线场景下把它们放置到 torchvision 期望的位置 data/FashionMNIST/raw
fashion_source_raw = os.path.join(data_dir, 'fashion')
fashion_expected_raw = os.path.join(data_dir, 'FashionMNIST', 'raw')
required_files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
]
if os.path.isdir(fashion_source_raw):
    os.makedirs(fashion_expected_raw, exist_ok=True)
    for fname in required_files:
        src = os.path.join(fashion_source_raw, fname)
        dst = os.path.join(fashion_expected_raw, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)

# 数据已经存在于本地 data/FashionMNIST/raw 目录下，启用 download 以触发离线解压与处理（不会联网）
trainset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=True, transform=train_transform)
testset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=False, transform=test_transform)

# 设备选择：优先 Apple MPS，其次 CUDA，最后 CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# DataLoader 参数：为避免 macOS 多进程启动问题，这里统一使用单进程
num_workers = 0
pin_memory = (device == 'cuda')

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False
)
model = SCNN(T=T_TIMESTEPS).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

criterion = nn.MSELoss(reduction='mean')


print("--- Starting SCNN Training (Tuned for Convergence) ---")
max_accuracy = 0.0
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        functional.reset_net(model)
        
        outputs = model(inputs)
        target_one_hot = F.one_hot(labels, num_classes=10).to(dtype=outputs.dtype)
        loss = criterion(outputs, target_one_hot)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(trainloader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    scheduler.step()


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            functional.reset_net(model)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f} %')
    

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        print(f'New best accuracy: {max_accuracy:.2f} %. Saving model parameters...')
        output_dir = os.path.join(script_dir)
        os.makedirs(output_dir, exist_ok=True)
        for name, param in model.named_parameters():
            np.savetxt(os.path.join(output_dir, f'{name}.txt'), param.detach().cpu().numpy().flatten())

print('--- Finished Training ---')
print(f'Best accuracy achieved: {max_accuracy:.2f} %')
print("--- Final model parameters have been exported. ---")
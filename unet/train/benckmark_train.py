import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from unet.model.benchmark import UNet
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt  # 添加matplotlib库以绘制图表

# 自定义数据集
class DustDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [img for img in os.listdir(img_dir) if img != "desktop.ini"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 灰度图

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()  # 二值化处理
        return image, mask

# 训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
epochs = 100
learning_rate = 1e-4
img_size = 256

# 数据增强
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# 数据集路径（需根据实际修改）
train_dataset = DustDataset(
    img_dir="../../datasets/train/images",
    mask_dir="../../datasets/train/masks",
    transform=transform
)
val_dataset = DustDataset(
    img_dir="../../datasets/val/images",
    mask_dir="../../datasets/val/masks",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 初始化模型、损失函数和优化器
model = UNet(n_channels=3).to(device)
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 初始化记录损失和准确率的列表
train_losses = []
val_losses = []
accuracies = []

# 训练循环
best_val_loss = float("inf")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        predicted = (outputs > 0.5).float()
        total += masks.numel()
        correct += (predicted == masks).sum().item()

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

    # 统计损失和准确率
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    accuracy = correct / total

    print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")

    # 记录损失和准确率
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    accuracies.append(accuracy)

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

# 绘制损失和准确率曲线
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

print("训练完成！")

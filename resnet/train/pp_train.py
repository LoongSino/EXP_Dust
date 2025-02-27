

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import warnings
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import matplotlib

from resnet.model.pyramid_pooling import resnet50_pp


# 忽略弃用警告
warnings.filterwarnings("ignore", category=UserWarning)

# 设置随机种子以保证结果的可复现性
torch.manual_seed(42)

# 定义设备：GPU或CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
data_dir = '../../datasets_000'  # 替换为你的数据集路径
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=8) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# 加载预训练的ResNet50模型
model = resnet50_pp(num_classes=4)

# # 冻结所有层（可选）
# for param in model.parameters():
#     param.requires_grad = False

# 修改最后一层以适应你的分类任务
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(class_names))
num_cif = model.classifier.in_features
model.classifier = nn.Linear(num_cif, len(class_names))

# 将模型移动到GPU上（如果有GPU的话）
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.fc.parameters(), lr=0.002)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.002)


# 训练模型的函数
def train_model(model, criterion, optimizer, num_epochs):
    best_acc = 0.0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零参数梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只在训练阶段进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录损失和准确率
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 绘制损失收敛曲线
    plt.figure(figsize=(10, 5))
    plt.title("训练和验证损失")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(range(0, num_epochs), train_loss_history, label="训练损失")
    plt.plot(range(0, num_epochs), val_loss_history, label="验证损失")
    plt.xticks(range(0, num_epochs, 10))
    plt.legend()

    plt.tight_layout()

    # 保存损失图
    plt.savefig('../performance/pp_loss.png')
    plt.show()

    # 绘制准确度图
    plt.figure(figsize=(10, 5))
    plt.title("训练和验证准确率")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")

    # 将训练和验证准确率从GPU移动到CPU并转换为NumPy数组
    train_acc_history = [acc.cpu().numpy() for acc in train_acc_history]
    val_acc_history = [acc.cpu().numpy() for acc in val_acc_history]

    plt.plot(range(0, num_epochs), train_acc_history, label="训练准确率")
    plt.plot(range(0, num_epochs), val_acc_history, label="验证准确率")
    plt.xticks(range(0, num_epochs, 10))
    plt.legend()

    plt.tight_layout()

    # 保存准确率图
    plt.savefig('../performance/pp_acc.png')
    plt.show()

    return model


# 开始训练模型
if __name__ == '__main__':
    freeze_support()
    model_trained = train_model(model, criterion, optimizer, num_epochs=100)

    # 保存训练好的模型
    torch.save(model_trained.state_dict(), '../checkpoints/resnet50_pp_dust.pth')

    print('train complete.')

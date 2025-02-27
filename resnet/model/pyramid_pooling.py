# 引入金字塔池化的ResNet50模型 02

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 4, 8]):
        super(PyramidPooling, self).__init__()

        self.pool_sizes = pool_sizes
        self.stages = nn.ModuleList([])

        for pool_size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1)
            ))

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        outputs = [x]

        for stage in self.stages:
            feat = stage[0](x)
            feat = stage[1](feat)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            outputs.append(feat)

        return torch.cat(outputs, dim=1)


class ResidualClassifier(nn.Module):
    def __init__(self, total_channels, num_classes):
        super().__init__()
        self.in_features = total_channels  # 添加这行
        self.fc1 = nn.Linear(total_channels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        identity = self.fc1(x)
        out = self.relu(identity)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out + identity)  # 残差连接
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class Bottleneck(nn.Module):
    """ 标准的瓶颈残差块 """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = nn.ReLU()(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=4):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 去除最后一次空间下采样，扩大特征图
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.pyramid_pooling = PyramidPooling(2048, [1, 2, 4, 8])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 修改分类器以适应新的特征维度
        # 原始特征 + 4个池化分支的特征
        # total_channels = 2048 + (2048 // 4) * 85  # 2048 + 512 * 21
        total_channels = 2048 + (2048 // 4) * 4
        # self.fc = nn.Linear(total_channels, num_classes)

        self.classifier = ResidualClassifier(total_channels, num_classes=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pyramid_pooling(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        x = self.classifier(x)

        return x


def resnet50_pp(num_classes=4):
    """ 构建ResNet50模型 """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

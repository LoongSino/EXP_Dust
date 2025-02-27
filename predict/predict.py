import torch
import torch.nn as nn
from unet.model.benchmark import UNet
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = UNet(n_channels=3).to(device)
model.load_state_dict(torch.load("../unet/train/best_model.pth", map_location=device, weights_only=True))
model.eval()


# 定义预测函数
def predict_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),    ])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_mask = (output > 0.5).float()

    return predicted_mask


# 测试图片路径
test_image_path = "../datasets/test/images/69.jpg"

# 进行预测
predicted_mask = predict_image(test_image_path, model, device)

# 显示结果
plt.figure(figsize=(10, 5))

# 显示原始图片
plt.subplot(1, 2, 1)
original_image = Image.open(test_image_path).convert("RGB")
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

# 显示预测的掩码
plt.subplot(1, 2, 2)
plt.imshow(predicted_mask.cpu().squeeze().numpy(), cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

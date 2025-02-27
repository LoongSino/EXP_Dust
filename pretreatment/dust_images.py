import cv2
import numpy as np


def apply_mask(original_image, mask):
    # 读取图片
    if isinstance(original_image, str):
        original = cv2.imread(original_image)
    else:
        original = original_image

    if isinstance(mask, str):
        mask = cv2.imread(mask, 0)  # 以灰度图方式读取掩码

    # 确保掩码是二值图像
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 将掩码转换为3通道（如果原图是彩色的）
    if len(original.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 进行按位与操作
    result = cv2.bitwise_and(original, mask)

    return result


# 使用示例
original_path = "../dust_images/RandomDataset_100/train/Original_inputs/44.jpg"
mask_path = "../dust_images/RandomDataset_100/train/Segmentation_masks/44.jpg"
result = apply_mask(original_path, mask_path)
cv2.imwrite("result.jpg", result)

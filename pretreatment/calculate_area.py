import random

from PIL import Image
import numpy as np
import os
import shutil
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


# 计算面积
def calculate_percentage(mask_image_path):
    # 打开mask图像
    mask = Image.open(mask_image_path).convert('L')  # 转换为灰度图
    width, height = mask.size

    # 计算总像素数
    total_pixels = width * height

    # 将图像转换为numpy数组进行快速操作
    mask_np = np.array(mask)

    # 统计非零像素的数量，即标注区域的面积
    labeled_area = np.sum(mask_np > 0)  # 假设阈值为0区分前景与背景

    # 计算百分比
    percentage = (labeled_area / total_pixels) * 100

    return percentage


def classify_images_by_percentage(mask_dir, original_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取mask目录中的所有图像文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 用于记录每个百分比范围内的图像数量
    percentage_distribution = defaultdict(int)

    for mask_file in mask_files:
        mask_image_path = os.path.join(mask_dir, mask_file)
        percentage = calculate_percentage(mask_image_path)

        # 确定对应的原始图像文件路径
        original_image_path = os.path.join(original_dir, mask_file)
        if not os.path.exists(original_image_path):
            print(f"Warning: Original image {mask_file} not found in {original_dir}.")
            continue

        # 根据百分比确定存放的子目录
        if percentage == 0:
            sub_dir = os.path.join(output_dir, 'Normal')
        else:
            sub_dir = os.path.join(output_dir, 'Dust')

        # 确保子目录存在
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        # 复制原始图像到相应的子目录
        shutil.copy(original_image_path, os.path.join(sub_dir, mask_file))
        print(f"Image {mask_file} has a labeled area of {percentage:.2f}% and the original image is saved in {sub_dir}.")

        # 更新百分比分布统计
        if percentage == 0:
            percentage_distribution['Normal'] += 1
        else:
            percentage_distribution['Dust'] += 1

    # 将结果保存为 JSON
    output_json_path = '../datasets/NormalOrDust.json'
    with open(output_json_path, 'w') as json_file:
        json.dump(percentage_distribution, json_file, indent=4)

    # 定义标签和对应的值
    labels = ['Normal', 'Dust']
    values = [percentage_distribution[label] for label in labels]

    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 创建直方图
    plt.figure(figsize=(10, 6))
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']  # 添加颜色
    plt.bar(labels, values, color=colors)

    # 添加标题和标签
    plt.title('有无扬尘直方图')

    # 显示直方图
    plt.tight_layout()  # 确保标签不被截断

    # 保存直方图为图片
    plt.savefig('../datasets/dust_degree_histogram.png')


# 使用函数
mask_dir = '../dust_images/SequentialDataset_7k/Segmentation_masks'
original_dir = '../dust_images/SequentialDataset_7k/Original_input'
output_dir = '../datasets/classified_images'

classify_images_by_percentage(mask_dir, original_dir, output_dir)



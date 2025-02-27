import os
import random
import shutil
import numpy as np
from sklearn.model_selection import train_test_split


def split_folder(input_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, maximum=1000):
    # 确保训练集和验证集的比例之和为1
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1), "ratios must add up to 1"

    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的每个类别文件夹
    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        if os.path.isdir(class_path):
            # 列出类别文件夹中的所有文件
            files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

            # 如果类别文件夹中没有文件，则跳过
            if len(files) == 0:
                print(f"No files found in {class_folder}. Skipping...")
                continue

            # 限制每个类别的最大文件数量
            files = random.sample(files, min(maximum, len(files)))

            # 计算每个集合中的文件数量
            total_files = len(files)
            train_count = int(total_files * train_ratio)
            val_count = int(total_files * val_ratio)
            test_count = total_files - train_count - val_count

            # 将文件分割成训练集、验证集和测试集
            train_files, val_files, test_files = train_test_split(
                files, train_size=train_count, test_size=test_count, random_state=42)

            # 为类别创建输出目录
            train_output = os.path.join(output_folder, 'train', class_folder)
            val_output = os.path.join(output_folder, 'val', class_folder)
            test_output = os.path.join(output_folder, 'test', class_folder)

            os.makedirs(train_output, exist_ok=True)
            os.makedirs(val_output, exist_ok=True)
            os.makedirs(test_output, exist_ok=True)

            # 将文件复制到相应的目录
            for file in train_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(train_output, file))
            for file in val_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(val_output, file))
            for file in test_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(val_output, file))


split_folder('../datasets/classified_images', '../datasets', 0.8, 0.1, 0.1, 1000)

print('划分数据集完成')

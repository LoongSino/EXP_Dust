import os
import random
import shutil
from tqdm import tqdm

# 配置路径
base_path = '../dust_images/SequentialDataset_7k'
output_root = '../datasets'  # 输出根目录
src_dirs = {
    'images': 'Original_input',
    'annotations': 'Hand_Annotated',
    'masks': 'Segmentation_masks'
}

dest_dirs = ['train', 'val', 'test']
split_ratio = [0.7, 0.2, 0.1]

# 创建目标目录结构
for split in dest_dirs:
    for dtype in src_dirs:
        dir_path = os.path.join(output_root, split, dtype)
        os.makedirs(dir_path, exist_ok=True)

# 获取所有基础文件名（不带扩展名）
src_images_dir = os.path.join(base_path, src_dirs['images'])
image_files = [os.path.splitext(f)[0] for f in os.listdir(src_images_dir)]
random.shuffle(image_files)

# 计算划分索引
total = len(image_files)
train_end = int(total * split_ratio[0])
val_end = train_end + int(total * split_ratio[1])

# 按划分复制文件
for i, filename in tqdm(enumerate(image_files), total=total):
    split = 'test'  # 默认值
    if i < train_end:
        split = 'train'
    elif i < val_end:
        split = 'val'

    # 复制所有类型文件
    for dtype, src_subdir in src_dirs.items():
        src_dir = os.path.join(base_path, src_subdir)
        src_file = None

        # 查找存在的文件扩展名
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            candidate = os.path.join(src_dir, f"{filename}{ext}")
            if os.path.exists(candidate):
                src_file = candidate
                break

        if not src_file:
            print(f"Warning: Missing {dtype} file for {filename}")
            continue

        # 构建目标路径
        dest_path = os.path.join(output_root, split, dtype, os.path.basename(src_file))
        shutil.copy(src_file, dest_path)

print(f"Dataset splitting completed! Output saved to: {os.path.abspath(output_root)}")

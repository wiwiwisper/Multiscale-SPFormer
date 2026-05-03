"""
整理数据脚本：从 .pth 文件中提取每个实例的点云，按茎/叶分类保存

目标结构：
phenotype/data/leaf/日期/编号/leaf_实例ID.npy
phenotype/data/stem/日期/编号/stem_实例ID.npy

每个 .npy 文件包含一个字典：
{
    'coords': (N, 3),  # 点云坐标
    'colors': (N, 3),  # RGB 颜色
    'semantic_id': int,  # 实例ID
    'class_id': int      # 类别ID (0=stem, 1=leaf)
}
"""

import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# 配置
SOURCE_DIR = '../data/myplants/train'
TARGET_DIR = './data'
CLASS_NAMES = {0: 'stem', 1: 'leaf'}

def parse_filename(filename):
    """解析文件名：10_2025_03_02.pth -> (编号=10, 日期=2025_03_02)"""
    name = filename.replace('.pth', '')
    parts = name.split('_')
    num = parts[0]
    date = '_'.join(parts[1:])
    return num, date

def extract_instances(pth_file):
    """从 .pth 文件提取所有实例"""
    data = torch.load(pth_file)
    coords = data[0]  # (N, 3)
    colors = data[1]  # (N, 3)
    semantic_labels = data[2]  # (N,) 实例ID
    class_labels = data[3]  # (N,) 类别ID

    instances = []
    unique_instances = np.unique(semantic_labels)

    for inst_id in unique_instances:
        mask = semantic_labels == inst_id
        inst_coords = coords[mask]
        inst_colors = colors[mask]
        inst_class = int(class_labels[mask][0])  # 同一实例的类别相同

        instances.append({
            'coords': inst_coords,
            'colors': inst_colors,
            'semantic_id': int(inst_id),
            'class_id': inst_class
        })

    return instances

def main():
    # 创建目标目录
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 获取所有 .pth 文件
    pth_files = sorted(Path(SOURCE_DIR).glob('*.pth'))
    print(f'找到 {len(pth_files)} 个 .pth 文件')

    stats = {'stem': 0, 'leaf': 0}

    for pth_file in tqdm(pth_files, desc='处理文件'):
        num, date = parse_filename(pth_file.name)

        # 提取实例
        instances = extract_instances(str(pth_file))

        # 按类别保存
        for inst in instances:
            class_name = CLASS_NAMES[inst['class_id']]
            inst_id = inst['semantic_id']

            # 创建目录：data/leaf/2025_03_02/10/
            save_dir = Path(TARGET_DIR) / class_name / date / num
            save_dir.mkdir(parents=True, exist_ok=True)

            # 保存文件：leaf_123.npy
            save_path = save_dir / f'{class_name}_{inst_id}.npy'
            np.save(save_path, inst)

            stats[class_name] += 1

    print('\n--- 统计 ---')
    print(f'茎实例数: {stats["stem"]}')
    print(f'叶实例数: {stats["leaf"]}')
    print(f'\n数据已保存到: {TARGET_DIR}/')

if __name__ == '__main__':
    main()

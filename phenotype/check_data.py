import torch
import numpy as np

# 检查数据格式
data = torch.load('../data/myplants/train/10_2025_03_02.pth')
print('Tuple length:', len(data))
for i, item in enumerate(data):
    if hasattr(item, 'shape'):
        print(f'Item {i}: type={type(item).__name__}, shape={item.shape}')
    else:
        print(f'Item {i}: type={type(item).__name__}, len={len(item) if hasattr(item, "__len__") else "N/A"}')

# 假设格式为 (coords, colors, semantic_labels, instance_labels, ...)
coords, colors, sem_labels, inst_labels = data[0], data[1], data[2], data[3]
print('\n--- Data Info ---')
print(f'Coords: {coords.shape}')
print(f'Colors: {colors.shape}')
print(f'Semantic labels: {sem_labels.shape}, unique: {np.unique(sem_labels)}')
print(f'Instance labels: {inst_labels.shape}, unique instances: {len(np.unique(inst_labels))}')

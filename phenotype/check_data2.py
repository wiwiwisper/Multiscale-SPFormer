import torch
import numpy as np

data = torch.load('../data/myplants/test/1_2025_06_04.pth')
print('字段数:', len(data))
for i, item in enumerate(data):
    uniq = np.unique(item)
    print(f'data[{i}]: shape={item.shape}, dtype={item.dtype}, '
          f'unique_count={len(uniq)}, min={item.min():.3f}, max={item.max():.3f}')

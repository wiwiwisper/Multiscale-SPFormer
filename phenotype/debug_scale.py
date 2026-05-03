import torch
import numpy as np

data = torch.load('../data/myplants/test/1_2025_06_04.pth')
coords = data[0].astype(np.float32)
sp_ids = data[2].astype(np.int32)

n_sp = sp_ids.max() + 1
centroids = np.zeros((n_sp, 3), dtype=np.float32)
counts = np.zeros(n_sp, dtype=np.int32)
np.add.at(centroids, sp_ids, coords)
np.add.at(counts, sp_ids, 1)
centroids /= counts[:, None]

print("超点质心坐标范围:")
print(f"  X: {centroids[:,0].min():.4f} ~ {centroids[:,0].max():.4f}")
print(f"  Y: {centroids[:,1].min():.4f} ~ {centroids[:,1].max():.4f}")
print(f"  Z: {centroids[:,2].min():.4f} ~ {centroids[:,2].max():.4f}")

for scale in [8.0, 16.0, 32.0, 64.0]:
    grid_raw = np.floor(centroids / scale).astype(np.int32)
    unique_raw = np.unique(grid_raw, axis=0)
    grid = grid_raw - grid_raw.min(axis=0)
    shape = grid.max(axis=0) + 1
    keys = grid[:,0]*(shape[1]*shape[2]) + grid[:,1]*shape[2] + grid[:,2]
    n_coarse = len(np.unique(keys))
    print(f"\nscale={scale}: 粗超点数={n_coarse}, 原始grid唯一值 (未offset):")
    for g in unique_raw:
        print(f"  {g}")

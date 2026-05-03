"""
GT实例分割可视化：根据真值标注上色，输出 gt.ply。

上色规则：
  - 茎（class=0）：统一棕色 (139, 69, 19)
  - 叶片（class=1）：每个实例随机不重复鲜亮颜色，禁用棕色
"""

import numpy as np
import torch

DATA_FILE = '../data/myplants/test/1_2025_06_04.pth'
OUT_FILE  = './visualization/gt.ply'

STEM_COLOR = np.array([139, 69, 19], dtype=np.uint8)


def distinct_colors(n, seed=42):
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, n + 20, endpoint=False)
    rng.shuffle(hues)
    colors = []
    for h in hues:
        if len(colors) >= n:
            break
        if h < 0.08 or h > 0.95:
            continue
        hi = int(h * 6) % 6
        f  = h * 6 - int(h * 6)
        p, q, t = 0.0, 1.0 * (1 - f), 1.0 * f
        v = 1.0
        rgb = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][hi]
        colors.append((np.array(rgb) * 255).astype(np.uint8))
    while len(colors) < n:
        colors.append(rng.integers(50, 230, 3, dtype=np.uint8))
    return colors


def save_ply(path, coords, colors):
    n = len(coords)
    dt = np.dtype([('x',np.float32),('y',np.float32),('z',np.float32),
                   ('red',np.uint8),('green',np.uint8),('blue',np.uint8)])
    v = np.empty(n, dtype=dt)
    v['x'],v['y'],v['z'] = coords[:,0],coords[:,1],coords[:,2]
    v['red'],v['green'],v['blue'] = colors[:,0],colors[:,1],colors[:,2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(v.tobytes())
    print(f"已保存: {path}  ({n} 个点)")


def main():
    data         = torch.load(DATA_FILE)
    coords       = data[0].astype(np.float32)
    class_labels = data[3].astype(np.int32)
    inst_labels  = data[4].astype(np.int32)

    N = len(coords)
    leaf_inst_ids = np.unique(inst_labels[class_labels == 1])
    n_leaf = len(leaf_inst_ids)
    print(f"点数: {N},  茎实例: 1,  叶实例: {n_leaf}")

    leaf_colors = distinct_colors(n_leaf)
    leaf_cmap   = {iid: leaf_colors[i] for i, iid in enumerate(leaf_inst_ids)}

    point_colors = np.zeros((N, 3), dtype=np.uint8)
    point_colors[class_labels == 0] = STEM_COLOR
    for iid, c in leaf_cmap.items():
        point_colors[(class_labels == 1) & (inst_labels == iid)] = c

    save_ply(OUT_FILE, coords, point_colors)


if __name__ == '__main__':
    main()

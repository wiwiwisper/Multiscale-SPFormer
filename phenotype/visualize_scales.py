"""
可视化脚本：针对植株1最后一天数据，输出4个 PLY 文件
  1. voxel_mapped.ply   — 体素化后映射回原始点，按体素ID染色
  2. sp_level1.ply      — 一级超点（数据集原始超点划分）
  3. sp_level2.ply      — 二级超点（coarse_scale=0.5）
  4. sp_level3.ply      — 三级超点（coarse_scale=1.0）

输出目录：phenotype/visualization/
"""

import numpy as np
import torch
import os
from pathlib import Path

# ── 配置 ─────────────────────────────────────────────────────────────
DATA_FILE   = '../data/myplants/test/1_2025_06_04.pth'
OUT_DIR     = './visualization'
VOXEL_SCALE = 50        # 坐标 * 50 后体素化（与训练一致）
COARSE_1    = 0.5       # 二级超点网格尺寸（原始坐标单位）
COARSE_2    = 1.0       # 三级超点网格尺寸
COARSE_3    = 2.0       # 四级超点网格尺寸
COARSE_4    = 4.0       # 五级超点网格尺寸
COARSE_5    = 8.0       # 六级超点网格尺寸
COARSE_6    = 16.0      # 七级超点网格尺寸
COARSE_7    = 32.0      # 八级超点网格尺寸
COARSE_8    = 64.0      # 九级超点网格尺寸
# ─────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def load_data(path):
    data = torch.load(path)
    coords      = data[0].astype(np.float32)   # (N, 3) 原始坐标（单位：米）
    colors      = data[1].astype(np.float32)   # (N, 3) RGB（归一化）
    sp_labels   = data[2].astype(np.int32)     # (N,)   超点索引（6049 unique → 一级超点）
    class_labels = data[3].astype(np.int32)    # (N,)   类别（0=stem, 1=leaf）
    return coords, colors, sp_labels, class_labels


def random_color_map(n, seed=42):
    """生成 n 个可区分的随机颜色，返回 (n, 3) uint8"""
    rng = np.random.default_rng(seed)
    # 用 HSV 均匀撒点再转 RGB，避免颜色过于相近
    hues = np.linspace(0, 1, n, endpoint=False)
    rng.shuffle(hues)
    colors = np.zeros((n, 3), dtype=np.float32)
    for i, h in enumerate(hues):
        # HSV→RGB，S=1.0, V=1.0 全饱和全亮度
        hi = int(h * 6) % 6
        f  = h * 6 - int(h * 6)
        p, q, t = 0.0, 1.0 * (1 - f), 1.0 * f
        v = 1.0
        rgb = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][hi]
        colors[i] = rgb
    return (colors * 255).astype(np.uint8)


def ids_to_colors(ids, seed=42):
    """将 id 数组映射为 RGB 颜色数组 (N, 3) uint8"""
    unique_ids = np.unique(ids)
    cmap = random_color_map(len(unique_ids), seed=seed)
    id2color = {uid: cmap[i] for i, uid in enumerate(unique_ids)}
    colors = np.stack([id2color[i] for i in ids])
    return colors


def save_ply(path, coords, colors):
    """保存带颜色的点云 PLY 文件（二进制，numpy 向量化，速度快）"""
    n = len(coords)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    # 构建结构化 numpy 数组
    dt = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('red', np.uint8), ('green', np.uint8), ('blue', np.uint8),
    ])
    vertex = np.empty(n, dtype=dt)
    vertex['x'] = coords[:, 0]
    vertex['y'] = coords[:, 1]
    vertex['z'] = coords[:, 2]
    vertex['red']   = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue']  = colors[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertex.tobytes())
    print(f"  已保存: {path}  ({n} 个点)")


def voxelize(coords, scale):
    """将坐标 * scale 后取整，返回每个点的体素整数坐标 (N, 3)"""
    voxel_coords = np.floor(coords * scale).astype(np.int32)
    return voxel_coords


def voxel_to_ids(voxel_coords):
    """将体素坐标编码为唯一整数 ID"""
    # 平移到非负
    offset = voxel_coords.min(axis=0)
    vc = voxel_coords - offset
    # 编码为 1D key
    shape = vc.max(axis=0) + 1
    keys = vc[:, 0] * (shape[1] * shape[2]) + vc[:, 1] * shape[2] + vc[:, 2]
    # 映射到紧凑 ID
    _, ids = np.unique(keys, return_inverse=True)
    return ids


def create_coarse_sp(sp_coords, fine_ids, scale):
    """
    将细超点按空间网格聚合为粗超点。
    先平移坐标至全非负，避免负坐标跨越 0 导致错误分组。
    sp_coords: (M_fine, 3)  细超点质心
    fine_ids:  (N,)         每个原始点对应的细超点 ID
    scale:     float        粗网格尺寸
    返回: (N,) 每个原始点对应的粗超点 ID
    """
    # 平移到全非负（消除负坐标跨越原点的问题）
    shifted = sp_coords - sp_coords.min(axis=0)
    grid = np.floor(shifted / scale).astype(np.int32)           # (M_fine, 3)
    shape = grid.max(axis=0) + 1
    keys = grid[:, 0] * (shape[1] * shape[2]) + grid[:, 1] * shape[2] + grid[:, 2]
    _, coarse_ids_per_fine = np.unique(keys, return_inverse=True)  # (M_fine,)
    return coarse_ids_per_fine[fine_ids]                            # (N,)


def compute_sp_centroids(coords, sp_ids):
    """计算每个超点的质心"""
    n_sp = sp_ids.max() + 1
    centroids = np.zeros((n_sp, 3), dtype=np.float32)
    counts = np.zeros(n_sp, dtype=np.int32)
    np.add.at(centroids, sp_ids, coords)
    np.add.at(counts, sp_ids, 1)
    centroids /= counts[:, None].clip(min=1)
    return centroids


def main():
    print(f"加载数据: {DATA_FILE}")
    coords, orig_colors, level1_ids, class_labels = load_data(DATA_FILE)
    n = len(coords)
    print(f"  点数: {n}, 一级超点数: {level1_ids.max()+1}")

    # ── 1. 体素化 → 映射回原始点按体素着色 ──────────────────────────
    print("\n[1/6] 体素化...")
    voxel_coords = voxelize(coords, VOXEL_SCALE)
    voxel_ids = voxel_to_ids(voxel_coords)
    n_vox = voxel_ids.max() + 1
    print(f"  体素数: {n_vox}")
    voxel_colors = ids_to_colors(voxel_ids, seed=0)
    save_ply(f"{OUT_DIR}/voxel_mapped.ply", coords, voxel_colors)

    # ── 2. 一级超点（数据集原始超点划分）────────────────────────────
    print("\n[2/6] 一级超点（原始超点）...")
    sp1_colors = ids_to_colors(level1_ids, seed=1)
    save_ply(f"{OUT_DIR}/sp_level1.ply", coords, sp1_colors)

    # ── 3. 二级超点（coarse_scale=0.5）──────────────────────────────
    print("\n[3/6] 二级超点...")
    # 计算一级超点质心
    sp1_centroids = compute_sp_centroids(coords, level1_ids)  # (M1, 3)
    level2_ids = create_coarse_sp(sp1_centroids, level1_ids, COARSE_1)
    n_sp2 = level2_ids.max() + 1
    print(f"  二级超点数: {n_sp2}")
    sp2_colors = ids_to_colors(level2_ids, seed=2)
    save_ply(f"{OUT_DIR}/sp_level2.ply", coords, sp2_colors)

    # ── 4. 三级超点（coarse_scale=1.0）──────────────────────────────
    print("\n[4/6] 三级超点...")
    level3_ids = create_coarse_sp(sp1_centroids, level1_ids, COARSE_2)
    n_sp3 = level3_ids.max() + 1
    print(f"  三级超点数: {n_sp3}")
    sp3_colors = ids_to_colors(level3_ids, seed=3)
    save_ply(f"{OUT_DIR}/sp_level3.ply", coords, sp3_colors)

    # ── 5. 四级超点（coarse_scale=2.0）──────────────────────────────
    print("\n[5/10] 四级超点...")
    level4_ids = create_coarse_sp(sp1_centroids, level1_ids, COARSE_3)
    n_sp4 = level4_ids.max() + 1
    print(f"  四级超点数: {n_sp4}")
    sp4_colors = ids_to_colors(level4_ids, seed=4)
    save_ply(f"{OUT_DIR}/sp_level4.ply", coords, sp4_colors)

    # ── 6. 五级超点（coarse_scale=4.0）──────────────────────────────
    print("\n[6/10] 五级超点...")
    level5_ids = create_coarse_sp(sp1_centroids, level1_ids, COARSE_4)
    n_sp5 = level5_ids.max() + 1
    print(f"  五级超点数: {n_sp5}")
    sp5_colors = ids_to_colors(level5_ids, seed=5)
    save_ply(f"{OUT_DIR}/sp_level5.ply", coords, sp5_colors)

    # ── 7. 六级超点（coarse_scale=8.0）──────────────────────────────
    print("\n[7/10] 六级超点...")
    level6_ids = create_coarse_sp(sp1_centroids, level1_ids, COARSE_5)
    n_sp6 = level6_ids.max() + 1
    print(f"  六级超点数: {n_sp6}")
    sp6_colors = ids_to_colors(level6_ids, seed=6)
    save_ply(f"{OUT_DIR}/sp_level6.ply", coords, sp6_colors)

    # ── 8. 七级超点（coarse_scale=16.0）─────────────────────────────
    print("\n[8/10] 七级超点...")
    level7_ids = create_coarse_sp(sp1_centroids, level1_ids, COARSE_6)
    n_sp7 = level7_ids.max() + 1
    print(f"  七级超点数: {n_sp7}")
    sp7_colors = ids_to_colors(level7_ids, seed=7)
    save_ply(f"{OUT_DIR}/sp_level7.ply", coords, sp7_colors)

    # ── 9. 八级超点（coarse_scale=32.0）─────────────────────────────
    print("\n[9/10] 八级超点...")
    level8_ids = create_coarse_sp(sp1_centroids, level1_ids, COARSE_7)
    n_sp8 = level8_ids.max() + 1
    print(f"  八级超点数: {n_sp8}")
    sp8_colors = ids_to_colors(level8_ids, seed=8)
    save_ply(f"{OUT_DIR}/sp_level8.ply", coords, sp8_colors)

    # ── 10. 九级超点（coarse_scale=64.0）────────────────────────────
    print("\n[10/10] 九级超点...")
    level9_ids = create_coarse_sp(sp1_centroids, level1_ids, COARSE_8)
    n_sp9 = level9_ids.max() + 1
    print(f"  九级超点数: {n_sp9}")
    sp9_colors = ids_to_colors(level9_ids, seed=9)
    save_ply(f"{OUT_DIR}/sp_level9.ply", coords, sp9_colors)

    print(f"\n完成！所有文件已输出到 {OUT_DIR}/")
    print(f"  体素数:     {n_vox}")
    print(f"  一级超点数: {level1_ids.max()+1}")
    print(f"  二级超点数: {n_sp2}")
    print(f"  三级超点数: {n_sp3}")
    print(f"  四级超点数: {n_sp4}")
    print(f"  五级超点数: {n_sp5}")
    print(f"  六级超点数: {n_sp6}")
    print(f"  七级超点数: {n_sp7}")
    print(f"  八级超点数: {n_sp8}")
    print(f"  九级超点数: {n_sp9}")


if __name__ == '__main__':
    main()

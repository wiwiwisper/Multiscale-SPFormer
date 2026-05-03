"""
体素化可视化：将每个体素渲染为一个小立方体，输出网格PLY。
颜色取该体素内所有点的平均RGB。

输出：phenotype/visualization/voxel_cubes.ply
"""

import numpy as np
import torch

DATA_FILE  = '../data/myplants/test/1_2025_06_04.pth'
OUT_FILE   = './visualization/voxel_cubes.ply'
VOXEL_SCALE = 50   # 与训练一致，1/50 = 0.02m 每体素


def main():
    data   = torch.load(DATA_FILE)
    coords = data[0].astype(np.float32)   # (N, 3) 原始坐标（米）
    colors = data[1].astype(np.float32)   # (N, 3) RGB，范围 0~1

    # ── 1. 计算每个点所属体素的整数坐标 ──────────────────────────────
    vox = np.floor(coords * VOXEL_SCALE).astype(np.int32)   # (N, 3)

    # 编码为 1D key，找唯一体素
    offset = vox.min(axis=0)
    vc     = vox - offset
    shape  = vc.max(axis=0) + 1
    keys   = vc[:, 0] * (shape[1] * shape[2]) + vc[:, 1] * shape[2] + vc[:, 2]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n_vox = len(unique_keys)
    print(f"体素数: {n_vox}")

    # ── 2. 每个体素的平均RGB颜色 ──────────────────────────────────────
    colors_u8 = (colors * 255).clip(0, 255).astype(np.uint8)
    vox_color = np.zeros((n_vox, 3), dtype=np.float64)
    vox_count = np.zeros(n_vox, dtype=np.int32)
    np.add.at(vox_color, inverse, colors_u8.astype(np.float64))
    np.add.at(vox_count, inverse, 1)
    vox_color = (vox_color / vox_count[:, None]).astype(np.uint8)   # (n_vox, 3)

    # ── 3. 每个体素的最小角坐标（原始单位）──────────────────────────
    vox_z = unique_keys % shape[2]
    vox_y = (unique_keys // shape[2]) % shape[1]
    vox_x = unique_keys // (shape[1] * shape[2])
    vox_min = (np.stack([vox_x, vox_y, vox_z], axis=1) + offset).astype(np.float32) / VOXEL_SCALE
    # vox_min[i] = 体素i的最小角坐标，体素边长 = 1/VOXEL_SCALE

    # ── 4. 构建立方体顶点与三角面 ──────────────────────────────────────
    step = 1.0 / VOXEL_SCALE

    # 8个顶点的局部偏移（相对于最小角）
    corner_offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32) * step   # (8, 3)

    # 12个三角面的局部顶点索引（顺时针，外法线朝外）
    local_faces = np.array([
        [0, 2, 1], [0, 3, 2],   # 底面
        [4, 5, 6], [4, 6, 7],   # 顶面
        [0, 1, 5], [0, 5, 4],   # 前面
        [2, 7, 6], [2, 3, 7],   # 后面
        [0, 4, 7], [0, 7, 3],   # 左面
        [1, 2, 6], [1, 6, 5],   # 右面
    ], dtype=np.int32)           # (12, 3)

    # 所有体素的顶点：(n_vox, 8, 3) → (n_vox*8, 3)
    vertices = vox_min[:, None, :] + corner_offsets[None, :, :]
    vertices = vertices.reshape(-1, 3)

    # 顶点颜色：每个体素8个顶点共享同一颜色
    vertex_colors = np.repeat(vox_color, 8, axis=0)   # (n_vox*8, 3)

    # 所有体素的面：全局顶点索引 = 局部索引 + 体素基址(i*8)
    base = np.arange(n_vox, dtype=np.int32) * 8        # (n_vox,)
    faces = local_faces[None, :, :] + base[:, None, None]  # (n_vox, 12, 3)
    faces = faces.reshape(-1, 3)                        # (n_vox*12, 3)

    print(f"顶点数: {len(vertices)},  面数: {len(faces)}")

    # ── 5. 写二进制PLY ────────────────────────────────────────────────
    n_v, n_f = len(vertices), len(faces)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    vdt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                    ('red', np.uint8), ('green', np.uint8), ('blue', np.uint8)])
    vdata = np.empty(n_v, dtype=vdt)
    vdata['x'], vdata['y'], vdata['z'] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    vdata['red'], vdata['green'], vdata['blue'] = vertex_colors[:, 0], vertex_colors[:, 1], vertex_colors[:, 2]

    fdt = np.dtype([('n', np.uint8), ('v0', np.int32), ('v1', np.int32), ('v2', np.int32)])
    fdata = np.empty(n_f, dtype=fdt)
    fdata['n'] = 3
    fdata['v0'], fdata['v1'], fdata['v2'] = faces[:, 0], faces[:, 1], faces[:, 2]

    with open(OUT_FILE, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vdata.tobytes())
        f.write(fdata.tobytes())

    print(f"已保存: {OUT_FILE}")


if __name__ == '__main__':
    main()

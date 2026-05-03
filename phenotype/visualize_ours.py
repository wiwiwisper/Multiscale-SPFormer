"""
将模型预测的实例分割结果可视化为PLY。

上色规则：
  - 茎（label=1）：统一棕色 (139, 69, 19)
  - 叶（label=2）：每个实例随机不重复颜色，禁用棕色
  - 未覆盖点：黑色 (0, 0, 0)

输出两份：
  - ours_with_black.ply（含黑色未覆盖点）
  - ours_no_black.ply（去除黑色点）
"""

import numpy as np
import torch
import os

DATA_FILE   = '../data/myplants/test/1_2025_06_04.pth'
PRED_FILE   = '../exps/myplants_ours2_nogeo/test_results/pred_instance/1_2025_06_04.txt'
PRED_DIR    = '../exps/myplants_ours2_nogeo/test_results/pred_instance'
OUT_DIR     = './visualization'
SCORE_THR   = 0.0   # 与测试配置保持一致

STEM_COLOR  = np.array([139, 69, 19], dtype=np.uint8)
BLACK       = np.array([0,   0,  0],  dtype=np.uint8)


def load_coords(path):
    data = torch.load(path)
    return data[0].astype(np.float32)   # (N, 3)


def load_predictions(pred_file, pred_dir):
    """返回 list of (mask_path, label_id, score)，按 score 降序"""
    preds = []
    with open(pred_file) as f:
        for line in f:
            parts = line.strip().split()
            rel_mask, label_id, score = parts[0], int(parts[1]), float(parts[2])
            mask_path = os.path.join(pred_dir, rel_mask)
            preds.append((mask_path, label_id, score))
    preds.sort(key=lambda x: -x[2])   # 高分优先
    return preds


def load_mask(mask_path):
    """读取逐点0/1 mask，返回 bool 数组 (N,)"""
    return np.loadtxt(mask_path, dtype=np.uint8).astype(bool)


def distinct_leaf_colors(n, seed=42):
    """生成 n 个随机且互不重复、不近棕色的叶片颜色"""
    rng = np.random.default_rng(seed)
    colors = []
    hues = np.linspace(0, 1, n + 20, endpoint=False)
    rng.shuffle(hues)
    for h in hues:
        if len(colors) >= n:
            break
        # 排除棕色附近（红-橙色调，H≈0~0.08 或 H≈0.95~1.0，低饱和）
        if h < 0.08 or h > 0.95:
            continue
        hi = int(h * 6) % 6
        f  = h * 6 - int(h * 6)
        p, q, t = 0.95*(1-0.8), 0.95*(1-0.8*f), 0.95*(1-0.8*(1-f))
        v = 0.95
        rgb = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][hi]
        colors.append((np.array(rgb) * 255).astype(np.uint8))
    # 若不够则补随机色
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
    print(f"  已保存: {path}  ({n} 个点)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("加载点云坐标...")
    coords = load_coords(DATA_FILE)
    N = len(coords)
    print(f"  点数: {N}")

    print("加载预测结果...")
    preds = load_predictions(PRED_FILE, PRED_DIR)
    # 过滤低置信度
    preds = [(mp, lb, sc) for mp, lb, sc in preds if sc > SCORE_THR]
    # NYU_ID=[0,1]：保存时 stem→0, leaf→1
    n_stem = sum(1 for _,lb,_ in preds if lb == 0)
    n_leaf = sum(1 for _,lb,_ in preds if lb == 1)
    print(f"  实例总数: {len(preds)}  （茎: {n_stem}, 叶: {n_leaf}）")

    # 生成叶片颜色表
    leaf_colors = distinct_leaf_colors(n_leaf)
    leaf_color_iter = iter(leaf_colors)

    # 初始化：所有点为黑色，记录每个点是否已被分配
    point_colors = np.zeros((N, 3), dtype=np.uint8)   # 默认黑色
    assigned     = np.zeros(N, dtype=bool)

    print("逐实例上色（高分优先，先到先得）...")
    for mask_path, label_id, score in preds:
        mask = load_mask(mask_path)          # (N,) bool
        # 只给尚未分配的点上色
        target = mask & ~assigned

        if label_id == 0:   # 茎 (nyu_id=0)
            point_colors[target] = STEM_COLOR
        else:               # 叶 (nyu_id=1)
            c = next(leaf_color_iter)
            point_colors[target] = c

        assigned[target] = True

    covered  = assigned.sum()
    black_n  = (~assigned).sum()
    print(f"  已覆盖点: {covered} / {N}   黑色点: {black_n}")

    # ── 输出1：含黑色点 ───────────────────────────────────────────────
    out1 = os.path.join(OUT_DIR, 'ours_with_black.ply')
    save_ply(out1, coords, point_colors)

    # ── 输出2：去除黑色点 ─────────────────────────────────────────────
    keep = assigned
    out2 = os.path.join(OUT_DIR, 'ours_no_black.ply')
    save_ply(out2, coords[keep], point_colors[keep])

    # ── 输出3：仅黑色点 ───────────────────────────────────────────────
    only_black = ~assigned
    out3 = os.path.join(OUT_DIR, 'ours_only_black.ply')
    save_ply(out3, coords[only_black], point_colors[only_black])

    # ── 输出4：仅茎点 ────────────────────────────────────────────────
    stem_mask = assigned & np.all(point_colors == STEM_COLOR, axis=1)
    out4 = os.path.join(OUT_DIR, 'ours_only_stem.ply')
    save_ply(out4, coords[stem_mask], point_colors[stem_mask])

    # ── 输出5：仅叶点 ────────────────────────────────────────────────
    leaf_mask = assigned & ~np.all(point_colors == STEM_COLOR, axis=1)
    out5 = os.path.join(OUT_DIR, 'ours_only_leaf.ply')
    save_ply(out5, coords[leaf_mask], point_colors[leaf_mask])

    print("\n完成！")


if __name__ == '__main__':
    main()

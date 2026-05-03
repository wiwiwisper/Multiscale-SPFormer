# tools/rebuild_pth.py
import os, glob, argparse, numpy as np, torch
import open3d as o3d

def read_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float32)              # [N,3], float32
    if len(pcd.colors) == 0:
        rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32) # [0,1]
    else:
        rgb = np.asarray(pcd.colors, dtype=np.float32)          # Open3D 已是 [0,1]
    return xyz, rgb

def build_sp_index(xyz, voxel_size=0.02):
    """返回 0..K-1 的超点索引。优先用 segmentator，失败则体素聚类兜底。"""
    try:
        import segmentator
        if hasattr(segmentator, "build_superpoint_index"):
            sp = segmentator.build_superpoint_index(coord=xyz, voxel_size=voxel_size,
                                                    k_nn=30, relabel=True)
            return np.asarray(sp, dtype=np.int32)
        if hasattr(segmentator, "compute_superpoints"):
            sp = segmentator.compute_superpoints(xyz=xyz, voxel_size=voxel_size,
                                                 k=30, relabel=True)
            return np.asarray(sp, dtype=np.int32)
    except Exception:
        pass
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv.astype(np.int32)                                   # 0..K-1 连续

def process_scene(stem_ply, leaf_plys, out_path, voxel_size):
    # stem：sem=0, inst=0
    s_xyz, s_rgb = read_ply(stem_ply)
    xyzs, rgbs = [s_xyz], [s_rgb]
    sems = [np.zeros((s_xyz.shape[0],), dtype=np.int32)]
    insts = [np.zeros((s_xyz.shape[0],), dtype=np.int32)]
    # leaf：sem=1, inst=1,2,…
    iid = 1
    for lp in sorted(leaf_plys):
        x, c = read_ply(lp)
        xyzs.append(x); rgbs.append(c)
        sems.append(np.ones((x.shape[0],), dtype=np.int32))
        insts.append(np.full((x.shape[0],), iid, dtype=np.int32))
        iid += 1
    xyz = np.concatenate(xyzs, 0)
    rgb = np.concatenate(rgbs, 0)
    sem = np.concatenate(sems, 0)
    ins = np.concatenate(insts, 0)
    sp = build_sp_index(xyz, voxel_size=voxel_size)
    torch.save((xyz.astype(np.float32),
                rgb.astype(np.float32),
                sp.astype(np.int32),
                sem.astype(np.int32),
                ins.astype(np.int32)),
               out_path)
    print(f"[OK] {os.path.basename(out_path)} points={xyz.shape[0]} sp={sp.max()+1}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="raw/PlantDivision/<plant>/<date>/")
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--voxel", type=float, default=0.04, help="建议 val/test 用 0.04~0.06")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 遍历 raw 结构
    for plant in sorted(glob.glob(os.path.join(args.raw_root, "*"))):
        for date in sorted(glob.glob(os.path.join(plant, "*"))):
            stem = os.path.join(date, "stem.ply")
            leaves = glob.glob(os.path.join(date, "leaf*.ply"))
            if not os.path.isfile(stem):
                continue
            scene_id = f"{os.path.basename(plant)}_{os.path.basename(date)}"
            out = os.path.join(args.out_dir, scene_id + ".pth")
            process_scene(stem, leaves, out, args.voxel)

if __name__ == "__main__":
    main()

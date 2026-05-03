#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
叶片表面积计算 - 方法1
基于原始C++代码的算法：
1. 从点云重建三维网格
2. 计算真实三维表面积
3. 将顶点投影到XY平面，计算投影面积
"""

import open3d as o3d
import numpy as np
import os
import csv
from pathlib import Path
from tqdm import tqdm


def get_surface_area(mesh):
    """
    计算三角网格的表面积
    对每个三角面片，使用叉乘计算面积并累加
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    surface_area = 0.0
    
    for tri in triangles:
        # 获取三角形的三个顶点
        a = vertices[tri[0]]
        b = vertices[tri[1]]
        c = vertices[tri[2]]
        
        # 计算两条边向量
        v1 = b - a
        v2 = c - a
        
        # 叉乘得到法向量
        cross = np.cross(v1, v2)
        
        # 三角形面积 = |叉乘| / 2
        area = np.linalg.norm(cross) / 2.0
        surface_area += area
    
    return surface_area


def reconstruct_surface_from_pointcloud(ply_file):
    """
    从点云文件重建三维网格表面
    使用Poisson或Ball Pivoting算法
    """
    print(f"读取点云: {ply_file}")
    
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_file)
    
    if not pcd.has_points():
        print(f"错误: 点云为空")
        return None
    
    print(f"点云大小: {len(pcd.points)} 个点")
    
    # 估计法向量（如果没有的话）
    if not pcd.has_normals():
        print("估计法向量...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        # 定向法向量
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
    print("重建表面...")
    
    # 方法1: Poisson表面重建（适合闭合表面）
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        # 移除低密度顶点（可选）
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
    except Exception as e:
        print(f"Poisson重建失败: {e}")
        print("尝试使用Ball Pivoting算法...")
        
        # 方法2: Ball Pivoting算法（适合开放表面）
        try:
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist
            
            radii = [radius, radius * 2, radius * 4]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        except Exception as e:
            print(f"Ball Pivoting重建失败: {e}")
            return None
    
    if not mesh.has_triangles():
        print("错误: 网格重建失败")
        return None
    
    print(f"网格: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个面片")
    
    return mesh


def calculate_leaf_areas(ply_file):
    """
    计算单个叶片的真实表面积和投影面积
    
    Returns:
        tuple: (真实表面积, 投影面积) 或 (None, None)
    """
    # 从点云重建网格
    mesh = reconstruct_surface_from_pointcloud(ply_file)
    
    if mesh is None:
        return None, None
    
    # 计算真实三维表面积
    true_surface_area = get_surface_area(mesh)
    print(f"真实表面积: {true_surface_area:.6f}")
    
    # 创建投影网格：将所有顶点的Z坐标设为0
    projected_mesh = o3d.geometry.TriangleMesh(mesh)
    vertices = np.asarray(projected_mesh.vertices)
    vertices[:, 2] = 0  # 投影到XY平面
    projected_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # 计算投影面积
    projected_area = get_surface_area(projected_mesh)
    print(f"投影面积: {projected_area:.6f}")
    
    return true_surface_area, projected_area


def process_dataset(root_path, output_csv):
    """
    批量处理整个数据集
    
    Args:
        root_path: 数据集根目录，可以是：
                   - PlantDivision目录: '/path/to/PlantDivision'
                   - 某个编号目录: '/path/to/PlantDivision/4'
                   - 某个日期目录: '/path/to/PlantDivision/4/2025_04_13'
        output_csv: 输出CSV文件路径
    """
    root = Path(root_path)
    
    if not root.exists():
        print(f"错误: 路径不存在 {root_path}")
        return
    
    print(f"扫描路径: {root}")
    print(f"路径类型: {'目录' if root.is_dir() else '文件'}")
    
    # 收集所有叶片文件
    leaf_files = []
    
    # 递归查找所有leaf_*.ply文件
    print("\n开始查找叶片文件...")
    
    if root.is_dir():
        # 使用递归glob查找所有leaf_*.ply文件
        all_ply_files = list(root.rglob("leaf_*.ply"))
        print(f"找到 {len(all_ply_files)} 个leaf_*.ply文件")
        
        # 过滤掉包含'mesh'的文件
        for ply_file in all_ply_files:
            if 'mesh' not in ply_file.name:
                leaf_files.append(ply_file)
                print(f"  ✓ {ply_file}")
            else:
                print(f"  ✗ {ply_file} (跳过mesh文件)")
    
    print(f"\n总共需要处理: {len(leaf_files)} 个叶片文件\n")
    
    # 创建输出CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            '文件路径', '区域编号', '实例编号', 
            '真实表面积', '投影面积', '状态'
        ])
        
        # 处理每个文件
        for ply_file in tqdm(leaf_files, desc="处理叶片"):
            print(f"\n{'='*60}")
            
            # 解析文件名: leaf_001_1.ply
            filename = ply_file.name
            parts = filename.replace('.ply', '').split('_')
            
            if len(parts) >= 3:
                region = parts[1]  # 001
                instance = parts[2]  # 1
            else:
                region = 'unknown'
                instance = 'unknown'
            
            # 计算表面积
            try:
                true_area, projected_area = calculate_leaf_areas(str(ply_file))
                
                if true_area is not None:
                    writer.writerow([
                        str(ply_file), region, instance,
                        f"{true_area:.6f}", f"{projected_area:.6f}",
                        '成功'
                    ])
                    csvfile.flush()  # 实时写入
                else:
                    writer.writerow([
                        str(ply_file), region, instance,
                        '', '', '重建失败'
                    ])
                    csvfile.flush()
            
            except Exception as e:
                print(f"处理失败: {e}")
                writer.writerow([
                    str(ply_file), region, instance,
                    '', '', f'错误: {str(e)}'
                ])
                csvfile.flush()
    
    print(f"\n{'='*60}")
    print(f"处理完成! 结果已保存到: {output_csv}")


def main():
    """主函数"""
    
    # 设置数据集路径 - 可以修改为任意层级的路径
    # 示例1: 整个数据集
    # dataset_root = "/mnt/sdc1/acailo/SPFormer/data/myplants/raw/PlantDivision"
    
    # 示例2: 某个植株（编号4）
    # dataset_root = "/mnt/sdc1/acailo/SPFormer/data/myplants/raw/PlantDivision/4"
    
    # 示例3: 某个植株的某个日期
    dataset_root = "/mnt/sdc1/acailo/SPFormer/data/myplants/raw/PlantDivision/4/2025_04_13"
    
    # 输出文件路径
    output_csv = "./leaf_area_xdd.csv"
    
    print("="*60)
    print("叶片表面积计算 - 方法1")
    print("="*60)
    print(f"数据集路径: {dataset_root}")
    print(f"输出文件: {output_csv}")
    print("="*60)
    
    # 处理数据集
    process_dataset(dataset_root, output_csv)


if __name__ == "__main__":
    main()
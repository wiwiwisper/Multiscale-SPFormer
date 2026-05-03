#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
叶片表面积批量计算程序
基于原始C++代码的计算逻辑，用Python实现
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import trimesh
from scipy.spatial import Delaunay
import warnings
warnings.filterwarnings('ignore')


def read_ply_file(file_path):
    """
    读取PLY文件
    """
    try:
        # 使用trimesh读取点云或网格
        mesh = trimesh.load(file_path)
        return mesh
    except Exception as e:
        print(f"  错误: 无法读取文件 - {e}")
        return None


def points_to_mesh(points):
    """
    将点云转换为网格（使用Delaunay三角化）
    如果点云是平面的，使用2D Delaunay
    """
    if len(points) < 3:
        return None
    
    try:
        # 检查点是否基本共面（检查z方向的变化）
        z_range = np.ptp(points[:, 2])
        
        if z_range < 1e-6:  # 近似平面
            # 使用2D Delaunay三角化
            points_2d = points[:, :2]
            tri = Delaunay(points_2d)
            
            # 创建3D网格
            mesh = trimesh.Trimesh(vertices=points, faces=tri.simplices)
        else:
            # 使用3D凸包
            mesh = trimesh.convex.convex_hull(points)
        
        return mesh
    except Exception as e:
        print(f"  网格重建失败: {e}")
        return None


def get_surface_area(mesh):
    """
    计算网格的表面积
    与原始C++代码逻辑一致：计算每个三角形面积的和
    """
    if mesh is None or not hasattr(mesh, 'faces'):
        return None
    
    try:
        # 获取所有三角形的顶点
        vertices = mesh.vertices
        faces = mesh.faces
        
        total_area = 0.0
        
        for face in faces:
            # 获取三角形的三个顶点
            a = vertices[face[0]]
            b = vertices[face[1]]
            c = vertices[face[2]]
            
            # 计算两条边向量
            ab = b - a
            ac = c - a
            
            # 计算叉积（法向量）
            n = np.cross(ab, ac)
            
            # 如果z分量为负，翻转法向量（与原代码一致）
            if n[2] < 0:
                n = -n
            
            # 计算三角形面积 = |叉积| / 2
            area = np.sqrt(np.sum(n**2)) / 2.0
            total_area += area
        
        return total_area
    except Exception as e:
        print(f"  表面积计算失败: {e}")
        return None


def parse_leaf_filename(filename):
    """
    解析叶子文件名，提取区域编号和实例编号
    例如: leaf_001_2.ply -> 区域001, 实例2
    """
    import re
    
    # 使用正则表达式匹配 leaf_XXX_Y.ply 格式
    match = re.match(r'leaf_(\d+)_(\d+)\.ply', filename)
    
    if match:
        region = match.group(1)  # 区域编号
        instance = match.group(2)  # 实例编号
        return region, instance
    
    return None, None


def process_single_file(file_path):
    """
    处理单个PLY文件
    """
    result = {
        'file_path': str(file_path),
        'filename': file_path.name,
        'region': None,
        'instance': None,
        '3d_surface_area': None,
        '2d_projected_area': None,
        'status': '未知错误'
    }
    
    # 解析文件名
    region, instance = parse_leaf_filename(file_path.name)
    result['region'] = region
    result['instance'] = instance
    
    # 读取文件
    data = read_ply_file(file_path)
    if data is None:
        result['status'] = '读取失败'
        return result
    
    # 如果是点云，转换为网格
    if isinstance(data, trimesh.PointCloud):
        mesh = points_to_mesh(data.vertices)
        if mesh is None:
            result['status'] = '网格重建失败'
            return result
    else:
        mesh = data
    
    # 计算3D表面积
    area_3d = get_surface_area(mesh)
    if area_3d is None:
        result['status'] = '3D面积计算失败'
        return result
    
    result['3d_surface_area'] = area_3d
    
    # 创建投影网格（所有点的z坐标设为0）
    projected_vertices = mesh.vertices.copy()
    projected_vertices[:, 2] = 0  # 将z坐标设为0
    projected_mesh = trimesh.Trimesh(vertices=projected_vertices, faces=mesh.faces)
    
    # 计算2D投影面积
    area_2d = get_surface_area(projected_mesh)
    if area_2d is None:
        result['status'] = '2D面积计算失败'
        return result
    
    result['2d_projected_area'] = area_2d
    result['status'] = '成功'
    
    return result


def find_all_ply_files(root_path):
    """
    递归查找所有PLY文件
    只处理 leaf_ 开头的文件，跳过 stem 和 mesh 文件
    """
    ply_files = []
    root = Path(root_path)
    
    for ply_file in root.rglob('*.ply'):
        if ply_file.is_file():
            filename = ply_file.name
            
            # 只处理以 leaf_ 开头的文件
            if not filename.startswith('leaf_'):
                continue
            
            # 跳过 mesh 文件（back_mesh, front_mesh, full_mesh）
            if 'mesh' in filename:
                continue
            
            ply_files.append(ply_file)
    
    return sorted(ply_files)


def main():
    """
    主函数
    """
    # ==================== 配置参数 ====================
    # 数据集根路径
    root_path = "/mnt/sdc1/acailo/SPFormer/data/myplants/raw/PlantDivision/4/2025_04_13"
    
    # 输出路径
    output_path = "/mnt/sdc1/acailo/SPFormer/data/myplants/"
    
    # 输出文件名
    output_filename = "surface_areas_results.csv"
    # ==================================================
    
    print("=" * 60)
    print("叶片表面积批量计算程序")
    print("=" * 60)
    
    # 查找所有PLY文件
    print(f"\n正在扫描目录: {root_path}")
    ply_files = find_all_ply_files(root_path)
    
    if len(ply_files) == 0:
        print("错误: 未找到任何PLY文件！")
        return
    
    print(f"找到 {len(ply_files)} 个PLY文件\n")
    
    # 处理所有文件
    results = []
    success_count = 0
    fail_count = 0
    
    for file_path in tqdm(ply_files, desc="处理进度"):
        result = process_single_file(file_path)
        results.append(result)
        
        if result['status'] == '成功':
            success_count += 1
        else:
            fail_count += 1
    
    # 保存结果到CSV
    df = pd.DataFrame(results)
    output_file = os.path.join(output_path, output_filename)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"总文件数: {len(ply_files)}")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"\n结果已保存到: {output_file}")
    print("=" * 60)
    
    # 显示统计信息
    if success_count > 0:
        successful_results = df[df['status'] == '成功']
        
        print(f"\n�� 总体统计:")
        print(f"  3D表面积:")
        print(f"    平均值: {successful_results['3d_surface_area'].mean():.6f}")
        print(f"    最小值: {successful_results['3d_surface_area'].min():.6f}")
        print(f"    最大值: {successful_results['3d_surface_area'].max():.6f}")
        print(f"    标准差: {successful_results['3d_surface_area'].std():.6f}")
        
        print(f"\n  2D投影面积:")
        print(f"    平均值: {successful_results['2d_projected_area'].mean():.6f}")
        print(f"    最小值: {successful_results['2d_projected_area'].min():.6f}")
        print(f"    最大值: {successful_results['2d_projected_area'].max():.6f}")
        print(f"    标准差: {successful_results['2d_projected_area'].std():.6f}")
        
        # 按区域统计
        if 'region' in successful_results.columns:
            print(f"\n�� 按区域统计:")
            region_stats = successful_results.groupby('region').agg({
                '3d_surface_area': ['count', 'mean', 'std'],
                '2d_projected_area': ['mean', 'std']
            }).round(6)
            
            print("\n区域 | 叶片数 | 3D面积(均值±标准差) | 2D面积(均值±标准差)")
            print("-" * 70)
            for region in sorted(successful_results['region'].unique()):
                region_data = successful_results[successful_results['region'] == region]
                count = len(region_data)
                area_3d_mean = region_data['3d_surface_area'].mean()
                area_3d_std = region_data['3d_surface_area'].std()
                area_2d_mean = region_data['2d_projected_area'].mean()
                area_2d_std = region_data['2d_projected_area'].std()
                
                print(f"{region:^4} | {count:^6} | {area_3d_mean:.4f}±{area_3d_std:.4f} | {area_2d_mean:.4f}±{area_2d_std:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
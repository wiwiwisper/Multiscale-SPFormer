#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
叶片表面积计算 - 方法2
基于导师提供的seg_front_back.py代码
使用KMeans对面片法向量聚类，分离正反面后计算单面面积
"""

import open3d as o3d
import numpy as np
import torch
from sklearn.cluster import KMeans
import csv
from pathlib import Path
from tqdm import tqdm

# 禁用Open3D的警告信息
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)



def process_single_leaf(file_path, visualize=False, save_point_clouds=False):
    """
    处理单个叶片文件（原始代码的核心逻辑）
    
    Args:
        file_path: .ply文件路径
        visualize: 是否显示可视化窗口
        save_point_clouds: 是否保存正反面点云为.ply文件
    
    Returns:
        dict: 包含正面面积、反面面积等信息
    """
    print(f"\n处理文件: {file_path}")
    
    # 加载 .ply 文件
    # 需要包含三角网格mesh信息
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        mesh = o3d.io.read_triangle_mesh(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None
    
    # 检查是否有网格信息
    if not mesh.has_triangles():
        # 如果没有网格，需要从点云重建
        print("从点云重建表面...")
        
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)
        
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9
            )
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            print("  ✓ Poisson重建成功")
        except:
            print("  Poisson重建失败，使用Ball Pivoting...")
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist
            radii = [radius, radius * 2, radius * 4]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
            print("  ✓ Ball Pivoting重建成功")
        
        if not mesh.has_triangles():
            print("  ✗ 表面重建失败")
            return None
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    print(f"网格信息: {len(vertices)} 个顶点, {len(triangles)} 个三角面片")
    
    # 计算面的法向量
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    
    # 转换为 PyTorch 张量
    v0 = torch.tensor(v0, dtype=torch.float32)
    v1 = torch.tensor(v1, dtype=torch.float32)
    v2 = torch.tensor(v2, dtype=torch.float32)
    
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # 叉乘
    norm_lengths = torch.norm(normals, dim=1, keepdim=True)  # 计算模
    
    # 避免除以零
    normals = torch.where(
        norm_lengths > 0,
        normals / norm_lengths,
        torch.zeros_like(normals),
    )
    
    # 转回 CPU
    normals = normals.cpu().numpy()
    
    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(normals)
    
    # 按照聚类结果分组
    group1_indices = np.where(labels == 0)[0]
    group2_indices = np.where(labels == 1)[0]
    
    print(f"Group 1 has {len(group1_indices)} faces")
    print(f"Group 2 has {len(group2_indices)} faces")
    
    # 找出正面的叶子（面片数量多的是正面）
    if len(group1_indices) > len(group2_indices):
        selected_group_faces = triangles[labels == 0]
        front_group_indices = group1_indices
        back_group_indices = group2_indices
    else:
        selected_group_faces = triangles[labels == 1]
        front_group_indices = group2_indices
        back_group_indices = group1_indices
    
    group_v0 = vertices[selected_group_faces[:, 0]]
    group_v1 = vertices[selected_group_faces[:, 1]]
    group_v2 = vertices[selected_group_faces[:, 2]]
    
    # 三角形面积公式(正面的)
    group_areas = 0.5 * np.linalg.norm(np.cross(group_v1 - group_v0, group_v2 - group_v0), axis=1)
    
    # 总面积（正面）
    total_area_front = np.sum(group_areas)
    
    print(f"正面表面积: {total_area_front:.6f}")
    
    # 计算反面面积（用于对比）
    back_faces = triangles[back_group_indices]
    back_v0 = vertices[back_faces[:, 0]]
    back_v1 = vertices[back_faces[:, 1]]
    back_v2 = vertices[back_faces[:, 2]]
    back_areas = 0.5 * np.linalg.norm(np.cross(back_v1 - back_v0, back_v2 - back_v0), axis=1)
    total_area_back = np.sum(back_areas)
    
    print(f"反面表面积: {total_area_back:.6f}")
    
    # ============== 可视化部分 ==============
    if visualize or save_point_clouds:
        # 创建正面和反面的网格（用于可视化）
        if len(group1_indices) > len(group2_indices):
            mesh1 = o3d.geometry.TriangleMesh()
            mesh1.vertices = mesh.vertices
            mesh1.triangles = o3d.utility.Vector3iVector(triangles[group1_indices])
            mesh1.paint_uniform_color([1, 0, 0])  # 红色 - 正面
            
            mesh2 = o3d.geometry.TriangleMesh()
            mesh2.vertices = mesh.vertices
            mesh2.triangles = o3d.utility.Vector3iVector(triangles[group2_indices])
            mesh2.paint_uniform_color([0, 0, 1])  # 蓝色 - 反面
        else:
            mesh1 = o3d.geometry.TriangleMesh()
            mesh1.vertices = mesh.vertices
            mesh1.triangles = o3d.utility.Vector3iVector(triangles[group2_indices])
            mesh1.paint_uniform_color([1, 0, 0])  # 红色 - 正面
            
            mesh2 = o3d.geometry.TriangleMesh()
            mesh2.vertices = mesh.vertices
            mesh2.triangles = o3d.utility.Vector3iVector(triangles[group1_indices])
            mesh2.paint_uniform_color([0, 0, 1])  # 蓝色 - 反面
        
        # 提取正面和反面的点
        front_faces_all = triangles[front_group_indices]
        back_faces_all = triangles[back_group_indices]
        
        front_points = np.unique(front_faces_all)
        back_points = np.unique(back_faces_all)
        
        front_vertices = vertices[front_points]
        back_vertices = vertices[back_points]
        
        # 创建点云对象
        front_pcd = o3d.geometry.PointCloud()
        front_pcd.points = o3d.utility.Vector3dVector(front_vertices)
        front_pcd.paint_uniform_color([1, 0, 0])  # 红色
        
        back_pcd = o3d.geometry.PointCloud()
        back_pcd.points = o3d.utility.Vector3dVector(back_vertices)
        back_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
        
        # 保存点云文件
        if save_point_clouds:
            from pathlib import Path
            file_stem = Path(file_path).stem
            output_dir = Path(file_path).parent / "separated_point_clouds"
            output_dir.mkdir(exist_ok=True)
            
            front_output = output_dir / f"{file_stem}_front_points.ply"
            back_output = output_dir / f"{file_stem}_back_points.ply"
            
            o3d.io.write_point_cloud(str(front_output), front_pcd)
            o3d.io.write_point_cloud(str(back_output), back_pcd)
            
            print(f"保存正面点云: {front_output}")
            print(f"保存反面点云: {back_output}")
        
        # 显示可视化窗口
        if visualize:
            print("\n显示可视化窗口...")
            print("红色=正面, 蓝色=反面")
            
            # 显示正反面网格
            o3d.visualization.draw_geometries(
                [mesh1, mesh2], 
                window_name="正反面网格 (红=正面, 蓝=反面)",
                width=800, height=600
            )
            
            # 显示正面点云
            o3d.visualization.draw_geometries(
                [front_pcd], 
                window_name="正面点云",
                width=800, height=600
            )
            
            # 显示反面点云
            o3d.visualization.draw_geometries(
                [back_pcd], 
                window_name="反面点云",
                width=800, height=600
            )
            
            # 显示正反面点云叠加
            o3d.visualization.draw_geometries(
                [front_pcd, back_pcd], 
                window_name="正反面点云叠加 (红=正面, 蓝=反面)",
                width=800, height=600
            )
    
    # 返回结果
    result = {
        'front_area': total_area_front,
        'back_area': total_area_back,
        'front_faces': len(front_group_indices),
        'back_faces': len(back_group_indices),
        'total_faces': len(triangles)
    }
    
    return result


def process_dataset(root_path, output_csv, visualize=False, save_point_clouds=False):
    """
    批量处理整个数据集
    
    Args:
        root_path: 数据集根目录
        output_csv: 输出CSV文件路径
        visualize: 是否显示可视化窗口（批量处理时建议关闭）
        save_point_clouds: 是否保存正反面分离的点云文件
    """
    root = Path(root_path)
    
    if not root.exists():
        print(f"错误: 路径不存在 {root_path}")
        return
    
    print(f"扫描路径: {root}")
    
    # 收集所有叶片文件
    leaf_files = []
    
    if root.is_dir():
        all_ply_files = list(root.rglob("leaf_*.ply"))
        print(f"找到 {len(all_ply_files)} 个leaf_*.ply文件")
        
        for ply_file in all_ply_files:
            if 'mesh' not in ply_file.name:
                leaf_files.append(ply_file)
                print(f"  ✓ {ply_file}")
    
    print(f"\n总共需要处理: {len(leaf_files)} 个叶片文件\n")
    
    if len(leaf_files) == 0:
        print("未找到任何叶片文件！")
        return
    
    # 创建输出CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            '文件路径', '区域编号', '实例编号', 
            '正面面积', '反面面积', '正面面片数', '反面面片数', '总面片数', '状态'
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
                result = process_single_leaf(
                    str(ply_file), 
                    visualize=visualize, 
                    save_point_clouds=save_point_clouds
                )
                
                if result is not None:
                    writer.writerow([
                        str(ply_file), region, instance,
                        f"{result['front_area']:.6f}",
                        f"{result['back_area']:.6f}",
                        result['front_faces'],
                        result['back_faces'],
                        result['total_faces'],
                        '成功'
                    ])
                    csvfile.flush()
                else:
                    writer.writerow([
                        str(ply_file), region, instance,
                        '', '', '', '', '', '处理失败'
                    ])
                    csvfile.flush()
            
            except Exception as e:
                print(f"处理失败: {e}")
                writer.writerow([
                    str(ply_file), region, instance,
                    '', '', '', '', '', f'错误: {str(e)}'
                ])
                csvfile.flush()
    
    print(f"\n{'='*60}")
    print(f"处理完成! 结果已保存到: {output_csv}")


def main():
    """主函数"""
    
    # 设置数据集路径
    dataset_root = "/mnt/sdc1/acailo/SPFormer/data/myplants/raw/PlantDivision/4/2025_04_13"
    output_csv = "./leaf_area_dyt.csv"
    
    # 可视化设置
    visualize = False  # 批量处理时建议设为False，单个文件测试时可以设为True
    save_point_clouds = False  # 是否保存分离后的正反面点云
    
    print("="*60)
    print("叶片表面积计算 - 方法2")
    print("基于KMeans法向量聚类分离正反面")
    print("="*60)
    print(f"数据集路径: {dataset_root}")
    print(f"输出文件: {output_csv}")
    print(f"可视化: {'开启' if visualize else '关闭'}")
    print(f"保存点云: {'是' if save_point_clouds else '否'}")
    print("="*60)
    
    # 处理数据集
    process_dataset(dataset_root, output_csv, visualize, save_point_clouds)


# ============== 单文件测试函数（带完整可视化） ==============
def test_single_file(file_path):
    """
    测试单个文件并显示完整可视化
    用于调试和查看单个叶片的分离效果
    
    Args:
        file_path: 单个.ply文件的路径
    """
    print("="*60)
    print("单文件测试模式（带可视化）")
    print("="*60)
    
    result = process_single_leaf(
        file_path, 
        visualize=True,  # 开启可视化
        save_point_clouds=True  # 保存点云
    )
    
    if result:
        print("\n结果:")
        print(f"  正面面积: {result['front_area']:.6f} dm² = {result['front_area']*100:.2f} cm²")
        print(f"  反面面积: {result['back_area']:.6f} dm² = {result['back_area']*100:.2f} cm²")
        print(f"  正面面片数: {result['front_faces']}")
        print(f"  反面面片数: {result['back_faces']}")


if __name__ == "__main__":
    # 批量处理模式
    # main()
    
    # 单文件测试模式（取消注释下面的代码来测试单个文件）
    test_single_file("/mnt/sdc1/acailo/SPFormer/data/myplants/raw/PlantDivision/4/2025_04_13/leaf_001_1.ply")
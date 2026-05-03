import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import os
from pathlib import Path

def calculate_single_leaf_area(file_path, visualize=False):
    """
    计算单个叶片的表面积
    
    参数:
        file_path: ply文件路径
        visualize: 是否可视化结果
    
    返回:
        total_area: 叶片总面积（平方米）
    """
    print(f"\n处理文件: {file_path}")
    
    # 1. 加载点云数据
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"  点云包含 {len(pcd.points)} 个点")
    
    # 2. 估计法向量（因为原始文件的法向量都是0）
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # 3. 从点云重建三角网格（使用泊松表面重建）
    print("  正在重建网格...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # 移除低密度的顶点（通常是噪声）
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print(f"  网格包含 {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")
    
    # 4. 获取网格数据
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(triangles) == 0:
        print("  警告: 无法重建网格，尝试使用Ball Pivoting算法...")
        # 备用方案：使用Ball Pivoting算法
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        print(f"  Ball Pivoting: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")
    
    if len(triangles) == 0:
        print("  错误: 无法重建网格")
        return 0.0
    
    # 5. 计算面的法向量
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    
    # 使用numpy计算叉乘
    normals = np.cross(v1 - v0, v2 - v0)
    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    
    # 归一化法向量，避免除以零
    normals = np.where(
        norm_lengths > 0,
        normals / norm_lengths,
        np.zeros_like(normals)
    )
    
    # 6. 使用KMeans聚类分离正面和反面
    print("  正在分离正反面...")
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(normals)
    
    group1_indices = np.where(labels == 0)[0]
    group2_indices = np.where(labels == 1)[0]
    
    print(f"  正面: {max(len(group1_indices), len(group2_indices))} 个面")
    print(f"  反面: {min(len(group1_indices), len(group2_indices))} 个面")
    
    # 7. 选择面数较多的一组作为正面（因为正面通常更完整）
    if len(group1_indices) > len(group2_indices):
        selected_group_faces = triangles[labels == 0]
        front_group_indices = group1_indices
        back_group_indices = group2_indices
    else:
        selected_group_faces = triangles[labels == 1]
        front_group_indices = group2_indices
        back_group_indices = group1_indices
    
    # 8. 计算正面的面积
    group_v0 = vertices[selected_group_faces[:, 0]]
    group_v1 = vertices[selected_group_faces[:, 1]]
    group_v2 = vertices[selected_group_faces[:, 2]]
    
    # 三角形面积公式：0.5 * ||(v1-v0) × (v2-v0)||
    group_areas = 0.5 * np.linalg.norm(np.cross(group_v1 - group_v0, group_v2 - group_v0), axis=1)
    
    # 总面积
    total_area = np.sum(group_areas)
    
    print(f"  叶片面积: {total_area:.6f} 平方米")
    print(f"  叶片面积: {total_area * 10000:.2f} 平方厘米")
    
    # 9. 可视化（可选）
    if visualize:
        # 创建正反面mesh用于可视化
        mesh_front = o3d.geometry.TriangleMesh()
        mesh_front.vertices = mesh.vertices
        mesh_front.triangles = o3d.utility.Vector3iVector(triangles[front_group_indices])
        mesh_front.paint_uniform_color([1, 0, 0])  # 红色
        
        mesh_back = o3d.geometry.TriangleMesh()
        mesh_back.vertices = mesh.vertices
        mesh_back.triangles = o3d.utility.Vector3iVector(triangles[back_group_indices])
        mesh_back.paint_uniform_color([0, 0, 1])  # 蓝色
        
        o3d.visualization.draw_geometries([mesh_front, mesh_back], 
                                         window_name=f"正反面分离 - {Path(file_path).name}")
    
    return total_area


def batch_calculate_leaf_areas(root_dir, output_csv="leaf_areas.csv"):
    """
    批量计算数据集中所有叶片的面积
    
    参数:
        root_dir: 数据集根目录（PlantDivision文件夹路径）
        output_csv: 输出CSV文件路径
    """
    results = []
    
    # 遍历所有PLY文件
    root_path = Path(root_dir)
    ply_files = list(root_path.rglob("leaf_*.ply"))
    
    print(f"找到 {len(ply_files)} 个叶片文件")
    
    for i, ply_file in enumerate(ply_files, 1):
        print(f"\n进度: {i}/{len(ply_files)}")
        
        try:
            # 解析文件路径信息
            relative_path = ply_file.relative_to(root_path)
            parts = relative_path.parts
            
            # 获取区域编号、日期、文件名
            if len(parts) >= 3:
                region_num = parts[0]  # 如 "4"
                date_str = parts[1]    # 如 "2025_04_13"
                file_name = parts[2]   # 如 "leaf_004_2.ply"
            else:
                region_num = "unknown"
                date_str = "unknown"
                file_name = ply_file.name
            
            # 计算面积
            area = calculate_single_leaf_area(str(ply_file), visualize=False)
            
            # 保存结果
            results.append({
                "file_path": str(ply_file.relative_to(root_path)),
                "region": region_num,
                "date": date_str,
                "filename": file_name,
                "area_m2": area,
                "area_cm2": area * 10000
            })
            
        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                "file_path": str(ply_file.relative_to(root_path)),
                "region": "error",
                "date": "error",
                "filename": ply_file.name,
                "area_m2": 0,
                "area_cm2": 0
            })
    
    # 保存结果到CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"\n\n结果已保存到: {output_csv}")
    print(f"总共处理了 {len(results)} 个文件")
    print(f"成功: {sum(1 for r in results if r['area_m2'] > 0)}")
    print(f"失败: {sum(1 for r in results if r['area_m2'] == 0)}")
    
    # 显示统计信息
    if len(df[df['area_cm2'] > 0]) > 0:
        print(f"\n面积统计:")
        print(f"  平均面积: {df[df['area_cm2'] > 0]['area_cm2'].mean():.2f} 平方厘米")
        print(f"  最小面积: {df[df['area_cm2'] > 0]['area_cm2'].min():.2f} 平方厘米")
        print(f"  最大面积: {df[df['area_cm2'] > 0]['area_cm2'].max():.2f} 平方厘米")
    
    return df


# 使用示例
if __name__ == "__main__":
    # # 示例1: 计算单个文件
    # single_file = "/mnt/sdc1/acailo/SPFormer/data/myplants/raw/PlantDivision/4/2025_04_13/leaf_003_1.ply"
    # area = calculate_single_leaf_area(single_file, visualize=True)
    
    # 示例2: 批量计算整个数据集
    # 修改这里为你的数据集根目录路径
    dataset_root = "/mnt/sdc1/acailo/SPFormer/data/myplants/raw/PlantDivision/4/2025_04_13"
    
    # 如果只想测试，可以先用单个文件
    # print("=" * 60)
    # print("测试单个文件")
    # print("=" * 60)
    # test_file = "/mnt/sdc1/acailo/SPFormer/data/myplants/raw/PlantDivision/4/2025_04_13/leaf_001_2.ply"
    # area = calculate_single_leaf_area(test_file, visualize=False)
    
    # 如果测试成功，取消下面的注释来批量处理
    print("\n" + "=" * 60)
    print("批量处理数据集")
    print("=" * 60)
    df = batch_calculate_leaf_areas(dataset_root, output_csv="leaf_areas.csv")
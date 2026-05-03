"""
参数对比测试：找到最佳的重建参数
"""
import open3d as o3d
import sys
sys.path.append('.')
from dyt_leaf_area import *

# 测试文件
test_file = "data/myplants/raw/PlantDivision/4/2025_03_12/leaf_004_2.ply"

# 测试不同参数组合
param_configs = [
    # (depth, density_threshold, 描述)
    (9, 0.1, "精细+保守清理"),
    (9, 0.3, "精细+适中清理"),
    (9, 0.5, "精细+激进清理"),
    (8, 0.1, "适中+保守清理"),
    (8, 0.3, "适中+适中清理"),
    (8, 0.5, "适中+激进清理"),
    (7, 0.3, "粗糙+适中清理"),
]

print("\n" + "="*70)
print("参数对比测试")
print("="*70)

results = []

for depth, density_threshold, desc in param_configs:
    print(f"\n{'='*70}")
    print(f"测试: {desc} (depth={depth}, density_threshold={density_threshold})")
    print(f"{'='*70}")
    
    try:
        result = calculate_leaf_area(
            test_file,
            method='poisson',
            depth=depth,
            density_threshold=density_threshold,
            visualize=False,  # 先不可视化，最后再看最好的
            save_mesh=False,
            area_ratio_threshold=0.7
        )
        
        results.append({
            'depth': depth,
            'density': density_threshold,
            'desc': desc,
            'area': result['area'],
            'quality': result.get('separation_quality', 0),
            'method': result['method']
        })
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        continue

# 输出对比结果
print("\n" + "="*70)
print("参数对比结果汇总")
print("="*70)
print(f"{'描述':<20s}  {'depth':<6s}  {'density':<8s}  {'面积':<12s}  {'质量':<8s}  {'方法'}")
print("="*70)

for r in results:
    print(f"{r['desc']:<20s}  {r['depth']:<6d}  {r['density']:<8.1f}  "
          f"{r['area']:<12.6f}  {r['quality']:<8.3f}  {r['method']}")

# 找出质量最好的
best = max(results, key=lambda x: x['quality'])
print(f"\n推荐参数: depth={best['depth']}, density_threshold={best['density']}")
print(f"  → {best['desc']}")
print(f"  → 面积: {best['area']:.6f}, 分离质量: {best['quality']:.3f}")

# 用最佳参数重新可视化
print(f"\n用最佳参数可视化...")
result = calculate_leaf_area(
    test_file,
    method='poisson',
    depth=best['depth'],
    density_threshold=best['density'],
    visualize=True,  # 可视化
    save_mesh=True,  # 保存
    area_ratio_threshold=0.7
)
"""
车针形状观察：框选车针 → 邻域提取 → 法向估计 → 表面重建 → 网格可视化
不做任何形状假设，重建出什么就是什么
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

CLOUD_PATH = "point_cloud.ply"   # ← 改成真实文件名
NEIGHBOR = 15.0                  # 框选点周围保留范围(mm)

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

cloud = o3d.io.read_point_cloud(CLOUD_PATH)
print(f"点云: {len(cloud.points)}点")

# ---- ① 框选车针（和主程序一样的操作）----
vis = o3d.visualization.VisualizerWithVertexSelection()
vis.create_window(window_name="Shift+左键框选整段车针，选完按Q")
vis.add_geometry(cloud)
vis.run()
picked = [p.index for p in vis.get_picked_points()]
vis.destroy_window()
assert len(picked) >= 10, f"只选了{len(picked)}个点，至少10个，重跑"
picked_pts = np.asarray(cloud.points)[picked]

# ---- ② 邻域提取车针点 ----
d, _ = cKDTree(picked_pts).query(np.asarray(cloud.points))
drill = cloud.select_by_index(np.where(d < NEIGHBOR)[0])
print(f"车针局部: {len(drill.points)}点")

# ---- ③ 法向估计，并统一朝外（远离PCA轴线）----
_, eigvec = np.linalg.eigh(np.cov(picked_pts.T))
axis = eigvec[:, -1]
drill.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
pts = np.asarray(drill.points)
nrm = np.asarray(drill.normals)
# 每点的"朝外"方向 = 离开轴线的径向
h = (pts - pts.mean(0)) @ axis
radial = (pts - pts.mean(0)) - np.outer(h, axis)
radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
flip = (nrm * radial).sum(1) < 0
nrm[flip] = -nrm[flip]
drill.normals = o3d.utility.Vector3dVector(nrm)

# ---- ④ 表面重建：泊松（光滑封闭）+ 球 pivoting（忠实原始点）----
print("泊松重建中...")
mesh_p, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    drill, depth=9
)
densities = np.asarray(densities)
# 泊松会在没点的地方编造表面，按密度裁掉最低10%的顶点
mesh_p.remove_vertices_by_mask(densities < np.quantile(densities, 0.1))
mesh_p.compute_vertex_normals()
mesh_p.paint_uniform_color([1.0, 0.3, 0.3])

print("球 pivoting 重建中...")
radii = o3d.utility.DoubleVector([0.1, 0.2, 0.4, 0.8])
mesh_b = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(drill, radii)
mesh_b.compute_vertex_normals()
mesh_b.paint_uniform_color([0.3, 0.8, 1.0])

# ---- ⑤ 显示 + 存盘 ----
o3d.visualization.draw_geometries([drill], window_name="① 原始车针点云")
o3d.visualization.draw_geometries(
    [mesh_p, drill], window_name="② 泊松重建（红）+ 原始点云"
)
o3d.visualization.draw_geometries(
    [mesh_b, drill], window_name="③ 球pivoting重建（蓝）+ 原始点云"
)
o3d.io.write_triangle_mesh("bur_shape_poisson.ply", mesh_p)
o3d.io.write_triangle_mesh("bur_shape_bpa.ply", mesh_b)
print("已保存 bur_shape_poisson.ply / bur_shape_bpa.ply（可拖进MeshInspector细看）")
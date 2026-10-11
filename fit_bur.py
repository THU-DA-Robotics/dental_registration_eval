# -*- coding: utf-8 -*-
"""
车针位姿拟合 (v1.0)
=====================================================
功能: 读取配准矩阵 M (scan_to_cloud_M.npz), 在点云中分割车针并拟合
      (圆柱 RANSAC + Gauss-Newton 精修 + 尖端定位 + 可视化)。
输入: scan_to_cloud_M.npz (由 register_scan_to_cloud.py 生成)
      original_scan.ply, merged_cloud_clean.ply
输出: 打印车针尖端位置/朝向, 可视化窗口(按 C 显隐车针模型)。
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import os

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ================== 参数 ==================
SCAN_PATH = "original_scan.ply"
CLOUD_PATH = "merged_cloud_clean.ply"
M_PATH = "scan_to_cloud_M.npz"

DIST_TEETH = 1.5
R_DRILL = 0.5
VISUALIZE = True

# ================== 工具函数 ==================
def pick_idx(geometry, msg):
    """在Open3D几何体上选顶点,按Q结束并返回顶点下标。"""
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=msg)
    vis.add_geometry(geometry)
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()
    return [p.index for p in picked]
# ================== 加载 ==================
scan = o3d.io.read_triangle_mesh(SCAN_PATH)
cloud = o3d.io.read_point_cloud(CLOUD_PATH)
scan.compute_vertex_normals()
assert len(scan.vertices) > 0 and len(cloud.points) > 0, "文件没读到,检查文件名"
print(f"口扫: {len(scan.vertices)}顶点 | 点云: {len(cloud.points)}点")

data = np.load(M_PATH)
M = data["M"]
print(f"读取配准矩阵 M, 精配 fitness: {data['fit_fine']:.3f}")

scan_aligned = o3d.geometry.TriangleMesh(scan)
scan_aligned.transform(M)

# ================== ③ 分割车针 ==================
dist, _ = cKDTree(np.asarray(scan_aligned.vertices)).query(np.asarray(cloud.points))
far_pcd = cloud.select_by_index(np.where(dist > DIST_TEETH)[0])
print(f"距离过滤: {len(far_pcd.points)} 个非牙齿点")

picked_idx = pick_idx(far_pcd, "Shift+左键框选车针上的点(整段钻尖),选完按Q")
assert len(picked_idx) >= 10, f"只选了{len(picked_idx)}个点,至少选10个,重跑"
picked_pts = np.asarray(far_pcd.points)[picked_idx]

_, Vpk = np.linalg.eigh(np.cov(picked_pts.T))
span = np.ptp(picked_pts @ Vpk[:, -1])
print(f"点选覆盖长度: {span:.1f} mm(钻尖约10mm,建议>5mm)")
assert span > 5.0, "选点太集中:请旋转到车针侧面,沿长度方向框选整段钻尖"

d_pick, _ = cKDTree(picked_pts).query(np.asarray(far_pcd.points))
drill_pts = np.asarray(far_pcd.points)[d_pick < 15.0]
print(f"点选局部: {len(drill_pts)} 点参与拟合")

# ================== ④ SR10车针拟合 ==================
BUR_R = R_DRILL
BUR_HEAD_LEN = 8.0
BUR_SHANK_R = 0.8

# 分离工作段和连接杆
_, eigvec = np.linalg.eigh(np.cov(picked_pts.T))
drill_axis_init = eigvec[:, -1]
c0 = picked_pts.mean(axis=0)
rel = drill_pts - c0
t_temp = rel @ drill_axis_init
rho_temp = np.linalg.norm(rel - np.outer(t_temp, drill_axis_init), axis=1)
is_working = np.abs(rho_temp - BUR_R) < 0.25
working_pts = drill_pts[is_working]
shank_pts = drill_pts[~is_working]
print(f"分离: 工作段 {is_working.sum()} 点(ρ≈{BUR_R}mm, 用于拟合), 连接杆 {(~is_working).sum()} 点(仅显示)")
assert is_working.sum() >= 50, "工作段点数不足, 框选位置可能不对"
drill_pts = working_pts

# ---- 4.1 轴向初值:PCA ----
_, eigvec = np.linalg.eigh(np.cov(picked_pts.T))
drill_axis = eigvec[:, -1]

tmp = np.array([1.0, 0, 0]) if abs(drill_axis[0]) < 0.9 else np.array([0.0, 1.0, 0])
e1 = np.cross(drill_axis, tmp); e1 /= np.linalg.norm(e1)
e2 = np.cross(drill_axis, e1)
uv = np.c_[drill_pts @ e1, drill_pts @ e2]
uv_picks = np.c_[picked_pts @ e1, picked_pts @ e2]

band = 0.15

def search_center(uv, seed, rng, step):
    g = np.arange(-rng, rng + 1e-9, step)
    GX, GY = np.meshgrid(g, g)
    grid = np.c_[GX.ravel() + seed[0], GY.ravel() + seed[1]]
    best_n, best_c = -1, seed
    for i in range(0, len(grid), 200):
        G = grid[i : i + 200]
        rho = np.linalg.norm(uv[:, None, :] - G[None], axis=2)
        n = (np.abs(rho - BUR_R) < band).sum(0)
        j = int(np.argmax(n))
        if n[j] > best_n:
            best_n, best_c = int(n[j]), G[j]
    return best_c, best_n

print("圆心搜索中(几秒钟)...")
ctr, _ = search_center(uv, uv_picks.mean(0), rng=1.5, step=0.15)
ctr, n_fine = search_center(uv, ctr, rng=0.2, step=0.03)
print(f"圆心搜索: 最佳圆心圈住 {n_fine} 个圆柱面点")
assert n_fine >= 100, "找不到半径0.5mm的圆柱面,点选位置可能不在钻尖上,重跑"

# 精修圆心
for it in range(5):
    rho = np.linalg.norm(uv - ctr, axis=1)
    keep = np.abs(rho - BUR_R) < band
    P = uv[keep]
    r = np.linalg.norm(P - ctr, axis=1)
    res = r - BUR_R
    J = -(P - ctr) / r[:, None]
    delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
    ctr = ctr + delta
    print(f"  精修第{it + 1}轮: 保留{keep.sum()}/{len(keep)}点, 圆心移动{np.linalg.norm(delta):.4f}mm")
    if np.linalg.norm(delta) < 1e-4:
        break

# ---- 4.2 轴心+轴向联合精修 ----
origin = ctr[0] * e1 + ctr[1] * e2 + (drill_pts[keep].mean(0) @ drill_axis) * drill_axis
axis = drill_axis.copy()

for it in range(8):
    tmp = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    rel = drill_pts - origin
    h = rel @ axis
    rho = np.linalg.norm(rel - np.outer(h, axis), axis=1)
    keep = np.abs(rho - BUR_R) < band
    P = drill_pts[keep]
    res = rho[keep] - BUR_R

    def residual_fn(org, ax):
        r_ = P - org
        h_ = r_ @ ax
        return np.linalg.norm(r_ - np.outer(h_, ax), axis=1) - BUR_R

    eps = 1e-6
    J = np.zeros((len(P), 4))
    J[:, 0] = (residual_fn(origin + eps * e1, axis) - res) / eps
    J[:, 1] = (residual_fn(origin + eps * e2, axis) - res) / eps
    a1 = axis + eps * e1; a1 /= np.linalg.norm(a1)
    a2 = axis + eps * e2; a2 /= np.linalg.norm(a2)
    J[:, 2] = (residual_fn(origin, a1) - res) / eps
    J[:, 3] = (residual_fn(origin, a2) - res) / eps

    delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
    origin = origin + delta[0] * e1 + delta[1] * e2
    axis = axis + delta[2] * e1 + delta[3] * e2
    axis /= np.linalg.norm(axis)
    print(f"  联合精修第{it + 1}轮: 保留{keep.sum()}/{len(keep)}点, 修正量{np.linalg.norm(delta):.5f}")
    if np.linalg.norm(delta) < 1e-5:
        break

drill_axis = axis
origin_pt = origin

# ---- 4.3 最终 (t, rho) 坐标 ----
rel = drill_pts - origin_pt
t = rel @ drill_axis
rho = np.linalg.norm(rel - np.outer(t, drill_axis), axis=1)
wall = np.abs(rho - BUR_R) < band
print(f"壁面点径向距离: median={np.median(rho[wall]):.3f}mm(应≈{BUR_R})")

# ---- 4.4 尖端定位 ----
core = rho <= BUR_R + band
tc = t[core]
bins = np.arange(tc.min(), tc.max() + 0.2, 0.1)
cnt, edges = np.histogram(tc, bins=bins)
nz = np.where(cnt >= 3)[0]
assert len(nz) > 0, "轴附近点太少,检查框选位置"
t_front = edges[nz[-1] + 1]
face = (rho < BUR_R - band) & (t > t_front - 0.3)
if face.sum() >= 5:
    t_front = float(np.median(t[face]))
    print(f"端面精修: {face.sum()}个端面点参与定位")
else:
    t_front = t_front - 0.05
    print("端面点不足,按桶中心定位")
t_back = edges[nz[0]]

# 朝向判断:离牙面近的一端是尖端
tree_scan = cKDTree(np.asarray(scan_aligned.vertices))
d_hi = tree_scan.query(origin_pt + t_front * drill_axis)[0]
d_lo = tree_scan.query(origin_pt + t_back * drill_axis)[0]
print(f"端面距牙面: 高端 {d_hi:.2f}mm | 低端 {d_lo:.2f}mm(近的一端定为尖端)")
if d_lo < d_hi:
    drill_axis = -drill_axis
    t = -t
    t_front, t_back = -t_back, -t_front
    print("尖端在选点低端,轴向已翻转")

t_tip = t_front
tip_c = origin_pt + t_tip * drill_axis
L_data = t_front - t_back
print(f"尖端定位: 端面桶含 {cnt[nz[-1]]} 点,覆盖长度 {L_data:.2f}mm")

# ---- 4.5 结果变换回口扫坐标系 ----
Minv = np.linalg.inv(M)
tip_s = Minv[:3, :3] @ tip_c + Minv[:3, 3]
axis_s = Minv[:3, :3] @ drill_axis
axis_s /= np.linalg.norm(axis_s)

print("\n========== 最终结果(口扫坐标系)==========")
print(f"车针尖端: ({tip_s[0]:.3f}, {tip_s[1]:.3f}, {tip_s[2]:.3f}) mm")
print(f"车针朝向: ({axis_s[0]:.4f}, {axis_s[1]:.4f}, {axis_s[2]:.4f})(指向尖端)")
print(f"半径 {BUR_R} mm(固定), 数据覆盖长度 {L_data:.2f} mm(工作部参考 {BUR_HEAD_LEN} mm)")

d_surface = cKDTree(np.asarray(scan.vertices)).query(tip_s)[0]
print(f"自检: 车针尖端距牙面 {d_surface:.2f} mm(钻后数据应贴近牙面)")

# ---- 4.6 可视化 ----
if VISUALIZE:
    sd = np.maximum(
        rho - BUR_R,
        np.maximum(t - t_tip, (t_tip - BUR_HEAD_LEN) - t),
    )
    sc = np.clip(sd / 0.1, -1, 1)
    colors = np.zeros((len(sd), 3))
    pos = sc > 0
    colors[pos] = np.c_[sc[pos], 1 - sc[pos], np.zeros(pos.sum())]
    colors[~pos] = np.c_[np.zeros((~pos).sum()), 1 + sc[~pos], -sc[~pos]]
    drill_colored = o3d.geometry.PointCloud()
    drill_colored.points = o3d.utility.Vector3dVector(drill_pts)
    drill_colored.colors = o3d.utility.Vector3dVector(colors)

    shank_colored = o3d.geometry.PointCloud()
    if len(shank_pts) > 0:
        shank_colored.points = o3d.utility.Vector3dVector(shank_pts)
        shank_colored.paint_uniform_color([0.5, 0.5, 0.5])
        print(f"连接杆显示: {len(shank_pts)} 点(灰色)")

    def _z_to(axis):
        z = np.array([0.0, 0, 1])
        cosang = float(np.clip(z @ axis, -1, 1))
        v = np.cross(z, axis)
        if np.linalg.norm(v) < 1e-8:
            rv = np.array([np.pi, 0, 0]) if cosang < 0 else np.zeros(3)
        else:
            rv = v / np.linalg.norm(v) * np.arccos(cosang)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(rv)

    bur = o3d.geometry.TriangleMesh.create_cylinder(
        radius=BUR_R, height=BUR_HEAD_LEN, resolution=50
    )
    bur.translate((0, 0, t_tip - BUR_HEAD_LEN / 2))
    bur.rotate(_z_to(drill_axis), center=(0, 0, 0))
    bur.translate(origin_pt)
    bur.compute_vertex_normals()
    bur.paint_uniform_color([1.0, 0.0, 1.0])

    print("提示:可视化窗口中按 C 键 显示/隐藏 品红色车针模型")
    print("      绿=面上 红=偏外 蓝=偏内(工作段), 灰=连接杆(仅参考)")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="验收:按C切换车针模型显隐")
    vis.add_geometry(scan_aligned)
    vis.add_geometry(drill_colored)
    if len(shank_pts) > 0:
        vis.add_geometry(shank_colored)
    vis.add_geometry(bur)
    state = {"on": True}

    def toggle(v):
        if state["on"]:
            v.remove_geometry(bur, reset_bounding_box=False)
        else:
            v.add_geometry(bur, reset_bounding_box=False)
        state["on"] = not state["on"]
        return False

    vis.register_key_callback(ord("C"), toggle)
    vis.run()
    vis.destroy_window()
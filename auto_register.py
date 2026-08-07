"""
车针位姿自动求解 v5：
四点对应(Kabsch)粗配 → 裁剪去杂 → 两轮ICP精配 → 点选车针 → 固定半径拟合
特点：全程确定性（无RANSAC随机）、无颜色过滤、不假设钻头位置，任何点云通用
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# ================== 参数 ==================
SCAN_PATH  = "original_scan.ply"   # ← 改成真实文件名
CLOUD_PATH = "point_cloud_20260708_181233_718.ply"     # ← 改成真实文件名

VOXEL       = 1.0    # ICP下采样体素(mm)
JUDGE_DIST  = 2.0    # 评分距离(mm)：此距离内即算对上
CROP_DIST   = 20.0   # 距粗配口扫此距离(mm)以外的点裁掉
DIST_TEETH  = 1.5    # 距口扫超过此值(mm)的点 = 非牙齿
CLUSTER_EPS = 2.0    # 聚类间距(mm)
R_DRILL     = 0.5    # 车针半径(mm)，已知规格，拟合时锁死
N_PICK      = 4      # 粗配对应点个数（学长的四点法）
VISUALIZE   = True

# ================== ① 加载 ==================
scan  = o3d.io.read_triangle_mesh(SCAN_PATH)
cloud = o3d.io.read_point_cloud(CLOUD_PATH)
scan.compute_vertex_normals()
assert len(scan.vertices) > 0 and len(cloud.points) > 0, "文件没读到，检查文件名"
print(f"口扫: {len(scan.vertices)}顶点 | 点云: {len(cloud.points)}点")

scan_pcd = o3d.geometry.PointCloud()
scan_pcd.points = scan.vertices

# ================== ② 配准 ==================
# ---------- 工具函数 ----------
def make_down(pcd):
    p = pcd.voxel_down_sample(VOXEL)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*2, max_nn=30))
    return p

def icp(src, tgt, thr, M0, method="pp", iters=100):
    est = (o3d.pipelines.registration.TransformationEstimationPointToPoint()
           if method == "pp" else
           o3d.pipelines.registration.TransformationEstimationPointToPlane())
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
    return o3d.pipelines.registration.registration_icp(src, tgt, thr, M0, est, crit)

def eval_fit(src, tgt, Mx):
    return o3d.pipelines.registration.evaluate_registration(src, tgt, JUDGE_DIST, Mx).fitness

def refine(src, tgt, M0, stages=((3.0,"pp"),(2.0,"pp"),(1.0,"pp"),(0.8,"plane"),(0.5,"plane"))):
    """逐级ICP精修，每级对比评分，变差就回滚（防滑动）"""
    cur, cur_f = M0, eval_fit(src, tgt, M0)
    for thr, m in stages:
        r = icp(src, tgt, thr, cur, m)
        f = eval_fit(src, tgt, r.transformation)
        if f >= cur_f:
            cur, cur_f = r.transformation, f
    return cur, cur_f

def pick_idx(pcd, msg):
    """Shift+左键框选，按Q结束，返回选中点的下标"""
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=msg)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return [p.index for p in vis.get_picked_points()]

def pick_landmarks(pcd, side):
    """逐个选N_PICK个特征点；每次框一小簇取中心，比单点更稳"""
    pts = []
    for i in range(N_PICK):
        idx = pick_idx(pcd, f"{side} 第{i+1}/{N_PICK}个特征点：框选后按Q")
        assert len(idx) > 0, "没选到点，重跑脚本"
        pts.append(np.asarray(pcd.points)[idx].mean(0))
    return np.array(pts)

def kabsch(P, Q):
    """由N对对应点求刚体变换 M: P→Q（SVD最小二乘，防镜像）"""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:           # 行列式<0是镜像，强制扳回旋转
        Vt[-1] *= -1
        R = Vt.T @ U.T
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = cq - R @ cp
    return M

def show_candidate(Mc, tgt, note):
    scan_try = o3d.geometry.TriangleMesh(scan)
    scan_try.transform(Mc)
    scan_try.paint_uniform_color([0.2, 0.8, 0.2])
    o3d.visualization.draw_geometries([scan_try, tgt], window_name=note)

# ---------- 主流程：点对应粗配 → 裁剪去杂 → 精配 ----------
s_down = make_down(scan_pcd)
c_all  = make_down(cloud)

while True:
    print(f"\n先在口扫上选{N_PICK}个特征点（建议：左磨牙尖→右磨牙尖→门牙中缝→尖牙尖）")
    P = pick_landmarks(scan_pcd, "【口扫】")
    print(f"再在点云上按【相同顺序】选对应的{N_PICK}个点")
    Q = pick_landmarks(cloud.voxel_down_sample(1.0), "【点云】")
    M_rough = kabsch(P, Q)
    M_rough, f_rough = refine(s_down, c_all, M_rough, stages=((8.0,"pp"),(5.0,"pp"),(3.0,"pp")))
    print(f"粗配评分: {f_rough:.3f}")
    show_candidate(M_rough, cloud, f"粗配 fitness={f_rough:.3f} — 大致对上牙齿区域即可")
    if input("粗配对吗？y继续精配 / 回车重新选点: ").strip().lower() != "y":
        continue

    # 裁剪去杂（通用，不依赖颜色）
    scan_rough = o3d.geometry.TriangleMesh(scan)
    scan_rough.transform(M_rough)
    d, _ = cKDTree(np.asarray(scan_rough.vertices)).query(np.asarray(cloud.points))
    cloud_fine = cloud.select_by_index(np.where(d < CROP_DIST)[0])
    print(f"裁剪去杂: {len(cloud_fine.points)}/{len(cloud.points)} 点进入精配")

    # 精配
    c2 = make_down(cloud_fine)
    print(f"[体检] 粗配结果在精配点云上的评分: {eval_fit(s_down, c2, M_rough):.3f}")
    M, f_fine = refine(s_down, c2, M_rough)
    print(f"精配评分: {f_fine:.3f}")
    show_candidate(M, cloud_fine, f"精配 fitness={f_fine:.3f} — 应牙尖对牙尖")
    if input("精配对吗？y采用 / 回车重新选点: ").strip().lower() == "y":
        break

# ---------- 验收 ----------
scan_aligned = o3d.geometry.TriangleMesh(scan)
scan_aligned.transform(M)
if VISUALIZE:
    verify = o3d.geometry.TriangleMesh(scan_aligned)
    verify.paint_uniform_color([1, 1, 1])   # 白色口扫叠彩色点云：交替贴合=配准成功
    o3d.visualization.draw_geometries([verify, cloud_fine],
                                      window_name="验收：表面应白彩交替贴合")

# ================== ③ 分割车针 ==================
dist, _ = cKDTree(np.asarray(scan_aligned.vertices)).query(np.asarray(cloud.points))
far_pcd = cloud.select_by_index(np.where(dist > DIST_TEETH)[0])
print(f"距离过滤: {len(far_pcd.points)} 个非牙齿点")

picked_idx = pick_idx(far_pcd, "Shift+左键框选车针上的点，选完按Q")
assert len(picked_idx) >= 10, f"只选了{len(picked_idx)}个点，至少选10个，重跑"
picked_pts = np.asarray(far_pcd.points)[picked_idx]

# 只在点选位置附近找车针：后面连着的长杆进不了这个范围
d_pick, _ = cKDTree(picked_pts).query(np.asarray(far_pcd.points))
drill_pts = np.asarray(far_pcd.points)[d_pick < 15.0]
print(f"点选局部: {len(drill_pts)} 点参与拟合（钻尖约10mm长，15mm足够覆盖）")

# ================== ④ 固定半径拟合圆柱（网格搜圆心版）==================
# 轴向初值：选点沿车针方向拉长即可（只选到一面不影响方向）
_, eigvec = np.linalg.eigh(np.cov(picked_pts.T))
drill_axis = eigvec[:, -1]

# 投影到垂直于轴的平面：圆柱面 → 半径0.5mm的圆
tmp = np.array([1.0, 0, 0]) if abs(drill_axis[0]) < 0.9 else np.array([0.0, 1.0, 0])
e1 = np.cross(drill_axis, tmp); e1 /= np.linalg.norm(e1)
e2 = np.cross(drill_axis, e1)
uv       = np.c_[drill_pts @ e1, drill_pts @ e2]
uv_picks = np.c_[picked_pts @ e1, picked_pts @ e2]

band = 0.15

def search_center(uv, seed, rng, step):
    """在seed附近按网格试圆心：哪个圆心能在0.5mm圆周上圈住最多点"""
    g = np.arange(-rng, rng + 1e-9, step)
    GX, GY = np.meshgrid(g, g)
    grid = np.c_[GX.ravel() + seed[0], GY.ravel() + seed[1]]
    best_n, best_c = -1, seed
    for i in range(0, len(grid), 200):
        G = grid[i:i + 200]
        rho = np.linalg.norm(uv[:, None, :] - G[None], axis=2)
        n = (np.abs(rho - R_DRILL) < band).sum(0)
        j = int(np.argmax(n))
        if n[j] > best_n:
            best_n, best_c = int(n[j]), G[j]
    return best_c, best_n

print("圆心搜索中（几秒钟）...")
ctr, _ = search_center(uv, uv_picks.mean(0), rng=1.5, step=0.15)  # 粗搜
ctr, n_fine = search_center(uv, ctr, rng=0.2, step=0.03)          # 细搜
print(f"圆心搜索: 最佳圆心圈住 {n_fine} 个圆柱面点")
assert n_fine >= 100, "找不到半径0.5mm的圆柱面，点选位置可能不在钻尖上，重跑"

# 精修圆心：圆柱面上的点沿径向退0.5mm就是轴心，投票平均
for it in range(3):
    rho = np.linalg.norm(uv - ctr, axis=1)
    keep = np.abs(rho - R_DRILL) < band
    rad = uv[keep] - ctr
    ctr = (uv[keep] - R_DRILL * rad / rho[keep, None]).mean(0)
    print(f"  精修第{it+1}轮: 保留{keep.sum()}/{len(keep)}点")

drill_pts = drill_pts[keep]

# 剩下的都是干净圆柱面点，最后精修一次轴向（点积保方向一致）
_, eigvec = np.linalg.eigh(np.cov(drill_pts.T))
new_axis = eigvec[:, -1]
if new_axis @ drill_axis < 0:
    new_axis = -new_axis
drill_axis = new_axis

# 在最终保留点上计算中心和长度
c = drill_pts.mean(0)
rel = drill_pts - c
t = rel @ drill_axis
center_c = c + (t.max() + t.min()) / 2 * drill_axis
length = t.max() - t.min()
if length > 15.0:
    print("警告: 长度偏大，可能混入了钻尖以外的结构，看标红窗口确认")

Minv = np.linalg.inv(M)
center_s = Minv[:3, :3] @ center_c + Minv[:3, 3]
axis_s = Minv[:3, :3] @ drill_axis
axis_s /= np.linalg.norm(axis_s)

print("\n========== 最终结果（口扫坐标系）==========")
print(f"车针中心: ({center_s[0]:.3f}, {center_s[1]:.3f}, {center_s[2]:.3f}) mm")
print(f"车针朝向: ({axis_s[0]:.4f}, {axis_s[1]:.4f}, {axis_s[2]:.4f})")
print(f"半径 {R_DRILL} mm(固定), 长度 {length:.2f} mm")

# 几何合理性自检：钻头悬空在牙列附近（<15mm合理）
d_surface = cKDTree(np.asarray(scan.vertices)).query(center_s)[0]
print(f"自检: 车针中心距牙面 {d_surface:.2f} mm（悬空钻头应<15mm）")

if VISUALIZE:
    drill_show = o3d.geometry.PointCloud()
    drill_show.points = o3d.utility.Vector3dVector(drill_pts)
    drill_show.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([scan_aligned, drill_show],
                                      window_name="红色 = 最终车针点")
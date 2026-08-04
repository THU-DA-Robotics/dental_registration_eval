"""
车针位姿自动求解 v4：
颜色过滤 → PCA旋转×锚点投票+RANSAC竞选 → 防滑动精修 → 点选车针 → 固定半径拟合
"""
import itertools
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# ================== 参数 ==================
SCAN_PATH  = "original_scan.ply"   # ← 改成真实文件名
CLOUD_PATH = "point_cloud.ply"     # ← 改成真实文件名

VOXEL       = 1.0    # 配准下采样体素(mm)
DIST_TEETH  = 1.5    # 距口扫超过此值(mm)的点 = 非牙齿
CLUSTER_EPS = 2.0    # 聚类间距(mm)
R_DRILL     = 0.5    # 车针半径(mm)，已知规格，拟合时锁死
VISUALIZE   = True

# ================== ① 加载 ==================
scan  = o3d.io.read_triangle_mesh(SCAN_PATH)
cloud = o3d.io.read_point_cloud(CLOUD_PATH)
scan.compute_vertex_normals()
assert len(scan.vertices) > 0 and len(cloud.points) > 0, "文件没读到，检查文件名"
print(f"口扫: {len(scan.vertices)}顶点 | 点云: {len(cloud.points)}点")

# ================== ② 全自动配准 ==================
# 2a. 颜色过滤：暖色点(牙齿+车针)参与配准，冷色点(杂物块)剔除
colors = np.asarray(cloud.colors)
if colors.size > 0:
    warm = np.where(colors[:, 0] > colors[:, 2] + 0.1)[0]
    cloud_reg = cloud.select_by_index(warm)
    print(f"颜色过滤: {len(cloud_reg.points)}/{len(cloud.points)} 点参与配准")
else:
    cloud_reg = cloud
    print("点云无颜色信息，未过滤")

scan_pcd = o3d.geometry.PointCloud()
scan_pcd.points = scan.vertices
s_down = scan_pcd.voxel_down_sample(VOXEL)
c_down = cloud_reg.voxel_down_sample(VOXEL)
s_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*2, max_nn=30))
c_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*2, max_nn=30))
s_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    s_down, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*5, max_nn=100))
c_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    c_down, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*5, max_nn=100))

def pca_frame(pts):
    c = pts.mean(0)
    _, V = np.linalg.eigh(np.cov(pts.T))
    return c, V

cs, Vs = pca_frame(np.asarray(s_down.points))
ct, Vt = pca_frame(np.asarray(c_down.points))

# 2b. 旋转候选24个
rotations = []
for perm in itertools.permutations(range(3)):
    Vp = Vs[:, list(perm)]
    for s0 in (1, -1):
        for s1 in (1, -1):
            for s2 in (1, -1):
                R = Vt @ np.diag([s0, s1, s2]) @ Vp.T
                if np.linalg.det(R) > 0:
                    rotations.append(R)

# 2c. 平移锚点
cloud_coarse = cloud_reg.voxel_down_sample(5.0)
lab = np.array(cloud_coarse.cluster_dbscan(eps=15.0, min_points=20))
anchors = [np.asarray(cloud_coarse.points)[lab == k].mean(0)
           for k in range(lab.max() + 1) if np.sum(lab == k) > 30]
anchors.append(ct)
print(f"{len(rotations)}个旋转 × {len(anchors)}个平移锚点")

# 2d. 快速投票
tgt_tree = cKDTree(np.asarray(c_down.points))
s_pts = np.asarray(s_down.points)
rng = np.random.default_rng(0)
sample = s_pts[rng.choice(len(s_pts), size=min(3000, len(s_pts)), replace=False)]
votes = []
for R in rotations:
    rot_sample = sample @ R.T
    base = -R @ cs
    for a in anchors:
        d, _ = tgt_tree.query(rot_sample + (a + base))
        votes.append((float(np.mean(d < 5.0)), R, a + base))
votes.sort(key=lambda v: -v[0])

candidate_Ms = []
for score, R, t in votes[:5]:
    Mc = np.eye(4)
    Mc[:3, :3] = R
    Mc[:3, 3] = t
    candidate_Ms.append(("投票", Mc))

# 2e. RANSAC 20次
for i in range(20):
    r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s_down, c_down, s_fpfh, c_fpfh, True, 2.0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(2.0)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(300000, 0.999))
    print(f"RANSAC第{i+1}次: fitness={r.fitness:.3f}")
    candidate_Ms.append((f"RANSAC{i+1}", r.transformation))

# 2f. 统一决赛：ICP后用1mm标准打分，全部候选排名
def eval1mm(Mx):
    return o3d.pipelines.registration.evaluate_registration(s_down, c_down, 1.0, Mx)

finalists = []
for src, Mc in candidate_Ms:
    r = o3d.pipelines.registration.registration_icp(
        s_down, c_down, 5.0, Mc,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    ev = eval1mm(r.transformation)
    finalists.append(((ev.fitness, -ev.inlier_rmse), src, r.transformation))
finalists.sort(key=lambda item: item[0], reverse=True)
print("决赛前五名:", [(s, round(k[0], 3)) for k, s, _ in finalists[:5]])

# 2g. 逐级精修（每级考完试，变差回滚）
def refine(M0):
    cur, cur_fit = M0, eval1mm(M0).fitness
    for thr, method in [(2.5, "pp"), (1.2, "pp"), (1.5, "plane")]:
        est = (o3d.pipelines.registration.TransformationEstimationPointToPoint()
               if method == "pp" else
               o3d.pipelines.registration.TransformationEstimationPointToPlane())
        r = o3d.pipelines.registration.registration_icp(s_down, c_down, thr, cur, est)
        f = eval1mm(r.transformation).fitness
        if f >= cur_fit:
            cur, cur_fit = r.transformation, f
    return cur, cur_fit

# 2h. 人工确认闸门：从冠军开始逐个弹窗，y接受 / 回车看下一个
M = None
for key, src, M0 in finalists[:8]:
    Mcand, f = refine(M0)
    scan_try = o3d.geometry.TriangleMesh(scan)
    scan_try.transform(Mcand)
    scan_try.paint_uniform_color([0.2, 0.8, 0.2])
    o3d.visualization.draw_geometries(
        [scan_try, cloud],
        window_name=f"候选[{src}] fitness={f:.3f} — 关闭窗口后到终端回答")
    ans = input(f"候选[{src}] (1mm fitness={f:.3f}) 配准正确吗？"
                f"正确输入 y 回车，看下一个直接回车: ")
    if ans.strip().lower() == "y":
        M = Mcand
        print(f"已确认采用[{src}]的配准结果")
        break
if M is None:
    raise RuntimeError("前8个候选都被拒绝了，请重跑脚本（RANSAC有随机性，多跑几次）")

# 生成配准后的口扫
scan_aligned = o3d.geometry.TriangleMesh(scan)
scan_aligned.transform(M)

# ================== ③ 距离筛选 + 手动点选车针 ==================
dist, _ = cKDTree(np.asarray(scan_aligned.vertices)).query(np.asarray(cloud.points))
far_pcd = cloud.select_by_index(np.where(dist > DIST_TEETH)[0])
print(f"非牙齿点: {len(far_pcd.points)} 个（车针+杂物）")

print("\n>>> 弹出窗口后：按住 Shift + 左键拖动，密集框选整根车针上的点")
print(">>> 选错了按 Shift + 右键撤销；选完按 Q 关闭窗口\n")
vis = o3d.visualization.VisualizerWithVertexSelection()
vis.create_window(window_name="Shift+左键框选车针上的点，按Q结束")
vis.add_geometry(far_pcd)
vis.run()
vis.destroy_window()
picked = vis.get_picked_points()
try:
    sel_idx = np.asarray(picked, dtype=int).ravel()
except Exception:
    sel_idx = np.array([p.index for p in picked], dtype=int)
assert len(sel_idx) >= 10, "选的点太少了，重跑并密集框选整根车针"

labels = np.array(far_pcd.cluster_dbscan(eps=CLUSTER_EPS, min_points=30))
picked_labels = labels[sel_idx]
picked_labels = picked_labels[picked_labels >= 0]
assert len(picked_labels) >= 5, (
    "你选的点几乎都不在任何聚类里（全是离群噪声）。\n"
    "重跑后：涂得更密、更大面积覆盖车针；还不行就把 CLUSTER_EPS 调大到 3"
)
vals, counts = np.unique(picked_labels, return_counts=True)
drill_k = vals[np.argmax(counts)]
drill_pts = np.asarray(far_pcd.points)[labels == drill_k]
print(f"车针团: 团{drill_k}, {len(drill_pts)} 点")
assert len(drill_pts) >= 100, f"车针团只有{len(drill_pts)}点，太少，重跑时多涂一些车针上的点"

# ================== ④ 固定半径拟合圆柱 ==================
band = 0.15
for it in range(5):
    eigval, eigvec = np.linalg.eigh(np.cov(drill_pts.T))
    drill_axis = eigvec[:, -1]
    c = drill_pts.mean(0)
    rel = drill_pts - c
    t = rel @ drill_axis
    r = np.linalg.norm(rel - np.outer(t, drill_axis), axis=1)
    keep = np.abs(r - R_DRILL) < band
    print(f"  精修第{it+1}轮: 保留{keep.sum()}/{len(keep)}点")
    if keep.all() or keep.sum() < 100:
        break
    drill_pts = drill_pts[keep]

center_c = c + (t.max() + t.min()) / 2 * drill_axis
length = t.max() - t.min()

Minv = np.linalg.inv(M)
center_s = Minv[:3, :3] @ center_c + Minv[:3, 3]
axis_s = Minv[:3, :3] @ drill_axis
axis_s /= np.linalg.norm(axis_s)

print("\n========== 最终结果（口扫坐标系）==========")
print(f"车针中心: ({center_s[0]:.3f}, {center_s[1]:.3f}, {center_s[2]:.3f}) mm")
print(f"车针朝向: ({axis_s[0]:.4f}, {axis_s[1]:.4f}, {axis_s[2]:.4f})")
print(f"半径 {R_DRILL} mm(固定), 长度 {length:.2f} mm")

# 几何合理性自检：车针中心应紧贴牙面（<5mm合理）
d_surface = cKDTree(np.asarray(scan.vertices)).query(center_s)[0]
print(f"自检: 车针中心距牙面 {d_surface:.2f} mm（<5mm 合理）")

if VISUALIZE:
    drill_show = o3d.geometry.PointCloud()
    drill_show.points = o3d.utility.Vector3dVector(drill_pts)
    drill_show.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([scan_aligned, drill_show],
                                      window_name="红色 = 最终车针点")
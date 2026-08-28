"""
车针位姿自动求解 v8.1：
四点对应(Kabsch)粗配 → 自研裁剪ICP超高精度精配（法向兼容过滤 + 分位数裁剪 + Huber点面）
→ 框选车针 → 网格搜索固定半径拟合 → 圆柱可视化
特点：纯刚性不变尺寸、全程确定性、只打一回合点（分窗口各自只显示自己）、
      错位区域(龈缘/邻面错配)不参与求解，最终贴合到 0.1mm 级

交互说明：
  口扫窗口：直接显示三角网格，按顺序右键点 4 个牙尖后关窗
  点云窗口：只显示点云，按相同顺序右键点 4 个对应牙尖后关窗
  之后粗配、精配、车针拟合全自动完成，中途若 fitness 达标则跳过确认
  车针：open3d窗口里Shift+左键框选整段钻尖(Ctrl反选)，按Q结束
"""

import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.spatial import cKDTree

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ================== 参数 ==================
SCAN_PATH = "original_scan.ply"  # ← 改成真实文件名
CLOUD_PATH = "point_cloud.ply"  # ← 改成真实文件名

VOXEL = 1.0  # ICP下采样体素(mm)
JUDGE_DIST = 2.0  # 评分距离(mm)：此距离内即算对上
CROP_DIST = 15.0  # 距粗配口扫此距离(mm)以外的点裁掉
DIST_TEETH = 1.5  # 距口扫超过此值(mm)的点 = 非牙齿
CLUSTER_EPS = 2.0  # 聚类间距(mm)
R_DRILL = 0.5  # 车针半径(mm)，已知规格，拟合时锁死
N_PICK_ROUGH = 4  # 首轮粗配准对应点对数
MIN_PICK_DIST = 0.5  # 连续选点最小间距(mm)，防止吸附到同一位置
AUTO_ACCEPT_FITNESS = 0.75  # 自动接受的 fitness 阈值
FINE_VOXEL = 0.3  # 超高精度精配的体素(mm)
DENSE_N = 200000  # 最终贴合阶段泊松盘采样点数
VISUALIZE = True

# ================== ① 加载 ==================
scan = o3d.io.read_triangle_mesh(SCAN_PATH)
cloud = o3d.io.read_point_cloud(CLOUD_PATH)
scan.compute_vertex_normals()
assert len(scan.vertices) > 0 and len(cloud.points) > 0, "文件没读到，检查文件名"
print(f"口扫: {len(scan.vertices)}顶点 | 点云: {len(cloud.points)}点")

scan_pcd = o3d.geometry.PointCloud()
scan_pcd.points = scan.vertices


# ================== ② 配准 ==================
# ---------- 工具函数 ----------
def make_down(pcd, voxel=VOXEL):
    p = pcd.voxel_down_sample(voxel)
    p.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 2, 1.5), max_nn=50)
    )
    return p


def eval_fit(src, tgt, Mx):
    return o3d.pipelines.registration.evaluate_registration(
        src, tgt, JUDGE_DIST, Mx
    ).fitness


def overlap_src(src, tgt_pts, Mx, margin):
    """只保留当前位姿下处于重叠区（距点云margin内）的源点——超出部分不参与配准"""
    s = o3d.geometry.PointCloud(src)
    s.transform(Mx)
    d, _ = cKDTree(tgt_pts).query(np.asarray(s.points))
    return src.select_by_index(np.where(d < margin)[0])


def kabsch(P, Q):
    """由N对对应点求刚体变换 M: P→Q（SVD最小二乘，防镜像）"""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = cq - R @ cp
    return M


# ---------- 自研裁剪ICP（抗龈缘/邻面错配） ----------
def p2plane_step(Pt, Q, N, huber_k):
    """线性化点-面最小二乘一步：求解小旋转w和平移t，Huber加权抗离群"""
    d = ((Pt - Q) * N).sum(1)  # 点面残差
    A = np.hstack([np.cross(Pt, N), N])  # (w·(p×n) + t·n) ≈ -d
    w = np.where(np.abs(d) <= huber_k, 1.0, huber_k / np.maximum(np.abs(d), 1e-12))
    sw = np.sqrt(w)
    x, *_ = np.linalg.lstsq(A * sw[:, None], -d * sw, rcond=None)
    M = np.eye(4)
    M[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(x[:3])
    M[:3, 3] = x[3:]
    return M


def icp_custom(src, tgt, M0, thr, iters=30, keep_ratio=0.9, mode="plane"):
    """裁剪ICP主循环：
    1) 法向兼容过滤（thr<=2mm时启用）：对应点法向夹角>60°的剔除，防止抓到邻面/龈缘错面
    2) 分位数裁剪：每轮只用残差最小的 keep_ratio 部分求解，错位区域不拉偏
    3) plane=Huber加权点面最小二乘 / pp=Kabsch
    """
    src_pts = np.asarray(src.points)
    src_nrm = np.asarray(src.normals) if src.has_normals() else None
    tgt_pts = np.asarray(tgt.points)
    tgt_nrm = np.asarray(tgt.normals) if tgt.has_normals() else None
    tree = cKDTree(tgt_pts)
    use_nrm = src_nrm is not None and tgt_nrm is not None and thr <= 2.0
    M = M0.copy()
    prev = None
    for _ in range(iters):
        R = M[:3, :3]
        src_t = src_pts @ R.T + M[:3, 3]
        d, idx = tree.query(src_t)
        mask = d < thr
        if mask.sum() < 1000:
            break
        if use_nrm:
            nrm_t = src_nrm @ R.T
            cosang = np.abs((nrm_t * tgt_nrm[idx]).sum(1))
            if (cosang[mask] > 0.5).sum() >= 1000:
                mask &= cosang > 0.5
        dd = d[mask]
        cut = np.percentile(dd, keep_ratio * 100)
        good = np.zeros(len(d), bool)
        good[np.where(mask)[0][dd <= cut]] = True
        P = src_t[good]
        Q = tgt_pts[idx[good]]
        if mode == "plane" and tgt_nrm is not None:
            delta = p2plane_step(P, Q, tgt_nrm[idx[good]], huber_k=max(0.3 * thr, 0.05))
        else:
            delta = kabsch(P, Q)
        M = delta @ M
        m = dd[dd <= cut].mean()
        if prev is not None and prev - m < 1e-4:
            break
        prev = m
    return M


def refine(src, tgt, M0, stages=None):
    """逐级裁剪ICP精修；每级先剔除无对应点，变差就回滚"""
    if stages is None:
        stages = (
            (3.0, "pp", 0.85, 40),
            (2.0, "pp", 0.85, 40),
            (1.0, "plane", 0.9, 30),
            (0.8, "plane", 0.9, 30),
            (0.5, "plane", 0.9, 30),
        )
    tgt_pts = np.asarray(tgt.points)
    cur = M0
    for thr, mode, keep, iters in stages:
        src_ov = overlap_src(src, tgt_pts, cur, margin=max(2.5 * thr, 0.4))
        if len(src_ov.points) < 500:
            continue
        f0 = eval_fit(src_ov, tgt, cur)
        M1 = icp_custom(src_ov, tgt, cur, thr, iters=iters, keep_ratio=keep, mode=mode)
        f1 = eval_fit(src_ov, tgt, M1)
        if f1 >= f0:
            cur = M1
    return cur, eval_fit(src, tgt, cur)


def ultra_refine(src, tgt, M0):
    """超高精度裁剪ICP：细到 0.1mm 阈值，多级 point-to-plane"""
    stages = (
        (1.0, "pp", 0.9, 40),
        (0.8, "plane", 0.9, 40),
        (0.6, "plane", 0.9, 40),
        (0.4, "plane", 0.92, 40),
        (0.3, "plane", 0.92, 40),
        (0.2, "plane", 0.95, 40),
        (0.15, "plane", 0.95, 40),
        (0.1, "plane", 0.97, 40),
    )
    return refine(src, tgt, M0, stages)


def fit_report(src, tgt, Mx, label):
    """贴合度体检：重叠区最近邻距离的统计"""
    s = o3d.geometry.PointCloud(src)
    s.transform(Mx)
    d, _ = cKDTree(np.asarray(tgt.points)).query(np.asarray(s.points))
    d_ov = d[d < 2 * JUDGE_DIST]
    print(
        f"{label}: 重叠比例={len(d_ov) / len(d) * 100:.1f}%, mean={d_ov.mean():.3f}mm, "
        f"median={np.median(d_ov):.3f}mm, 95%={np.percentile(d_ov, 95):.3f}mm"
    )


# ---------- 选点 ----------
def pick_idx(geometry, msg):
    """在Open3D几何体上选顶点，按Q结束并返回顶点下标。"""
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=msg)
    vis.add_geometry(geometry)
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()
    return [p.index for p in picked]


def _pv_add(pl, geometry, point_size=3.0, fallback_color="lightblue"):
    """把Open3D点云或三角网格加入PyVista窗口，并保留顶点颜色。"""
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        vertices = np.asarray(geometry.vertices)
        triangles = np.asarray(geometry.triangles)
        faces = np.column_stack(
            [np.full(len(triangles), 3, dtype=np.int64), triangles]
        ).ravel()
        pdata = pv.PolyData(vertices, faces)
        has_colors = geometry.has_vertex_colors()
    else:
        pdata = pv.PolyData(np.asarray(geometry.points))
        has_colors = geometry.has_colors()

    kw = {}
    if has_colors:
        colors = (
            geometry.vertex_colors
            if isinstance(geometry, o3d.geometry.TriangleMesh)
            else geometry.colors
        )
        pdata["colors"] = (np.asarray(colors) * 255).astype(np.uint8)
        kw = dict(scalars="colors", rgb=True, preference="point")
    else:
        kw = dict(color=fallback_color)

    if isinstance(geometry, o3d.geometry.TriangleMesh):
        pl.add_mesh(pdata, smooth_shading=True, **kw)
    else:
        pl.add_mesh(pdata, point_size=point_size, render_points_as_spheres=True, **kw)


def pick_click(geometry, msg, solid_color=None):
    """PyVista右键选点：窗口里只显示本对象，绝不混层，点哪层一目了然。
    标记球不可拾取 + 最小间距过滤，防重复吸附到上一个点。"""
    pl = pv.Plotter(title=msg)
    if solid_color is not None:
        _pv_add(pl, geometry, fallback_color=solid_color)
    else:
        _pv_add(pl, geometry)
    picked = []

    def cb(point, *args):
        p = np.asarray(point, float)
        if len(picked) > 0:
            d_min = np.linalg.norm(p - np.array(picked), axis=1).min()
            if d_min < MIN_PICK_DIST:
                print(f"  [忽略] 与已选点太近 ({d_min:.2f} mm)，请点别处")
                return
        picked.append(p)
        pl.add_mesh(
            pv.Sphere(radius=0.5, center=p),
            color="red",
            pickable=False,
            name=f"pick_marker_{len(picked)}",
        )
        pl.add_text(f"已选 {len(picked)} 点", name="pick_count", font_size=12)

    if isinstance(geometry, o3d.geometry.TriangleMesh):
        # PointPicker会穿过三角面搜索顶点；CellPicker只返回视线最先命中的表面。
        pl.enable_surface_point_picking(
            callback=cb,
            picker="cell",
            left_clicking=False,
            show_message="右键依次选择可见表面上的点，选完直接关窗",
        )
    else:
        pl.enable_point_picking(
            callback=cb,
            picker="point",
            left_clicking=False,
            show_message="右键逐个点选，选完直接关窗",
        )
    pl.show()
    return np.array(picked) if picked else np.empty((0, 3))


def pick_landmarks(geometry, side, n_points=N_PICK_ROUGH):
    """一个窗口内按顺序右键选择 n_points 个特征点；只显示当前对象。"""
    while True:
        pts = pick_click(
            geometry, f"{side}：按顺序右键选择{n_points}个特征点，选完关窗"
        )
        if len(pts) == n_points:
            return pts
        print(f"选了{len(pts)}个，需要恰好{n_points}个，重选")


def show_candidate(Mc, tgt, note):
    scan_try = o3d.geometry.TriangleMesh(scan)
    scan_try.transform(Mc)
    scan_try.paint_uniform_color([0.2, 0.8, 0.2])
    o3d.visualization.draw_geometries([scan_try, tgt], window_name=note)


def gate(M0, tgt_show, ref_src, ref_tgt, label):
    """确认闸门：y接受 / 回车重选"""
    print(f"{label}评分: {eval_fit(ref_src, ref_tgt, M0):.3f}")
    show_candidate(M0, tgt_show, f"{label} — 关窗后到终端回答")
    ans = input(f"{label}对吗？y接受 / 回车重选: ").strip().lower()
    return M0 if ans == "y" else None


# ---------- 主流程：四点对应粗配 → 裁剪去杂 → 裁剪ICP超高精度精配 ----------
s_down = make_down(scan_pcd)
c_all = make_down(cloud)

while True:
    print()
    print(
        f"先在口扫上按顺序选{N_PICK_ROUGH}个特征点（建议：左磨牙尖→右磨牙尖→门牙中缝→尖牙尖，尽量张开跨度）"
    )
    print("  窗口里直接显示口扫三角网格，请右键选择可见表面上的牙尖")
    P = pick_landmarks(scan, "【口扫】", N_PICK_ROUGH)
    print(f"再在点云上按【相同顺序】选对应的{N_PICK_ROUGH}个点")
    print("  窗口里只有点云，请按顺序右键点同一批牙尖")
    Q = pick_landmarks(cloud, "【点云】", N_PICK_ROUGH)

    M_rough = kabsch(P, Q)
    M_rough, _ = refine(
        s_down,
        c_all,
        M_rough,
        stages=((8.0, "pp", 0.7, 40), (5.0, "pp", 0.75, 40), (3.0, "pp", 0.8, 40)),
    )
    fit_rough = eval_fit(s_down, c_all, M_rough)
    print(f"自动粗配 fitness: {fit_rough:.3f}")

    # 粗配够好就跳过闸门
    if fit_rough < AUTO_ACCEPT_FITNESS:
        M_rough = gate(M_rough, cloud, s_down, c_all, "粗配")
        if M_rough is None:
            continue
    else:
        print("粗配结果达到自动接受阈值，继续精配...")

    # 裁剪去杂（通用，不依赖颜色）
    scan_rough = o3d.geometry.TriangleMesh(scan)
    scan_rough.transform(M_rough)
    d, _ = cKDTree(np.asarray(scan_rough.vertices)).query(np.asarray(cloud.points))
    cloud_fine = cloud.select_by_index(np.where(d < CROP_DIST)[0])
    print(f"裁剪去杂: {len(cloud_fine.points)}/{len(cloud.points)} 点进入精配")

    # 精配：裁剪ICP逐级下压
    c2 = make_down(cloud_fine)
    print(f"[体检] 粗配结果在精配点云上的评分: {eval_fit(s_down, c2, M_rough):.3f}")
    M, _ = refine(s_down, c2, M_rough)
    M, _ = ultra_refine(
        make_down(scan_pcd, FINE_VOXEL), make_down(cloud_fine, FINE_VOXEL), M
    )

    # 最终贴合：泊松盘均匀采样口扫表面 + 0.25mm点云 + Huber点面ICP
    print("最终贴合：口扫表面泊松盘采样20万点 + 0.25mm点云 + Huber点面ICP...")
    s_dense = scan.sample_points_poisson_disk(
        DENSE_N, init_factor=5, use_triangle_normal=True
    )
    c_dense = make_down(cloud_fine, 0.25)
    M, _ = refine(
        s_dense,
        c_dense,
        M,
        stages=(
            (0.4, "plane", 0.95, 40),
            (0.2, "plane", 0.95, 40),
            (0.15, "plane", 0.95, 40),
            (0.1, "plane", 0.97, 40),
        ),
    )

    # 贴合度体检报告
    fit_report(s_dense, c_dense, M, "精配贴合度")

    # 自动接受或闸门
    fit_fine = eval_fit(s_down, c2, M)
    print(f"精配 fitness: {fit_fine:.3f}")
    if fit_fine < AUTO_ACCEPT_FITNESS:
        M = gate(M, cloud_fine, s_down, c2, "精配")
        if M is None:
            continue
    else:
        print("精配结果达到自动接受阈值。")

    s_ov = overlap_src(s_down, np.asarray(c2.points), M, margin=2 * JUDGE_DIST)
    rr = o3d.pipelines.registration.evaluate_registration(s_ov, c2, JUDGE_DIST, M)
    print(f"精配(重叠区): fitness={rr.fitness:.3f}, RMSE={rr.inlier_rmse:.2f} mm")
    break

# ---------- 验收 ----------
scan_aligned = o3d.geometry.TriangleMesh(scan)
scan_aligned.transform(M)
if VISUALIZE:
    verify = o3d.geometry.TriangleMesh(scan_aligned)
    verify.paint_uniform_color([1, 1, 1])  # 白色口扫叠彩色点云：交替贴合=配准成功
    o3d.visualization.draw_geometries(
        [verify, cloud_fine], window_name="验收：表面应白彩交替贴合"
    )
    # 残差热力图：绿=贴合(0) 红=偏差≥0.5mm，哪里不服帖一眼看到
    sd_dist, _ = cKDTree(np.asarray(cloud_fine.points)).query(
        np.asarray(scan_aligned.vertices)
    )
    heat = np.clip(sd_dist / 0.5, 0, 1)
    verify2 = o3d.geometry.TriangleMesh(scan_aligned)
    verify2.vertex_colors = o3d.utility.Vector3dVector(
        np.c_[heat, 1.0 - heat, np.zeros_like(heat)]
    )
    o3d.visualization.draw_geometries(
        [verify2], window_name="残差热力图：绿=贴合 红=偏差≥0.5mm"
    )

# ================== ③ 分割车针 ==================
dist, _ = cKDTree(np.asarray(scan_aligned.vertices)).query(np.asarray(cloud.points))
far_pcd = cloud.select_by_index(np.where(dist > DIST_TEETH)[0])
print(f"距离过滤: {len(far_pcd.points)} 个非牙齿点")

picked_idx = pick_idx(far_pcd, "Shift+左键框选车针上的点（整段钻尖），选完按Q")
assert len(picked_idx) >= 10, f"只选了{len(picked_idx)}个点，至少选10个，重跑"
picked_pts = np.asarray(far_pcd.points)[picked_idx]

_, Vpk = np.linalg.eigh(np.cov(picked_pts.T))
span = np.ptp(picked_pts @ Vpk[:, -1])  # 选点沿最长方向的跨度
print(f"点选覆盖长度: {span:.1f} mm（钻尖约10mm，建议>5mm）")
assert span > 5.0, "选点太集中：请旋转到车针侧面，沿长度方向框选整段钻尖"

# 只在点选位置附近找车针：后面连着的长杆进不了这个范围
d_pick, _ = cKDTree(picked_pts).query(np.asarray(far_pcd.points))
drill_pts = np.asarray(far_pcd.points)[d_pick < 15.0]
print(f"点选局部: {len(drill_pts)} 点参与拟合（钻尖约10mm长，15mm足够覆盖）")

# ================== ④ SR10车针拟合（圆柱杆身 + 半球尖端，胶囊模型）==================
# SR10 几何（Straight Round End，柱形圆头、无锥度）：
#   工作部 = 直径1.0mm圆柱 + 半径0.5mm半球尖端；工作部长约8mm；FG柄部直径1.6mm
BUR_R = R_DRILL      # 工作部半径 0.5mm（已知规格，锁死）
BUR_HEAD_LEN = 8.0   # 工作部长度参考值(mm)，仅用于合理性检查
BUR_SHANK_R = 0.8    # 柄部半径(mm)，仅可视化用

# ---- 4.1 轴向初值：PCA ----
_, eigvec = np.linalg.eigh(np.cov(picked_pts.T))
drill_axis = eigvec[:, -1]

tmp = np.array([1.0, 0, 0]) if abs(drill_axis[0]) < 0.9 else np.array([0.0, 1.0, 0])
e1 = np.cross(drill_axis, tmp); e1 /= np.linalg.norm(e1)
e2 = np.cross(drill_axis, e1)
uv = np.c_[drill_pts @ e1, drill_pts @ e2]
uv_picks = np.c_[picked_pts @ e1, picked_pts @ e2]

band = 0.15


def search_center(uv, seed, rng, step):
    """在seed附近按网格试圆心：哪个圆心能在0.5mm圆周上圈住最多点"""
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


print("圆心搜索中（几秒钟）...")
ctr, _ = search_center(uv, uv_picks.mean(0), rng=1.5, step=0.15)
ctr, n_fine = search_center(uv, ctr, rng=0.2, step=0.03)
print(f"圆心搜索: 最佳圆心圈住 {n_fine} 个圆柱面点")
assert n_fine >= 100, "找不到半径0.5mm的圆柱面，点选位置可能不在钻尖上，重跑"

# 精修圆心：固定半径Gauss-Newton几何拟合（对单面弧覆盖无偏）
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

# ---- 4.2 轴心+轴向联合精修：固定半径圆柱几何最小二乘（GN，4自由度）----
# PCA轴向对单面弧覆盖有偏；直接最小化 点到轴线距离-R，同时修轴心和轴向
# ctr是绝对投影坐标，本身已是轴线横向位置；只需补上轴向位置（对齐到点云中部）
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
    J[:, 0] = (residual_fn(origin + eps * e1, axis) - res) / eps   # 轴心沿e1
    J[:, 1] = (residual_fn(origin + eps * e2, axis) - res) / eps   # 轴心沿e2
    a1 = axis + eps * e1; a1 /= np.linalg.norm(a1)
    a2 = axis + eps * e2; a2 /= np.linalg.norm(a2)
    J[:, 2] = (residual_fn(origin, a1) - res) / eps                # 轴向绕e1倾
    J[:, 3] = (residual_fn(origin, a2) - res) / eps                # 轴向绕e2倾

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
t = rel @ drill_axis                       # 沿轴坐标
rho = np.linalg.norm(rel - np.outer(t, drill_axis), axis=1)   # 到轴线的径向距离
wall = np.abs(rho - BUR_R) < band
print(f"壁面点径向距离: median={np.median(rho[wall]):.3f}mm（应≈{BUR_R}）")

# ---- 4.4 尖端定位：SF10为平头，两端都是平面；点云前缘即端面，朝牙面的一端是尖端 ----
core = rho <= BUR_R + band
tc = t[core]

bins = np.arange(tc.min(), tc.max() + 0.2, 0.1)
cnt, edges = np.histogram(tc, bins=bins)
nz = np.where(cnt >= 3)[0]
assert len(nz) > 0, "轴附近点太少，检查框选位置"
t_front = edges[nz[-1] + 1]           # 高端端面（密度感知前缘）
# 端面精修：端面圆盘点的轴向中位数 = 真实端面位置（修正桶右沿的+0.1mm前冲）
face = (rho < BUR_R - band) & (t > t_front - 0.3)
if face.sum() >= 5:
    t_front = float(np.median(t[face]))
    print(f"端面精修: {face.sum()}个端面点参与定位")
else:
    t_front = t_front - 0.05          # 端面点太少，退回桶中心
    print("端面点不足，按桶中心定位")
t_back = edges[nz[0]]                 # 低端端面

# 朝向判断：车针指向被钻牙齿，离牙面近的一端是尖端
tree_scan = cKDTree(np.asarray(scan_aligned.vertices))
d_hi = tree_scan.query(origin_pt + t_front * drill_axis)[0]
d_lo = tree_scan.query(origin_pt + t_back * drill_axis)[0]
print(f"端面距牙面: 高端 {d_hi:.2f}mm | 低端 {d_lo:.2f}mm（近的一端定为尖端）")
if d_lo < d_hi:
    drill_axis = -drill_axis
    t = -t
    t_front, t_back = -t_back, -t_front
    print("尖端在选点低端，轴向已翻转")

t_tip = t_front
tip_c = origin_pt + t_tip * drill_axis  # 尖端端面中心（点云坐标系）
L_data = t_front - t_back
print(f"尖端定位: 端面桶含 {cnt[nz[-1]]} 点，覆盖长度 {L_data:.2f}mm")

# ---- 4.5 结果变换回口扫坐标系 ----
Minv = np.linalg.inv(M)
tip_s = Minv[:3, :3] @ tip_c + Minv[:3, 3]
axis_s = Minv[:3, :3] @ drill_axis
axis_s /= np.linalg.norm(axis_s)

print("\n========== 最终结果（口扫坐标系）==========")
print(f"车针尖端: ({tip_s[0]:.3f}, {tip_s[1]:.3f}, {tip_s[2]:.3f}) mm")
print(f"车针朝向: ({axis_s[0]:.4f}, {axis_s[1]:.4f}, {axis_s[2]:.4f})（指向尖端）")
print(f"半径 {BUR_R} mm(固定), 数据覆盖长度 {L_data:.2f} mm（工作部参考 {BUR_HEAD_LEN} mm）")

d_surface = cKDTree(np.asarray(scan.vertices)).query(tip_s)[0]
print(f"自检: 车针尖端距牙面 {d_surface:.2f} mm（钻后数据应贴近牙面）")

# ---- 4.6 可视化：signed distance染色 + 实体SF10(严格8.0mm)，按C切换显隐 ----
if VISUALIZE:
    # ① signed distance 染色（绿=面上 红=偏外 蓝=偏内，±0.1mm满色）
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

    # ② 实体SF10：严格从尖端向后 8.0mm
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
    bur.paint_uniform_color([1.0, 0.0, 1.0])      # 品红，和红绿蓝染色点区分

    # ③ 按 C 显示/隐藏车针模型
    print("提示：可视化窗口中按 C 键 显示/隐藏 品红色车针模型（绿=面上 红=偏外 蓝=偏内）")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="验收：按C切换车针模型显隐")
    vis.add_geometry(scan_aligned)
    vis.add_geometry(drill_colored)
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
# -*- coding: utf-8 -*-
"""
口扫模型 ↔ 点云 配准 (v1.0)
=====================================================
功能: 四点对应(Kabsch)粗配 → 自研裁剪ICP超高精度精配
      (法向兼容过滤 + 分位数裁剪 + Huber点面) → 保存变换矩阵 M。
输出: scan_to_cloud_M.npz (含 M, fitness, cloud_fine 索引)
      供 fit_bur.py 使用。
交互: 口扫网格/点云 各选4个对应点, 中途 fitness 达标自动跳过确认。
"""
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.spatial import cKDTree
import os

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ================== 参数 ==================
SCAN_PATH = "original_scan.ply"
CLOUD_PATH = "merged_cloud_clean.ply"
OUT_M = "scan_to_cloud_M.npz"

VOXEL = 1.0
JUDGE_DIST = 2.0
CROP_DIST = 15.0
N_PICK_ROUGH = 4
MIN_PICK_DIST = 0.5
AUTO_ACCEPT_FITNESS = 0.75
FINE_VOXEL = 0.3
DENSE_N = 200000
VISUALIZE = True

# ================== 加载 ==================
scan = o3d.io.read_triangle_mesh(SCAN_PATH)
cloud = o3d.io.read_point_cloud(CLOUD_PATH)
scan.compute_vertex_normals()
assert len(scan.vertices) > 0 and len(cloud.points) > 0, "文件没读到,检查文件名"
print(f"口扫: {len(scan.vertices)}顶点 | 点云: {len(cloud.points)}点")
scan_pcd = o3d.geometry.PointCloud()
scan_pcd.points = scan.vertices

# ================== 工具函数 ==================
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
    s = o3d.geometry.PointCloud(src)
    s.transform(Mx)
    d, _ = cKDTree(tgt_pts).query(np.asarray(s.points))
    return src.select_by_index(np.where(d < margin)[0])

def kabsch(P, Q):
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

def p2plane_step(Pt, Q, N, huber_k):
    d = ((Pt - Q) * N).sum(1)
    A = np.hstack([np.cross(Pt, N), N])
    w = np.where(np.abs(d) <= huber_k, 1.0, huber_k / np.maximum(np.abs(d), 1e-12))
    sw = np.sqrt(w)
    x, *_ = np.linalg.lstsq(A * sw[:, None], -d * sw, rcond=None)
    M = np.eye(4)
    M[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(x[:3])
    M[:3, 3] = x[3:]
    return M

def icp_custom(src, tgt, M0, thr, iters=30, keep_ratio=0.9, mode="plane"):
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
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=msg)
    vis.add_geometry(geometry)
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()
    return [p.index for p in picked]

def _pv_add(pl, geometry, point_size=3.0, fallback_color="lightblue"):
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
                print(f"  [忽略] 与已选点太近 ({d_min:.2f} mm),请点别处")
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
        pl.enable_surface_point_picking(
            callback=cb,
            picker="cell",
            left_clicking=False,
            show_message="右键依次选择可见表面上的点,选完直接关窗",
        )
    else:
        pl.enable_point_picking(
            callback=cb,
            picker="point",
            left_clicking=False,
            show_message="右键逐个点选,选完直接关窗",
        )
    pl.show()
    return np.array(picked) if picked else np.empty((0, 3))

def pick_landmarks(geometry, side, n_points=N_PICK_ROUGH):
    while True:
        pts = pick_click(
            geometry, f"{side}:按顺序右键选择{n_points}个特征点,选完关窗"
        )
        if len(pts) == n_points:
            return pts
        print(f"选了{len(pts)}个,需要恰好{n_points}个,重选")

def show_candidate(Mc, tgt, note):
    scan_try = o3d.geometry.TriangleMesh(scan)
    scan_try.transform(Mc)
    scan_try.paint_uniform_color([0.2, 0.8, 0.2])
    o3d.visualization.draw_geometries([scan_try, tgt], window_name=note)

def gate(M0, tgt_show, ref_src, ref_tgt, label):
    print(f"{label}评分: {eval_fit(ref_src, ref_tgt, M0):.3f}")
    show_candidate(M0, tgt_show, f"{label} — 关窗后到终端回答")
    ans = input(f"{label}对吗?y接受 / 回车重选: ").strip().lower()
    return M0 if ans == "y" else None

# ---------- 主流程 ----------
s_down = make_down(scan_pcd)
c_all = make_down(cloud)

while True:
    print()
    print(
        f"先在口扫上按顺序选{N_PICK_ROUGH}个特征点(建议:左磨牙尖→右磨牙尖→门牙中缝→尖牙尖,尽量张开跨度)"
    )
    print("  窗口里直接显示口扫三角网格,请右键选择可见表面上的牙尖")
    P = pick_landmarks(scan, "【口扫】", N_PICK_ROUGH)
    print(f"再在点云上按【相同顺序】选对应的{N_PICK_ROUGH}个点")
    print("  窗口里只有点云,请按顺序右键点同一批牙尖")
    Q = pick_landmarks(cloud, "【点云】", N_PICK_ROUGH)

    M_rough = kabsch(P, Q)
    M_rough, _ = refine(
        s_down, c_all, M_rough,
        stages=((8.0, "pp", 0.7, 40), (5.0, "pp", 0.75, 40), (3.0, "pp", 0.8, 40)),
    )
    fit_rough = eval_fit(s_down, c_all, M_rough)
    print(f"自动粗配 fitness: {fit_rough:.3f}")

    if fit_rough < AUTO_ACCEPT_FITNESS:
        M_rough = gate(M_rough, cloud, s_down, c_all, "粗配")
        if M_rough is None:
            continue
    else:
        print("粗配结果达到自动接受阈值,继续精配...")

    # 裁剪去杂
    scan_rough = o3d.geometry.TriangleMesh(scan)
    scan_rough.transform(M_rough)
    d, _ = cKDTree(np.asarray(scan_rough.vertices)).query(np.asarray(cloud.points))
    cloud_fine_idx = np.where(d < CROP_DIST)[0]
    cloud_fine = cloud.select_by_index(cloud_fine_idx)
    print(f"裁剪去杂: {len(cloud_fine.points)}/{len(cloud.points)} 点进入精配")

    # 精配
    c2 = make_down(cloud_fine)
    print(f"[体检] 粗配结果在精配点云上的评分: {eval_fit(s_down, c2, M_rough):.3f}")
    M, _ = refine(s_down, c2, M_rough)
    M, _ = ultra_refine(
        make_down(scan_pcd, FINE_VOXEL), make_down(cloud_fine, FINE_VOXEL), M
    )

    # 最终贴合
    print("最终贴合:口扫表面泊松盘采样20万点 + 0.25mm点云 + Huber点面ICP...")
    s_dense = scan.sample_points_poisson_disk(
        DENSE_N, init_factor=5, use_triangle_normal=True
    )
    c_dense = make_down(cloud_fine, 0.25)
    M, _ = refine(
        s_dense, c_dense, M,
        stages=(
            (0.4, "plane", 0.95, 40),
            (0.2, "plane", 0.95, 40),
            (0.15, "plane", 0.95, 40),
            (0.1, "plane", 0.97, 40),
        ),
    )

    fit_report(s_dense, c_dense, M, "精配贴合度")

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
    verify.paint_uniform_color([1, 1, 1])
    o3d.visualization.draw_geometries(
        [verify, cloud_fine], window_name="验收:表面应白彩交替贴合"
    )
    sd_dist, _ = cKDTree(np.asarray(cloud_fine.points)).query(
        np.asarray(scan_aligned.vertices)
    )
    heat = np.clip(sd_dist / 0.5, 0, 1)
    verify2 = o3d.geometry.TriangleMesh(scan_aligned)
    verify2.vertex_colors = o3d.utility.Vector3dVector(
        np.c_[heat, 1.0 - heat, np.zeros_like(heat)]
    )
    o3d.visualization.draw_geometries(
        [verify2], window_name="残差热力图:绿=贴合 红=偏差≥0.5mm"
    )

# ---------- 保存结果 ----------
np.savez(
    OUT_M,
    M=M,
    fit_rough=fit_rough,
    fit_fine=fit_fine,
    cloud_fine_idx=cloud_fine_idx,
)
print(f"\n配准完成, 变换矩阵已保存到 {OUT_M}")
print(f"  M = \n{M}")
print(f"  精配 fitness: {fit_fine:.3f}")
print(f"  cloud_fine 点数: {len(cloud_fine.points)} (索引已保存)")
import copy
import json
import math
import os
import xml.etree.ElementTree as ET

print("[诊断] 脚本已启动，正在加载 numpy 和 open3d...", flush=True)
import numpy as np
print("[诊断] numpy 加载完成，正在加载 open3d...", flush=True)
import open3d as o3d
print("[诊断] open3d 加载完成，开始读取配置...", flush=True)


# ============================== 配置区 ==============================

# 原始口扫：只有牙齿，是固定参考坐标系
ORIGINAL_SCAN_PATH = "original_scan.ply"

# 带车针扫描：包含牙齿和车针
DRILL_SCAN_PATH = "drill_scan.ply"

# 如果你已经把 MeshLab 配准结果 Freeze 后导出成新文件：
#   1. 将 DRILL_SCAN_PATH 指向这个新文件；
#   2. 将下面改为 True。
SOURCE_ALREADY_ALIGNED = False

# 方式一：直接读取未合并 MeshLab 工程。支持 .mlp；如果你的工程被误存成 matrix.txt，也填 matrix.txt。
# 注意：必须是包含两个独立 MLMesh 层的工程，不能是 Flatten 后的 Merged Mesh 工程。
MESHLAB_PROJECT_PATH = "matrix.txt"

# 方式二：纯文本 4x4 矩阵缓存。首次自动/手动配准成功后会自动生成。
TRANSFORM_CACHE_PATH = "transform_matrix.txt"

# 若为 True，读取不到有效矩阵就报错；若为 False，则退回到脚本内手动粗配准。
STRICT_MATRIX_MODE = False

# 脚本内手动粗配准后是否再做 ICP。若粗配准已经较好，建议 False，避免 ICP 把结果带偏。
DO_ICP_AFTER_MANUAL = False
ICP_THRESHOLD_MM = 1.0
ICP_MAX_ITER = 200

# 数据预处理。口扫单位通常为 mm；若车针直径约 1~2 mm，0.05 mm 已足够。
TARGET_MESH_SAMPLE_POINTS = 100000
SOURCE_MESH_SAMPLE_POINTS = 200000
SOURCE_VOXEL_SIZE_MM = 0.05

# 车针自动分割：源点距离原始口扫超过该值，初步认为是不属于牙齿的点。
DIFF_DISTANCE_MM = 0.60
DBSCAN_EPS_MM = 1.20
DBSCAN_MIN_POINTS = 20

# 车针簇选择方式：
#   "manual_cluster"：显示全部蓝色候选点，人工 Shift+左键点选真正的螺旋车针；
#   "auto"          ：根据细长程度自动选簇。
# 针对当前截图，建议使用 manual_cluster，避免牙齿残差被误选。
DRILL_SELECTION_MODE = "manual_cluster"

# 若知道车针公称直径，请填入，例如 1.6；不知道则填 None。
# 填了以后会优先使用直径附近的空间点估计半径，能明显抑制偏差。
EXPECTED_DRILL_DIAMETER_MM = None
EXPECTED_DIAMETER_TOLERANCE = 0.25  # 允许相对偏差，0.25 表示 ±25%

# 差异分割失败时的手动球形提取半径。
FALLBACK_RADIUS_MM = 3.0

# 是否显示中间窗口。
SHOW_REGISTRATION_CHECK = True
SHOW_SEGMENTATION_CHECK = True

OUTPUT_JSON = "drill_pose_v7.json"
OUTPUT_DRILL_PCD = "drill_cluster_v7.ply"

# ==================================================================


def log(message):
    print(message, flush=True)


def is_identity_matrix(matrix, tolerance=1e-8):
    return np.allclose(matrix, np.eye(4), atol=tolerance)


def validate_transform(matrix, name="matrix"):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} 不是 4x4 矩阵，当前形状为 {matrix.shape}")
    rotation = matrix[:3, :3]
    determinant = np.linalg.det(rotation)
    log(f"[矩阵检查] {name}: det(R) = {determinant:.6f}")
    if determinant <= 0:
        raise ValueError(f"{name} 的旋转部分行列式为负，可能包含镜像变换")
    if abs(determinant - 1.0) > 0.05:
        log(f"[警告] {name} 的 det(R) 明显偏离 1，可能包含缩放或文件内容错误")
    return matrix


def read_meshlab_project_matrix(path):
    """
    读取 MeshLab 工程中的 MLMatrix44。

    返回：
        matrix, description, matrix_count, non_identity_count
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as exc:
        raise ValueError(f"无法解析 MeshLab 工程 {path}: {exc}") from exc

    matrices = []

    def local_name(tag):
        return tag.split("}")[-1]

    for mesh_node in root.iter():
        if local_name(mesh_node.tag) != "MLMesh":
            continue

        label = mesh_node.attrib.get("label", "未命名层")
        filename = mesh_node.attrib.get("filename", "")
        matrix_node = None
        for child in mesh_node:
            if local_name(child.tag) == "MLMatrix44":
                matrix_node = child
                break

        if matrix_node is None or matrix_node.text is None:
            continue

        values = [float(x) for x in matrix_node.text.split()]
        if len(values) != 16:
            continue

        matrix = np.asarray(values, dtype=float).reshape(4, 4)
        matrices.append((label, filename, matrix))

    if not matrices:
        raise ValueError("MeshLab 工程中没有找到任何 MLMatrix44")

    non_identity = [item for item in matrices if not is_identity_matrix(item[2])]
    if non_identity:
        # MeshLab 工程通常：第一个为参考层单位矩阵，后续非单位矩阵为被配准层。
        label, filename, matrix = non_identity[-1]
        description = f"MeshLab 层 '{label}' ({filename})，共 {len(matrices)} 个矩阵"
        return validate_transform(matrix, "MeshLab MLMatrix44"), description, len(matrices), len(non_identity)

    # 只有单位矩阵时，只能说明文件已合并或原本未变换，不能恢复手动配准。
    label, filename, matrix = matrices[-1]
    description = f"MeshLab 层 '{label}' ({filename})，所有矩阵均为单位矩阵"
    return matrix, description, len(matrices), 0


def read_transform(path):
    """自动识别纯文本 4x4 矩阵或 MeshLab 工程。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        content = file.read(2048).lstrip()

    if content.startswith("<"):
        return read_meshlab_project_matrix(path)

    matrix = np.loadtxt(path, dtype=float)
    return validate_transform(matrix, path), f"纯文本矩阵 {path}", 1, int(not is_identity_matrix(matrix))


def load_as_pointcloud(path, mesh_sample_points, voxel_size=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件：{path}")

    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) > 0:
        log(f"[加载] {path}: 三角网格，采样 {mesh_sample_points} 点")
        pcd = mesh.sample_points_uniformly(number_of_points=mesh_sample_points)
    else:
        pcd = o3d.io.read_point_cloud(path)
        log(f"[加载] {path}: 点云，{len(pcd.points)} 点")

    if len(pcd.points) == 0:
        raise ValueError(f"文件没有可用点：{path}")

    if voxel_size is not None and voxel_size > 0:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        log(f"[降采样] 体素 {voxel_size} mm: {before} -> {len(pcd.points)} 点")

    return pcd


def downsample_for_display(pcd, max_points=250000):
    count = len(pcd.points)
    if count <= max_points:
        return copy.deepcopy(pcd)
    every = max(1, int(math.ceil(count / max_points)))
    return pcd.uniform_down_sample(every)


def pick_points(pcd, title, min_count=0):
    log("\n" + title)
    log("  Shift + 鼠标左键：选点")
    log("  鼠标左键拖拽：旋转；滚轮：缩放；右键拖拽：平移")
    log("  Q：结束选点")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    indices = vis.get_picked_points()
    points = np.asarray(pcd.points)
    picked = points[indices] if len(indices) > 0 else np.empty((0, 3))
    log(f"[选点] 共选择 {len(picked)} 个点")

    if len(picked) < min_count:
        raise ValueError(f"至少需要 {min_count} 个点，当前只有 {len(picked)} 个")
    return picked, indices


def kabsch_transform(source_points, target_points):
    if len(source_points) != len(target_points):
        raise ValueError(f"对应点数量不同：source={len(source_points)}, target={len(target_points)}")
    if len(source_points) < 3:
        raise ValueError("至少需要 3 对对应点")

    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return validate_transform(transform, "手动粗配准矩阵")


def manual_registration(source_pcd, target_pcd):
    log("\n" + "=" * 64)
    log("未找到有效 MeshLab 矩阵，进入脚本内手动粗配准")
    log("请先在蓝色带车针扫描上选 4~6 个牙齿特征点")
    log("再在灰色原始口扫上按完全相同顺序选择对应点")
    log("=" * 64)

    source_display = copy.deepcopy(source_pcd)
    target_display = copy.deepcopy(target_pcd)
    source_display.paint_uniform_color([0.15, 0.50, 1.00])
    target_display.paint_uniform_color([0.70, 0.70, 0.70])

    source_points, _ = pick_points(
        source_display,
        "蓝色带车针扫描：选择牙齿特征点，Q 结束",
        min_count=3,
    )
    target_points, _ = pick_points(
        target_display,
        "灰色原始口扫：按相同顺序选择对应点，Q 结束",
        min_count=3,
    )

    transform = kabsch_transform(source_points, target_points)

    aligned_preview = copy.deepcopy(source_pcd)
    aligned_preview.transform(transform)
    source_show = downsample_for_display(aligned_preview)
    target_show = downsample_for_display(target_pcd)
    source_show.paint_uniform_color([0.15, 0.50, 1.00])
    target_show.paint_uniform_color([0.70, 0.70, 0.70])

    log("[检查] 粗配准预览：蓝色应大致覆盖灰色牙齿，关闭窗口继续")
    o3d.visualization.draw_geometries(
        [target_show, source_show],
        window_name="手动粗配准检查：蓝色=带车针扫描，灰色=原始口扫",
    )
    return transform


def run_icp(source_pcd, target_pcd, initial_transform):
    source_copy = copy.deepcopy(source_pcd)
    target_copy = copy.deepcopy(target_pcd)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(ICP_MAX_ITER)
    )
    result = o3d.pipelines.registration.registration_icp(
        source_copy,
        target_copy,
        float(ICP_THRESHOLD_MM),
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )
    log(f"[ICP] fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.4f} mm")
    return validate_transform(result.transformation, "ICP 最终矩阵")


def registration_quality(aligned_source, target_pcd, max_test_points=20000, inlier_limit=1.5):
    """用源点到目标点云的最近邻距离评估配准，只统计牙齿重合区域的近距离点。"""
    source_points = np.asarray(aligned_source.points)
    if len(source_points) > max_test_points:
        rng = np.random.default_rng(2026)
        indices = rng.choice(len(source_points), size=max_test_points, replace=False)
        test_points = source_points[indices]
    else:
        test_points = source_points

    tree = o3d.geometry.KDTreeFlann(target_pcd)
    distances = np.empty(len(test_points), dtype=float)
    for i, point in enumerate(test_points):
        _, _, squared = tree.search_knn_vector_3d(point, 1)
        distances[i] = math.sqrt(float(squared[0]))

    inliers = distances[distances <= inlier_limit]
    if len(inliers) < 20:
        inliers = distances

    quality = {
        "median_mm": float(np.median(inliers)),
        "p90_mm": float(np.quantile(inliers, 0.90)),
        "inlier_ratio": float(np.mean(distances <= inlier_limit)),
    }

    log("\n[配准质量]")
    log(f"  中位距离: {quality['median_mm']:.4f} mm")
    log(f"  90% 距离: {quality['p90_mm']:.4f} mm")
    log(f"  {inlier_limit:.1f} mm 内点比例: {quality['inlier_ratio']:.2%}")
    log("  一般牙齿重合区中位距离 < 0.20 mm、90% 距离 < 0.50 mm 可接受")
    return quality


def nearest_distances_to_reference(source_pcd, reference_pcd):
    points = np.asarray(source_pcd.points)
    tree = o3d.geometry.KDTreeFlann(reference_pcd)
    distances = np.empty(len(points), dtype=float)

    log(f"[差异分割] 正在计算 {len(points)} 个源点到原始口扫的最近距离...")
    for i, point in enumerate(points):
        _, _, squared = tree.search_knn_vector_3d(point, 1)
        distances[i] = math.sqrt(float(squared[0]))
        if (i + 1) % 100000 == 0:
            log(f"  已完成 {i + 1}/{len(points)}")
    return distances


def cluster_shape_score(points):
    centered = points - np.mean(points, axis=0)
    covariance = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    # 细长物体：最大方向长度远大于另外两个方向
    elongation = math.sqrt(eigenvalues[-1] / eigenvalues[-2])
    flatness = math.sqrt(eigenvalues[-2] / eigenvalues[-3])
    return elongation / max(flatness, 1e-6), elongation, flatness


def automatic_drill_segmentation(aligned_source, target_pcd):
    """
    差异分割 + 车针簇选择。

    manual_cluster 模式下，先显示所有蓝色差异候选点，由用户直接点选真正的
    螺旋车针；脚本提取被点中的一个或多个 DBSCAN 簇，避免自动评分误选牙齿残差。
    """
    distances = nearest_distances_to_reference(aligned_source, target_pcd)
    points = np.asarray(aligned_source.points)
    candidate_mask = distances > float(DIFF_DISTANCE_MM)
    candidate_points = points[candidate_mask]

    log(f"[差异分割] 距离阈值 {DIFF_DISTANCE_MM} mm，候选点 {len(candidate_points)} 个")
    if len(candidate_points) < DBSCAN_MIN_POINTS:
        raise RuntimeError("差异候选点太少，可能配准失败或 DIFF_DISTANCE_MM 过大")

    candidate_pcd = o3d.geometry.PointCloud()
    candidate_pcd.points = o3d.utility.Vector3dVector(candidate_points)
    labels = np.asarray(
        candidate_pcd.cluster_dbscan(
            eps=float(DBSCAN_EPS_MM),
            min_points=int(DBSCAN_MIN_POINTS),
            print_progress=False,
        )
    )

    clusters = []
    for label in sorted(set(labels)):
        if label < 0:
            continue
        cluster_points = candidate_points[labels == label]
        if len(cluster_points) < DBSCAN_MIN_POINTS:
            continue
        score, elongation, flatness = cluster_shape_score(cluster_points)
        clusters.append(
            {
                "label": int(label),
                "points": cluster_points,
                "count": int(len(cluster_points)),
                "score": float(score * math.log10(len(cluster_points) + 10.0)),
                "elongation": float(elongation),
                "flatness": float(flatness),
            }
        )

    if not clusters:
        raise RuntimeError("DBSCAN 未找到有效车针簇，请调大 DBSCAN_EPS_MM 或调小 DIFF_DISTANCE_MM")

    clusters.sort(key=lambda item: item["score"], reverse=True)
    log("\n[聚类候选]")
    for rank, cluster in enumerate(clusters[:10], start=1):
        log(
            f"  #{rank}: label={cluster['label']}, 点数={cluster['count']}, "
            f"细长比={cluster['elongation']:.2f}, 得分={cluster['score']:.2f}"
        )

    mode = str(DRILL_SELECTION_MODE).strip().lower()
    if mode == "manual_cluster":
        log("\n[人工选簇] 接下来只显示蓝色差异候选点")
        log("  请在真正的长条螺旋车针上 Shift+左键点 1~4 下")
        log("  如果车针被分成几段，请在不同段上各点一下，然后按 Q")
        candidate_show = copy.deepcopy(candidate_pcd)
        candidate_show.paint_uniform_color([0.20, 0.50, 1.00])
        _, picked_indices = pick_points(
            candidate_show,
            "只显示蓝色候选点：请点选真正的螺旋车针，Q 结束",
            min_count=1,
        )

        selected_labels = []
        valid_indices = np.where(labels >= 0)[0]
        for picked_index in picked_indices:
            label = int(labels[picked_index])
            if label < 0 and len(valid_indices) > 0:
                # 如果点到了离群噪声，改选最近的有效簇点
                picked_point = candidate_points[picked_index]
                nearest_valid = valid_indices[
                    np.argmin(np.linalg.norm(candidate_points[valid_indices] - picked_point, axis=1))
                ]
                if np.linalg.norm(candidate_points[nearest_valid] - picked_point) <= max(2.0, 2.0 * DBSCAN_EPS_MM):
                    label = int(labels[nearest_valid])
            if label >= 0 and label not in selected_labels:
                selected_labels.append(label)

        if not selected_labels:
            raise RuntimeError("没有选中任何有效候选簇，请重新运行并直接点在蓝色螺旋车针上")

        selected_mask = np.isin(labels, np.asarray(selected_labels, dtype=int))
        selection_mode = "manual_cluster"
        log(f"[人工选簇] 已选择 label={selected_labels}，共 {np.count_nonzero(selected_mask)} 点")
    elif mode == "auto":
        selected_labels = [clusters[0]["label"]]
        selected_mask = labels == selected_labels[0]
        selection_mode = "auto"
        log(f"[自动选簇] 选择得分最高的 label={selected_labels[0]}")
    else:
        raise ValueError("DRILL_SELECTION_MODE 只能是 'manual_cluster' 或 'auto'")

    drill_points = candidate_points[selected_mask]
    if len(drill_points) < 30:
        raise RuntimeError("选中的车针点太少，请检查是否点到了正确蓝色螺旋区域")

    score, elongation, flatness = cluster_shape_score(drill_points)
    selected = {
        "label": selected_labels,
        "count": int(len(drill_points)),
        "score": float(score),
        "elongation": float(elongation),
        "flatness": float(flatness),
        "selection_mode": selection_mode,
    }

    drill_pcd = o3d.geometry.PointCloud()
    drill_pcd.points = o3d.utility.Vector3dVector(drill_points)

    if SHOW_SEGMENTATION_CHECK:
        target_show = downsample_for_display(target_pcd, 180000)
        unselected_points = candidate_points[~selected_mask]
        unselected_pcd = o3d.geometry.PointCloud()
        unselected_pcd.points = o3d.utility.Vector3dVector(unselected_points)
        drill_show = copy.deepcopy(drill_pcd)
        target_show.paint_uniform_color([0.72, 0.72, 0.72])
        unselected_pcd.paint_uniform_color([0.20, 0.50, 1.00])
        drill_show.paint_uniform_color([1.00, 0.45, 0.00])
        log("[检查] 分割结果：橙色=你选中的车针，蓝色=其他差异候选，关闭窗口继续")
        o3d.visualization.draw_geometries(
            [target_show, unselected_pcd, drill_show],
            window_name="车针分割检查：橙色应覆盖长条螺旋车针",
        )

    return drill_pcd, selected, distances

def fallback_manual_segmentation(aligned_source):
    log("\n[备选] 自动差异分割失败，改为在车针上选 3~5 个点并提取球形邻域")
    display = downsample_for_display(aligned_source, 300000)
    _, indices = pick_points(display, "在车针表面选 3~5 个点，Q 结束", min_count=1)

    display_points = np.asarray(display.points)
    source_points = np.asarray(aligned_source.points)
    seed_center = np.mean(display_points[indices], axis=0)
    distances = np.linalg.norm(source_points - seed_center, axis=1)
    mask = distances <= float(FALLBACK_RADIUS_MM)

    drill_pcd = o3d.geometry.PointCloud()
    drill_pcd.points = o3d.utility.Vector3dVector(source_points[mask])
    if len(drill_pcd.points) < 30:
        raise RuntimeError("手动提取的车针点仍太少，请增大 FALLBACK_RADIUS_MM 或重新选点")

    info = {
        "label": -1,
        "count": int(len(drill_pcd.points)),
        "score": 0.0,
        "elongation": 0.0,
        "flatness": 0.0,
    }
    return drill_pcd, info


def orthogonal_basis(axis):
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, world_up)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    return u, v


def fit_pca_axis(points):
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    axis /= np.linalg.norm(axis)
    # 固定符号，保证多次运行输出稳定
    if axis[2] < 0:
        axis = -axis
    return center, axis


def fit_cylinder_robust(points):
    if len(points) < 30:
        raise ValueError(f"车针点太少，无法拟合：{len(points)}")

    work = np.asarray(points, dtype=float).copy()
    center = None
    axis = None

    for iteration in range(4):
        if len(work) < 20:
            break
        center, axis = fit_pca_axis(work)
        relative = work - center
        axial = relative @ axis
        radial_vector = relative - np.outer(axial, axis)
        radial = np.linalg.norm(radial_vector, axis=1)

        axial_low, axial_high = np.quantile(axial, [0.01, 0.99])
        radial_high = np.quantile(radial, 0.95)
        keep = (
            (axial >= axial_low)
            & (axial <= axial_high)
            & (radial <= radial_high)
        )

        # 已知公称直径时，第二轮开始保留直径附近的表面点
        if EXPECTED_DRILL_DIAMETER_MM is not None and iteration >= 1:
            expected_radius = float(EXPECTED_DRILL_DIAMETER_MM) / 2.0
            tolerance = expected_radius * float(EXPECTED_DIAMETER_TOLERANCE)
            diameter_keep = np.abs(radial - expected_radius) <= tolerance
            if np.count_nonzero(diameter_keep) >= 20:
                keep &= diameter_keep

        if np.count_nonzero(keep) >= 20:
            work = work[keep]

    center, axis = fit_pca_axis(work)
    u, v = orthogonal_basis(axis)
    relative = work - center
    axial = relative @ axis
    x = relative @ u
    y = relative @ v

    # 最小二乘圆心，修正部分扫描造成的均值偏移
    a_matrix = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x))])
    b_vector = x ** 2 + y ** 2
    solution, _, _, _ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    circle_x, circle_y = float(solution[0]), float(solution[1])

    radial = np.sqrt((x - circle_x) ** 2 + (y - circle_y) ** 2)
    if EXPECTED_DRILL_DIAMETER_MM is not None:
        expected_radius = float(EXPECTED_DRILL_DIAMETER_MM) / 2.0
        tolerance = expected_radius * float(EXPECTED_DIAMETER_TOLERANCE)
        near_surface = np.abs(radial - expected_radius) <= tolerance
        if np.count_nonzero(near_surface) >= 10:
            radius = float(np.median(radial[near_surface]))
        else:
            log("[警告] 点云未充分落在公称直径附近，使用公称直径的一半作为半径")
            radius = expected_radius
    else:
        # 取外侧 30% 点的中位数，避免内部噪声和少量飞点共同影响半径
        surface_cutoff = np.quantile(radial, 0.70)
        surface_radii = radial[radial >= surface_cutoff]
        radius = float(np.median(surface_radii))

    axial_low, axial_high = np.quantile(axial, [0.01, 0.99])
    height = float(axial_high - axial_low)
    axial_mid = float((axial_low + axial_high) / 2.0)
    cylinder_center = center + axial_mid * axis + circle_x * u + circle_y * v

    residuals = np.abs(radial - radius)
    result = {
        "center": cylinder_center,
        "axis": axis,
        "radius": float(radius),
        "diameter": float(2.0 * radius),
        "height": height,
        "point_count": int(len(work)),
        "radial_residual_median": float(np.median(residuals)),
        "radial_residual_p90": float(np.quantile(residuals, 0.90)),
    }

    log("\n========== 圆柱拟合结果 ==========")
    log(f"参与拟合点数: {result['point_count']}")
    log(f"中心位置   : {cylinder_center.tolist()} mm")
    log(f"轴线方向   : {axis.tolist()}")
    log(f"拟合半径   : {result['radius']:.4f} mm")
    log(f"拟合直径   : {result['diameter']:.4f} mm")
    log(f"拟合高度   : {result['height']:.4f} mm")
    log(f"径向残差中位数: {result['radial_residual_median']:.4f} mm")
    log(f"径向残差 90% : {result['radial_residual_p90']:.4f} mm")
    if EXPECTED_DRILL_DIAMETER_MM is not None:
        error = result["diameter"] - float(EXPECTED_DRILL_DIAMETER_MM)
        log(f"与公称直径差: {error:+.4f} mm")
    log("==================================")
    return result


def rotation_between_vectors(source_vector, target_vector):
    source = np.asarray(source_vector, dtype=float)
    target = np.asarray(target_vector, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)

    cross = np.cross(source, target)
    sin_angle = np.linalg.norm(cross)
    cos_angle = float(np.dot(source, target))

    if sin_angle < 1e-10:
        if cos_angle > 0:
            return np.eye(3)
        # 反向时绕任意垂直轴旋转 180 度
        helper = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * math.pi)

    axis = cross / sin_angle
    angle = math.atan2(sin_angle, cos_angle)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def create_cylinder_mesh(fit):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(fit["radius"]),
        height=float(fit["height"]),
        resolution=48,
        split=1,
    )
    mesh.paint_uniform_color([1.0, 0.05, 0.05])
    rotation = rotation_between_vectors(np.array([0.0, 0.0, 1.0]), fit["axis"])
    mesh.rotate(rotation, center=np.zeros(3))
    mesh.translate(fit["center"])
    return mesh


def create_axis_frame(fit, size=4.0):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size))
    rotation = rotation_between_vectors(np.array([0.0, 0.0, 1.0]), fit["axis"])
    frame.rotate(rotation, center=np.zeros(3))
    frame.translate(fit["center"])
    return frame


def save_transform(path, transform):
    np.savetxt(path, transform, fmt="%.10f")
    log(f"[保存] 纯文本 4x4 矩阵已写入：{path}")


def resolve_registration(source_pcd, target_pcd):
    """返回将 DRILL_SCAN 变换到 ORIGINAL_SCAN 坐标系的 4x4 矩阵。"""
    if SOURCE_ALREADY_ALIGNED:
        log("[配准] SOURCE_ALREADY_ALIGNED=True，认为输入文件已包含 MeshLab 配准结果")
        return np.eye(4), "Freeze 后导出的已对齐文件"

    # 优先使用缓存的纯文本矩阵
    if os.path.exists(TRANSFORM_CACHE_PATH):
        matrix, description, _, _ = read_transform(TRANSFORM_CACHE_PATH)
        log(f"[配准] 使用缓存矩阵：{description}")
        return matrix, description

    # 其次读取 MeshLab 工程
    if MESHLAB_PROJECT_PATH and os.path.exists(MESHLAB_PROJECT_PATH):
        matrix, description, matrix_count, non_identity_count = read_transform(MESHLAB_PROJECT_PATH)
        log(f"[配准] 读取到：{description}")

        if non_identity_count == 0:
            message = (
                "MeshLab 工程里只有单位矩阵。通常是因为你先 Flatten/Merge 后再保存工程，"
                "导致矩阵被烘焙成 Merged Mesh 的单位矩阵；这种文件无法恢复手动配准矩阵。"
            )
            if STRICT_MATRIX_MODE:
                raise RuntimeError(message)
            log("[警告] " + message)
            log("[警告] 将退回脚本内手动粗配准")
        else:
            save_transform(TRANSFORM_CACHE_PATH, matrix)
            return matrix, description

    if STRICT_MATRIX_MODE:
        raise RuntimeError("未找到有效配准矩阵，且 STRICT_MATRIX_MODE=True")

    transform = manual_registration(source_pcd, target_pcd)
    if DO_ICP_AFTER_MANUAL:
        transform = run_icp(source_pcd, target_pcd, transform)
    save_transform(TRANSFORM_CACHE_PATH, transform)
    return transform, "脚本内手动粗配准"


def main():
    log("=" * 72)
    log("车针自动配准与圆柱拟合 v7")
    log("=" * 72)

    log("\n[1/6] 加载数据")
    target_pcd = load_as_pointcloud(
        ORIGINAL_SCAN_PATH,
        mesh_sample_points=TARGET_MESH_SAMPLE_POINTS,
        voxel_size=None,
    )
    source_pcd = load_as_pointcloud(
        DRILL_SCAN_PATH,
        mesh_sample_points=SOURCE_MESH_SAMPLE_POINTS,
        voxel_size=SOURCE_VOXEL_SIZE_MM,
    )
    log(f"  原始口扫参考点数: {len(target_pcd.points)}")
    log(f"  带车针扫描点数  : {len(source_pcd.points)}")

    log("\n[2/6] 获取/应用配准矩阵")
    transform, transform_source = resolve_registration(source_pcd, target_pcd)
    aligned_source = copy.deepcopy(source_pcd)
    aligned_source.transform(transform)
    quality = registration_quality(aligned_source, target_pcd)

    if SHOW_REGISTRATION_CHECK:
        target_show = downsample_for_display(target_pcd, 180000)
        source_show = downsample_for_display(aligned_source, 220000)
        target_show.paint_uniform_color([0.72, 0.72, 0.72])
        source_show.paint_uniform_color([0.15, 0.50, 1.00])
        log("[检查] 配准结果：蓝色牙齿应覆盖灰色牙齿，关闭窗口继续")
        o3d.visualization.draw_geometries(
            [target_show, source_show],
            window_name="配准检查：灰色=原始口扫，蓝色=带车针扫描",
        )

    log("\n[3/6] 分割车针")
    try:
        drill_pcd, cluster_info, difference_distances = automatic_drill_segmentation(
            aligned_source, target_pcd
        )
    except Exception as exc:
        log(f"[警告] 自动分割失败：{exc}")
        drill_pcd, cluster_info = fallback_manual_segmentation(aligned_source)
        difference_distances = None

    log(f"[分割] 最终车针点数: {len(drill_pcd.points)}")

    log("\n[4/6] 稳健圆柱拟合")
    points = np.asarray(drill_pcd.points)
    fit = fit_cylinder_robust(points)

    log("\n[5/6] 可视化最终拟合")
    target_show = downsample_for_display(target_pcd, 180000)
    drill_show = copy.deepcopy(drill_pcd)
    target_show.paint_uniform_color([0.72, 0.72, 0.72])
    drill_show.paint_uniform_color([1.00, 0.45, 0.00])
    cylinder_mesh = create_cylinder_mesh(fit)
    axis_frame = create_axis_frame(fit, size=max(4.0, fit["height"] * 0.20))

    log("  灰色=原始口扫，橙色=车针点云，红色=拟合圆柱，RGB坐标轴 Z=车针轴线")
    o3d.visualization.draw_geometries(
        [target_show, drill_show, cylinder_mesh, axis_frame],
        window_name="最终结果：红色圆柱应贴合橙色车针点云",
    )

    log("\n[6/6] 保存输出")
    o3d.io.write_point_cloud(OUTPUT_DRILL_PCD, drill_pcd)
    output = {
        "position_mm": fit["center"].tolist(),
        "axis_direction": fit["axis"].tolist(),
        "radius_mm": fit["radius"],
        "diameter_mm": fit["diameter"],
        "height_mm": fit["height"],
        "fit_point_count": fit["point_count"],
        "radial_residual_median_mm": fit["radial_residual_median"],
        "radial_residual_p90_mm": fit["radial_residual_p90"],
        "expected_diameter_mm": EXPECTED_DRILL_DIAMETER_MM,
        "registration_transform_source": transform_source,
        "registration_quality": quality,
        "selected_cluster": cluster_info,
        "transformation_matrix": transform.tolist(),
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    log(f"[保存] 车针点云：{OUTPUT_DRILL_PCD}")
    log(f"[保存] 位姿结果：{OUTPUT_JSON}")
    log("\n任务完成")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
多视角点云融合 v9.6.4：CUDA双精度、实测几何引导及可选深度校正
=====================================================================
v9.6.4: --sphere-guide uses measured sphere geometry. --surface-guide requires the same complete static object.
--calibrate-depth explicitly changes output point geometry; raw PLY files are never modified.
Depth correction must precede each exported rigid matrix; see depth_calibration.json.
Calibration PDF measurements are evaluation only, never read by registration.
--manual-bridge enables optional interactive component bridging after automatic rescue.
Default is fully automatic; manual waiting is included in elapsed time.

v9.4 新增：预导入线程限制、共享预处理、精确有界KDTree搜索、稳定SE(3)步长、
v9.5: CUDA float64 ICP with exact KDTree; bounded exact acceptance scores.
受双向验证的FGR初值（失败仍回退原RANSAC）、有几何回退的稳健闭环优化、
独立视角计数并行融合、精确排除自身帧的共享KDTree评分、现有结果精修入口。
不按时间截断任务；performance.json 如实报告是否达到300秒。
--refine-existing 不包含原始配准时间；不能将其耗时当作冷启动耗时。
下列 v3.3/v9.3 说明是历史记录；v9.4 的 jobs=1 与并行使用相同窗口候选集。
=====================================================================
v3.3 在 v3.2 基础上, 只改两两配准为并行:
  ★ 多进程并行: 帧对分配给 n_jobs 个子进程(默认 CPU核数-1),
    子进程按需加载 ply 并缓存(每帧每进程只读一次), 不传输大点云;
  ★ 窗口制替代顺序早停: 只配 |i-j| <= pair_window(默认8) 的帧对
    (邻近帧才可能重叠), 与 pair_ring_step 跨帧边配合保证图连通;
  ★ n_jobs=1 时自动回退原顺序逻辑(行为与 v3.2 完全一致)。

v9.3 本版以已经验证效果良好的Source-Style Main为默认算法基础；
新增定量审计（逐帧、双向帧对、三角闭环），不参与任何配准/ICP。
默认守门改为 audit（只审计不剔除）：修复旧版误把自身当最近邻的问题，
--gate-enforce 才运行严格 leave-one-out 守门；--gate-legacy 可复现旧行为。
v9.3.2 默认用组内 BFS 位姿构建主图，自动和手动接入均把多帧分量作为一个刚体。

"""
import os, sys, glob, json, time, argparse, csv, hashlib
_SCRIPT_STARTED_AT = time.monotonic()
from pathlib import Path
from collections import OrderedDict
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Set native thread limits BEFORE importing numerical libraries (also in spawn workers).
for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_var] = '1'
import numpy as np
import open3d as o3d
try:
    o3d.utility.set_num_threads(1)
except AttributeError:
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
from scipy.spatial import cKDTree

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ================================================================
#  配置
# ================================================================
DEFAULT_CFG = dict(
    strategy       = 'multiway',
    ref_frame      = None,
    ref_cands      = 3,
    ref_early_stop = 0.9,
    pair_exhaustive  = False,
    pair_order       = 'auto',
    pair_ring        = True,
    pair_max_edges   = 3,
    pair_max_trials  = 10,
    pair_stop_overlap= 0.55,
    pair_ring_step   = 3,
    voxel_fpfh     = 2.0,
    fpfh_radius    = 5.0,
    ransac_dist    = 1.0,
    ransac_iter    = 80000,
    pair_icp_thr   = [4.0, 2.0, 1.0],
    keep_ratio     = 0.6,
    icp_voxel      = 1.0,
    pair_high_precision = True,
    hp_voxel        = 0.2,
    hp_icp_thr      = [1.0, 0.5, 0.3],
    hp_keep_ratio   = 0.7,
    hp_iter         = 150,
    hp_norm_radius  = 1.2,
    hp_min_overlap  = 0.7,
    hp_min_resid    = 0.02,
    hp_enter_overlap = 0.30,
    edge_overlap_min = 0.25,
    edge_med_max     = 1.2,
    weak_med_max     = 1.0,
    weak_overlap_min = 0.3,
    mw_icp_thr       = [2.0, 1.0, 0.5],
    mw_keep_ratio    = 0.7,
    mw_batch_size    = 50,
    score_voxel    = 0.5,
    overlap_dist   = 1.0,
    merge_voxel    = 0.3,
    cover_radius   = 0.6,
    cover_min      = 3,
    shell_filter   = False,
    shell_radius   = 1.0,
    shell_gap      = 0.12,
    shell_ratio    = 0.35,
    rescue_rounds  = 99,
    rescue_accept_overlap = 0.4,
    rescue_accept_median  = 0.6,
    manual_rescue  = False,
    gate_enable    = True,
    gate_max_rounds= 3,
    gate_refine_thr= [1.0, 0.5, 0.3, 0.15],
    gate_mad_k     = 3.0,
    gate_min_overlap = 0.35,
    gate_block_size    = 15.0,
    gate_block_min_pts = 200,
    scan_model_path = '',
    scan_voxel      = 1.0,
    scan_accept_median = 0.5,
    # ---- v3.3 并行配准 ----
    n_jobs         = -1,        # 并行进程数: -1=CPU核数-1; 1=回退原顺序逻辑
    pair_window    = 8,         # 并行模式窗口: 只配 |i-j|<=此值 的帧对
    lazy_capacity  = 48,    # 懒加载缓存容量(帧数): 内存里最多同时保留 48 帧
    # ---- 不改变配准数学逻辑的运行稳定性选项 ----
    precheck_ply = True,            # 先验证每个PLY完整性；有坏帧则立即报错而不是静默丢弃
    worker_max_tasks = 24,         # 定期重启进程限制缓存内存增长
    worker_raw_capacity = 10,      # 每个worker最多保留多少帧原始点云
    max_auto_jobs = 8,             # 只限制自动进程数，手动 --jobs 不受影响
    pair_stall_timeout = 180.0,    # 连续无结果180s才回收整个进程池
    pair_retry_rounds = 1,         # 只重试异常/超时任务
    pair_retry_jobs = 3,           # 重试时降低并发
    pair_score_cache_capacity = 8, # score降采样缓存（不改变采样及目标点集）
    pair_tree_cache_capacity = 4,  # 原始target KDTree缓存
    reuse_pairs = False,           # 只有显式 --reuse-pairs 才读取已有结果
    snapshot_initial = False,      # --snapshot-initial 导出救援前多彩初始结果
    initial_only = False,           # --initial-only 不进入救援/守门，先评价原始自动主图
    source_style_main = True,       # 主图采用与待接入分量相同的组内 BFS 位姿
    source_seed_frame = None,       # 若指定，此帧所在分量作为主图
    source_seed_component = 0,      # 按分量大小排序的序号(0为最大)，优先级低于seed_frame
    source_skip_auto_bridge = True, # 此实验模式先不自动添加跨分量桥边，保留源分量的独立性
    pair_cache_path = '',          # 留空放在 multiview_output 中
    # ---- v9.3: 独立定量审计，默认不因诊断修改已经配好的位姿 ----
    metrics_enable = True,
    metrics_voxel = 0.5,           # 单位与PLY坐标一致（通常是mm）
    metrics_points_per_frame = 2500,  # 每帧诊断样本上限；只影响报告，不影响最终融合
    metrics_overlap_dist = 1.0,    # 统计重叠残差的最近邻距离阈值
    metrics_unit = 'mm',           # 须核实原始PLY的坐标单位
    metrics_edges_enable = True,
    metrics_triangles_enable = True,
    metrics_reference_ply = '',    # 可选：已在同一坐标系的参考PLY，绝不自动对齐
    gate_mode = 'audit',           # audit(只记录)/enforce(修正并剔除)/legacy(旧逻辑)/off
    freeze_seed_main = True,      # 救援后不重新优化初始主图
    manual_refine_max_shift = 1.0, # 手动初值后允许的最大位移，单位同 PLY
)

# ================================================================
#  通用几何工具
# ================================================================
def _ply_header_vertex_count(path):
    """检查文件头完整性，返回声明vertex数。兼容ASCII/binary PLY。"""
    with open(path, 'rb') as f:
        h = f.read(131072)
    m = h.find(b'end_header')
    if m < 0:
        raise RuntimeError('PLY lacks end_header: %s' % path)
    header = h[:m].decode('ascii', errors='replace')
    for line in header.splitlines():
        words = line.strip().split()
        if len(words) == 3 and words[0].lower() == 'element' and words[1].lower() == 'vertex':
            return int(words[2])
    raise RuntimeError('PLY lacks element vertex: %s' % path)


def safe_read_point_cloud(path):
    """先检头再读点，避免Open3D在RPLY错误时只返回残缺点云。"""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise OSError('PLY missing or empty: %s' % path)
    declared = _ply_header_vertex_count(path)
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    if declared <= 0 or len(points) != declared:
        raise RuntimeError('PLY truncated/incomplete: declared=%d read=%d file=%s'
                           % (declared, len(points), path))
    if not np.isfinite(points).all():
        raise RuntimeError('PLY contains NaN/Inf points: %s' % path)
    return pcd


def precheck_point_cloud_files(files):
    """不跳过文件：有坏文件就提前终止，并保存帧索引一致性。"""
    bad = []
    start = time.monotonic()
    print('  [precheck] 完整检查 %d 个 PLY 文件...' % len(files))
    for i, f in enumerate(files):
        try:
            p = safe_read_point_cloud(f)
            if len(p.points) < 10:
                raise ValueError('too few points for registration: %d' % len(p.points))
        except Exception as ex:
            bad.append((i, os.path.basename(f), str(ex)))
            print('    [bad-ply] index=%d name=%s reason=%s'
                  % (i, os.path.basename(f), str(ex)[:180]))
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print('    [precheck] %d/%d, errors=%d' % (i + 1, len(files), len(bad)))
    print('  [precheck] 完成 %.1fs，坏帧%d' % (time.monotonic() - start, len(bad)))
    return bad


class PairCache:
    """流式checkpoint，仅在文件+pair算法参数完全相同且显式允许时复用。"""
    KEYS = (
        'voxel_fpfh', 'fpfh_radius', 'ransac_dist', 'ransac_iter',
        'pair_icp_thr', 'keep_ratio', 'icp_voxel', 'pair_high_precision',
        'hp_voxel', 'hp_icp_thr', 'hp_keep_ratio', 'hp_iter',
        'hp_norm_radius', 'hp_min_overlap', 'hp_min_resid',
        'hp_enter_overlap', 'score_voxel', 'overlap_dist', 'pair_fast_init', 'random_seed',
    )

    def __init__(self, files, cfg):
        outdir = cfg.get('_outdir', '.')
        self.path = cfg.get('pair_cache_path') or os.path.join(outdir, 'pair_results_baseline.jsonl')
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        spec = {
            'version': 'v950_cuda64_bounded_scores_v1',
            'files': [
                [os.path.abspath(f), os.stat(f).st_size, os.stat(f).st_mtime_ns]
                for f in files],
            'settings': {k: cfg.get(k) for k in self.KEYS},
            'open3d_version': getattr(o3d, '__version__', 'unknown'),
        }
        self.signature = hashlib.sha256(
            json.dumps(spec, sort_keys=True, default=str).encode('utf-8')).hexdigest()
        self.saved = {}
        if cfg.get('reuse_pairs', False) and os.path.isfile(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    header = json.loads(f.readline())
                    if header.get('type') != 'header' or header.get('signature') != self.signature:
                        print('  [pair-cache] 输入文件或算法设置改变：旧缓存不可复用，重新计算')
                    else:
                        for line in f:
                            try:
                                d = json.loads(line)
                                if d.get('type') != 'pair':
                                    continue
                                i, j = int(d['i']), int(d['j'])
                                if d.get('status') == 'none':
                                    self.saved[(d.get('stage', 'pairwise'), i, j)] = None
                                elif d.get('status') == 'ok':
                                    T = np.asarray(d['T'], dtype=float).reshape(4, 4)
                                    self.saved[(d.get('stage', 'pairwise'), i, j)] = (
                                        T, float(d['med']), float(d['ov']), bool(d['consistent']))
                            except Exception:
                                # 若程序强制终止，忽略最后一行可能的不完整JSON。
                                continue
                        print('  [pair-cache] 读取 %d 个已完成pair' % len(self.saved))
            except Exception as ex:
                print('  [pair-cache] 读取失败，重新计算: %s' % ex)
                self.saved = {}

        # 所有新运行都重新写入自描述的缓存头+已复用的结果：支持被中断后再恢复。
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'type': 'header', 'signature': self.signature}, ensure_ascii=False) + '\n')
            for (stage, i, j), res in sorted(self.saved.items()):
                f.write(json.dumps(self._record(stage, i, j, res), ensure_ascii=False) + '\n')
        os.replace(tmp, self.path)
        self.fp = open(self.path, 'a', encoding='utf-8', buffering=1)

    @staticmethod
    def _record(stage, i, j, res):
        base = {'type': 'pair', 'stage': stage, 'i': int(i), 'j': int(j)}
        if res is None:
            base['status'] = 'none'
        else:
            T, med, ov, consistent = res
            base.update(status='ok', T=np.asarray(T, float).reshape(16).tolist(),
                        med=float(med), ov=float(ov), consistent=bool(consistent))
        return base

    def store(self, stage, i, j, res):
        key = (stage, i, j)
        if key in self.saved:
            return
        self.saved[key] = res
        self.fp.write(json.dumps(self._record(stage, i, j, res), ensure_ascii=False) + '\n')

    def close(self):
        if hasattr(self, 'fp') and not self.fp.closed:
            self.fp.close()


def run_pairs_safe(files, cfg, task_list, on_result, stage, jobs, stop_on_success=False):
    """稳定的imap_unordered封装；worker返回pair+result，不能产生旧版解包冲突。

    若连续一段时间没有任务完成，终止整个Pool，并只重试未完成任务。
    注意这不是单任务硬超时，经过多少秒才终止取决于 pool 的整体进度。
    """
    import multiprocessing as mp
    tasks = list(dict.fromkeys((int(i), int(j)) for i, j in task_list))
    if not tasks:
        return False, []
    n_retry = max(0, int(cfg.get('pair_retry_rounds', 1)))
    stall = float(cfg.get('pair_stall_timeout', 180.0))
    maxjobs = max(1, int(jobs))
    all_done = 0
    last_log = time.monotonic()

    for attempt in range(n_retry + 1):
        if not tasks:
            return False, []
        nj = maxjobs if attempt == 0 else min(maxjobs, max(1, int(cfg.get('pair_retry_jobs', 3))))
        print('  [%s] 第%d轮 %d对, %d进程, 连续无进度超时%.0fs'
              % (stage, attempt + 1, len(tasks), nj, stall))
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=min(nj, len(tasks)), initializer=_pw_init,
                        initargs=(files, cfg),
                        maxtasksperchild=(int(cfg.get('worker_max_tasks', 0)) or None))
        pending = set(tasks)
        errors = set()
        stalled = False
        early_stop = False
        last_result = time.monotonic()
        try:
            it = pool.imap_unordered(_pw_work, tasks, chunksize=1)
            while pending:
                try:
                    i, j, res, error = it.next(timeout=min(10.0, stall))
                except mp.TimeoutError:
                    if time.monotonic() - last_result >= stall:
                        stalled = True
                        print('    [%s] %.0fs未返回新结果，回收Pool，待重试=%d'
                              % (stage, stall, len(pending)))
                        break
                    continue
                except StopIteration:
                    break
                last_result = time.monotonic()
                key = (int(i), int(j))
                if key not in pending:
                    continue
                pending.remove(key)
                all_done += 1
                if error:
                    print('    [pair-error] (%d,%d) %s | %s: %s'
                          % (i, j, os.path.basename(files[i]), os.path.basename(files[j]), error[:180]))
                    errors.add(key)
                else:
                    if on_result(i, j, res):
                        early_stop = True
                        break
                if (all_done % 50 == 0) or (not pending):
                    print('    [%s] 已完成 %d，当前轮剩余 %d'
                          % (stage, all_done, len(pending)))
        finally:
            if stalled or early_stop or pending:
                pool.terminate()
            else:
                pool.close()
            pool.join()
        if early_stop:
            return True, []
        tasks = sorted(pending | errors)
        if tasks and attempt < n_retry:
            print('  [%s] 将重试 %d 对，低并发重新建立worker' % (stage, len(tasks)))
    if tasks:
        raise RuntimeError('[%s] %d pair tasks failed after retries; checkpoint saved; no incomplete result exported: %s' % (stage, len(tasks), tasks[:20]))
    return False, tasks


def voxel_down(pcd, vx):
    return pcd.voxel_down_sample(vx)

def fpfh_feats(pcd, voxel, radius):
    d = voxel_down(pcd, voxel)
    if len(d.points) < 10:
        raise RuntimeError('Cannot compute FPFH: downsampled cloud has <10 points')
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    if not d.has_normals() or len(d.normals) != len(d.points):
        raise RuntimeError('Cannot compute FPFH: normals missing')
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    return d, f

def ransac_init(src_d, src_f, tgt_d, tgt_f, dist, n_iter):
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f, True, dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(n_iter, 0.999))
    return np.asarray(res.transformation)

def kabsch(P, Q, w=None):
    if w is None:
        w = np.ones(len(P))
    w = w / w.sum()
    cp = (P * w[:, None]).sum(0); cq = (Q * w[:, None]).sum(0)
    X = P - cp; Y = Q - cq
    H = (X * w[:, None]).T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    t = cq - R @ cp
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T

def p2plane_step(P, Q, Qn, keep_ratio=1.0, huber=1.345):
    r = ((P - Q) * Qn).sum(1)
    if keep_ratio < 1.0:
        k = max(3, int(len(r) * keep_ratio))
        idx = np.argsort(np.abs(r))[:k]
        P, Q, Qn, r = P[idx], Q[idx], Qn[idx], r[idx]
    a = np.abs(r).mean()
    if a < 1e-12:
        return np.eye(4)
    s = huber * a
    w = np.ones(len(r)); big = np.abs(r) > s
    w[big] = s / np.abs(r[big])
    A = np.zeros((len(P), 6))
    A[:, :3] = np.cross(P, Qn)
    A[:, 3:] = Qn
    sw = np.sqrt(w)
    Aw = A * sw[:, None]
    bw = r * sw
    H = Aw.T @ Aw
    b = -Aw.T @ bw
    try:
        dx = np.linalg.solve(H + 1e-9 * np.eye(6), b)
    except np.linalg.LinAlgError:
        return np.eye(4)
    dR = o3d.geometry.get_rotation_matrix_from_xyz(dx[:3])
    T = np.eye(4); T[:3, :3] = dR; T[:3, 3] = dx[3:]
    return T

def icp_custom(src, tgt, thr, keep_ratio=0.6, max_iter=60, conv=1e-4, T_init=None):
    src_pts = np.asarray(src.points)
    tgt_pts = np.asarray(tgt.points)
    if not tgt.has_normals():
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=thr * 2, max_nn=30))
    tgt_n = np.asarray(tgt.normals)
    tree = cKDTree(tgt_pts)
    T = np.eye(4) if T_init is None else T_init.copy()
    prev = np.inf
    for _ in range(max_iter):
        P = (T[:3, :3] @ src_pts.T).T + T[:3, 3]
        dist, idx = tree.query(P, k=1)
        Q = tgt_pts[idx]
        if thr <= 2.0:
            if not src.has_normals():
                src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=thr * 2, max_nn=30))
            src_n = (T[:3, :3] @ np.asarray(src.normals).T).T
            cosang = np.abs((src_n * tgt_n[idx]).sum(1))
            m = (dist < thr) & (cosang > 0.5)
        else:
            m = dist < thr
        if m.sum() < 10:
            break
        d_m = dist[m]
        k = max(10, int(len(d_m) * keep_ratio))
        cutoff = np.partition(d_m, k - 1)[k - 1]
        mm = m.copy(); mm[m] &= dist[m] <= cutoff
        Pk, Qk, Qnk = P[mm], Q[mm], tgt_n[idx[mm]]
        dT = p2plane_step(Pk, Qk, Qnk, keep_ratio=1.0)
        T = dT @ T
        cur = np.abs(((Pk - Qk) * Qnk).sum(1)).mean()
        if abs(prev - cur) < conv:
            break
        prev = cur
    return T, prev

def icp_pp(src, tgt, thr, keep_ratio=0.8, max_iter=80, conv=1e-4, T_init=None):
    src_pts = np.asarray(src.points)
    tgt_pts = np.asarray(tgt.points)
    tree = cKDTree(tgt_pts)
    T = np.eye(4) if T_init is None else T_init.copy()
    prev = np.inf
    for _ in range(max_iter):
        P = (T[:3, :3] @ src_pts.T).T + T[:3, 3]
        dist, idx = tree.query(P, k=1, distance_upper_bound=thr, eps=0.)
        m = dist < thr
        if m.sum() < 10:
            break
        d_m = dist[m]
        k = max(10, int(len(d_m) * keep_ratio))
        cutoff = np.partition(d_m, k - 1)[k - 1]
        mm = m.copy(); mm[m] &= dist[m] <= cutoff
        dT = kabsch(P[mm], tgt_pts[idx[mm]])
        T = dT @ T
        cur = d_m[d_m <= cutoff].mean()
        if abs(prev - cur) < conv:
            break
        prev = cur
    return T

def evaluate_pair(src, tgt, T, voxel=0.5, dist_thr=1.0):
    s = src.voxel_down_sample(voxel)
    P = (T[:3, :3] @ np.asarray(s.points).T).T + T[:3, 3]
    d, _ = cKDTree(np.asarray(tgt.points)).query(P, k=1)
    return float(np.median(d)), float((d < dist_thr).mean())

# ================================================================
#  模块1: 两两配准(强边 + 弱边缓存 + 前置过滤 + 多进程并行)
# ================================================================
@dataclass
class PairEdge:
    i: int
    j: int
    T_ij: np.ndarray
    information: np.ndarray
    median: float
    overlap: float
    valid: bool = True
    consistent: bool = True

# ---------------- 并行 worker(模块级, 供子进程调用) ----------------
_PW = {}

def _pw_init(files, cfg):
    """子进程按需读盘。帧缓存LRU + 定期重启以限制长期内存占用。"""
    global _PW
    _PW = {'files': files, 'cfg': cfg, 'raws': OrderedDict(), 'reg': None}


def _pw_raw(k):
    cache = _PW['raws']
    if k in cache:
        cache.move_to_end(k)
        return cache[k]
    p = safe_read_point_cloud(_PW['files'][k])
    cache[k] = p
    capacity = max(2, int(_PW['cfg'].get('worker_raw_capacity', 10)))
    if len(cache) > capacity:
        cache.popitem(last=False)
    return p

class LazyFrameLoader:
    """伪装成 raws 列表: __getitem__ 按需读盘 + LRU 缓存, 内存只保留最近 capacity 帧。
    对下游所有模块透明(它们只用 raws[k] / len(raws))。"""
    def __init__(self, files, capacity=48):
        self.files = files
        self.capacity = capacity
        self._cache = {}          # k -> pcd
        self._order = deque()     # LRU 顺序

    def __getitem__(self, k):
        if k in self._cache:
            self._order.remove(k); self._order.append(k)
            return self._cache[k]
        pcd = safe_read_point_cloud(self.files[k])
        self._cache[k] = pcd
        self._order.append(k)
        if len(self._order) > self.capacity:
            evict = self._order.popleft()
            del self._cache[evict]
        return pcd

    def __len__(self):
        return len(self.files)

class _LazyRaws:
    """伪装成 raws 列表: __getitem__ 走子进程缓存"""
    def __getitem__(self, k):
        return _pw_raw(k)
    def __len__(self):
        return len(_PW['files'])

def _pw_work(pair):
    """统一返回 (i,j,配准四元组或None,错误信息或None)。"""
    i, j = pair
    try:
        if _PW['reg'] is None:
            _PW['reg'] = PairwiseRegistrator(_PW['cfg'])
        res = _PW['reg']._align_pair(_LazyRaws(), i, j)
        return (i, j, res, None)
    except Exception as exc:
        return (i, j, None, '%s: %s' % (type(exc).__name__, str(exc)))

class PairwiseRegistrator:
    def __init__(self, cfg):
        self.cfg = cfg
        self._feat = OrderedDict()
        self._icp = OrderedDict()
        self._hp = OrderedDict()
        self._score = OrderedDict()
        self._target_trees = OrderedDict()
        self._files = None
        self._pair_cache = None
        self._edge_stage = {}

    def _feats(self, raws, k):
        if k not in self._feat:
            self._feat[k] = fpfh_feats(raws[k], self.cfg['voxel_fpfh'], self.cfg['fpfh_radius'])
        return self._feat[k]

    def _icp_cloud(self, raws, k):
        if k not in self._icp:
            self._icp[k] = voxel_down(raws[k], self.cfg['icp_voxel'])
        return self._icp[k]

    def _icp_cloud_hp(self, raws, k):
        if k not in self._hp:
            self._hp[k] = voxel_down(raws[k], self.cfg['hp_voxel'])
        return self._hp[k]

    def _evaluate_cached(self, raws, i, j, T, voxel, dist_thr):
        """等价于evaluate_pair(): source同voxel采样，target仍是原始点云。
        只复用本进程的score点云与target KDTree；不改变查询点集和阈值。
        """
        # 0.2高精评分仍调用原函数，避免无意义缓存不同分辨率。
        if voxel != self.cfg['score_voxel']:
            return evaluate_pair(raws[i], raws[j], T, voxel, dist_thr)
        if i not in self._score:
            self._score[i] = raws[i].voxel_down_sample(voxel)
        self._score.move_to_end(i)
        while len(self._score) > int(self.cfg.get('pair_score_cache_capacity', 8)):
            self._score.popitem(last=False)
        if j not in self._target_trees:
            self._target_trees[j] = cKDTree(np.asarray(raws[j].points))
        self._target_trees.move_to_end(j)
        while len(self._target_trees) > int(self.cfg.get('pair_tree_cache_capacity', 4)):
            self._target_trees.popitem(last=False)
        pts = np.asarray(self._score[i].points)
        P = (T[:3, :3] @ pts.T).T + T[:3, 3]
        d, _ = self._target_trees[j].query(P, k=1)
        return float(np.median(d)), float((d < dist_thr).mean())

    def _align_pair(self, raws, i, j):
        cfg = self.cfg
        sd, sf = self._feats(raws, i)
        td, tf = self._feats(raws, j)
        try:
            T0 = _pair_initial_transform(self, raws, i, j, sd, sf, td, tf)
        except Exception:
            return None
        med0, ov0 = self._evaluate_cached(raws, i, j, T0, cfg['score_voxel'], cfg['overlap_dist'])
        if ov0 < 0.05:
            return None
        med, ov = med0, ov0
        if not cfg.get('pair_high_precision', True):
            T = T0
            for thr in cfg['pair_icp_thr']:
                T, _ = icp_custom(self._icp_cloud(raws, i), self._icp_cloud(raws, j),
                                  thr, cfg['keep_ratio'], T_init=T)
            med, ov = self._evaluate_cached(raws, i, j, T, cfg['score_voxel'], cfg['overlap_dist'])
            med_r, ov_r = self._evaluate_cached(raws, j, i, np.linalg.inv(T),
                                        cfg['score_voxel'], cfg['overlap_dist'])
            consistent = (min(ov, ov_r) > 0.35) or (max(ov, ov_r) > 0.6 and min(ov, ov_r) > 0.25)
            return T, med, ov, consistent
        if ov < cfg.get('hp_enter_overlap', 0.30) or med > 2.0:
            return T0, med, ov, True
        s_mid = self._icp_cloud(raws, i); t_mid = self._icp_cloud(raws, j)
        T, _ = icp_custom(s_mid, t_mid, 2.0, 0.8, max_iter=80, T_init=T0)
        med_mid, ov_mid = self._evaluate_cached(raws, i, j, T, cfg['score_voxel'], cfg['overlap_dist'])
        if ov_mid < cfg.get('hp_enter_overlap', 0.30):
            return T, med_mid, ov_mid, True
        s_hp = self._icp_cloud_hp(raws, i); t_hp = self._icp_cloud_hp(raws, j)
        nr = cfg['hp_norm_radius']
        if not s_hp.has_normals():
            s_hp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=nr, max_nn=50))
        if not t_hp.has_normals():
            t_hp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=nr, max_nn=50))
        for thr in cfg['hp_icp_thr']:
            T, _ = icp_custom(s_hp, t_hp, thr, cfg['hp_keep_ratio'],
                              max_iter=cfg['hp_iter'], conv=5e-5, T_init=T)
        med, ov = _bounded_hp_score(s_hp, t_hp, T, cfg)
        if ov < cfg['hp_min_overlap']:
            med_lo, ov_lo = self._evaluate_cached(raws, i, j, T, cfg['score_voxel'], cfg['overlap_dist'])
            return T, med_lo, ov_lo, False
        med_r, ov_r = self._evaluate_cached(raws, j, i, np.linalg.inv(T),
                                    cfg['score_voxel'], cfg['overlap_dist'])
        consistent = (min(ov, ov_r) > 0.35) or (max(ov, ov_r) > 0.6 and min(ov, ov_r) > 0.25)
        return T, med, ov, consistent

    @staticmethod
    def _aabb_stats(raws):
        centers, radii = [], []
        for r in raws:
            aabb = r.get_axis_aligned_bounding_box()
            centers.append(np.asarray(aabb.get_center()))
            radii.append(np.linalg.norm(np.asarray(aabb.get_max_bound())
                                        - np.asarray(aabb.get_min_bound())) / 2)
        return np.array(centers), np.array(radii)

    def _make_edge(self, i, j, T, med, ov, consistent):
        cfg = self.cfg
        valid = (ov >= cfg['edge_overlap_min']) and (med <= cfg['edge_med_max']) and consistent
        w = max(ov, 1e-3) / max(med, 1e-3)
        info = np.eye(6) * min(w, 400.0)
        return PairEdge(i, j, T, info, med, ov, valid, consistent)

    # ---------------- 并行版主入口 ----------------
    def _register_parallel(self, raws, files, centers, radii):
        """原版相同候选集；只替换worker管理+缓存/日志接口，不改pair算法。"""
        cfg = self.cfg
        n = len(raws)
        ring = bool(cfg.get('pair_ring', True))
        window = cfg.get('pair_window', 8)
        ring_step = cfg.get('pair_ring_step', 0)
        n_jobs = cfg.get('n_jobs', -1)
        if n_jobs <= 0:
            n_jobs = max(1, min(int(cfg.get('max_auto_jobs', 8)), (os.cpu_count() or 2) - 1))

        def idist(i, j):
            d = abs(i - j)
            return min(d, n - d) if ring else d

        def prescreen(i, j):
            return np.linalg.norm(centers[i] - centers[j]) <= 1.5 * (radii[i] + radii[j])

        # 原始任务集合表达式不变。
        tasks = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = idist(i, j)
                in_window = d <= window
                in_step = ring_step > 0 and d <= ring_step + 1
                if (in_window or in_step) and prescreen(i, j):
                    tasks.add((i, j))
        tasks = sorted(tasks)
        print('  [pairwise-并行] 任务对 %d (窗口<=%d + 跨帧<=%d), %d 进程...'
              % (len(tasks), window, ring_step, n_jobs))

        edges, weak_edges = [], []
        t0 = time.time()
        self._pair_cache = PairCache(files, cfg)

        def accept(i, j, res, stage):
            self._pair_cache.store('pairwise', i, j, res)
            if res is None:
                return False
            T, med, ov, consistent = res
            e = self._make_edge(i, j, T, med, ov, consistent)
            edges.append(e)
            self._edge_stage[id(e)] = stage
            if (not e.valid and e.consistent
                    and e.median < cfg['weak_med_max']
                    and e.overlap > cfg['weak_overlap_min']):
                weak_edges.append(e)
            return False

        try:
            cached = [p for p in tasks if ('pairwise', p[0], p[1]) in self._pair_cache.saved]
            for i, j in cached:
                accept(i, j, self._pair_cache.saved[('pairwise', i, j)], 'pairwise-cache')
            todo = [p for p in tasks if ('pairwise', p[0], p[1]) not in self._pair_cache.saved]
            print('  [pairwise] 缓存 %d 对；本次新算 %d 对' % (len(cached), len(todo)))
            if todo:
                run_pairs_safe(files, cfg, todo,
                               lambda i, j, res: accept(i, j, res, 'pairwise'),
                               'pairwise', n_jobs)
            edges.sort(key=lambda e: (e.i, e.j))
            print('  [pairwise] 有效边 %d，弱边 %d，耗时 %.0fs'
                  % (sum(e.valid for e in edges), len(weak_edges), time.time() - t0))
            if cfg.get('source_style_main', False) and cfg.get('source_skip_auto_bridge', True):
                print('  [source-main] 跳过自动跨分量架桥，以便先测试局部源分量质量')
            else:
                edges = self._bridge_components(raws, edges, centers, radii)
            self._export_audit(edges, files)
        finally:
            self._pair_cache.close()
        return edges, weak_edges

    def _export_audit(self, edges, files):
        outdir = self.cfg.get('_outdir', '.')
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, 'pairwise_edge_audit.csv')
        with open(path, 'w', encoding='utf-8-sig', newline='') as fp:
            w = csv.writer(fp)
            w.writerow(['i', 'j', 'file_i', 'file_j', 'median', 'overlap',
                        'consistent', 'valid', 'source_stage', 'T_ij_rowmajor'])
            for e in sorted(edges, key=lambda x: (x.i, x.j)):
                w.writerow([e.i, e.j, os.path.basename(files[e.i]),
                            os.path.basename(files[e.j]), e.median, e.overlap,
                            int(e.consistent), int(e.valid),
                            self._edge_stage.get(id(e), 'unknown'),
                            json.dumps(np.asarray(e.T_ij).reshape(16).tolist())])
        print('  [pairwise] 审计CSV: %s' % path)

    # ---------------- 原顺序版(n_jobs=1 时回退, 行为同 v3.2) ----------------
    def _register_sequential(self, raws, centers, radii):
        cfg = self.cfg
        n = len(raws)
        cd = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2) if n > 1 else np.zeros((1, 1))
        order_mode = cfg.get('pair_order', 'auto')
        ring = bool(cfg.get('pair_ring', True))
        if order_mode == 'auto':
            spread = (cd.max() - cd.min()) / (radii.mean() + 1e-9) if n > 1 else 0.0
            order_mode = 'index' if spread < 0.5 else 'dist'
        exhaustive = bool(cfg.get('pair_exhaustive', False))
        max_edges, max_trials = cfg['pair_max_edges'], cfg['pair_max_trials']
        stop_ov = cfg['pair_stop_overlap']

        def idist(i, j):
            d = abs(i - j)
            return min(d, n - d) if ring else d

        def satisfied(k, edge_cnt, best_ov):
            return edge_cnt[k] >= max_edges or (edge_cnt[k] >= 2 and best_ov[k] >= stop_ov)

        print('  [pairwise] %d 帧, 候选排序=%s%s, 早停%s(>=2边且重叠>=%.0f%%, %d边封顶, 单帧<=%d次)...'
              % (n, order_mode, '(环形)' if (order_mode == 'index' and ring) else '',
                 '关' if exhaustive else '开', 100 * stop_ov, max_edges, max_trials))

        def prescreen(i, j):
            return cd[i, j] <= 1.5 * (radii[i] + radii[j])

        edges, weak_edges, have = [], [], set()
        edge_cnt = np.zeros(n, int)
        best_ov = np.zeros(n)
        trials = 0
        t0 = time.time()
        for i in range(n):
            if not exhaustive and satisfied(i, edge_cnt, best_ov):
                continue
            ring_step = cfg.get('pair_ring_step', 0)
            if order_mode == 'index' and ring_step > 0:
                near = [j for j in range(n)
                        if j != i and (min(i, j), max(i, j)) not in have and prescreen(i, j)]
                step_set = set()
                for s in range(2, ring_step + 1):
                    step_set.add((i + s) % n)
                    step_set.add((i - s) % n)
                step_cands = [j for j in near if j in step_set]
                rest = [j for j in near if j not in step_set]
                cands = step_cands + rest
            else:
                cands = [j for j in range(n)
                         if j != i and (min(i, j), max(i, j)) not in have and prescreen(i, j)]
            cands.sort(key=lambda j: idist(i, j) if order_mode == 'index' else cd[i, j])
            tried = 0
            for j in cands:
                if not exhaustive:
                    if satisfied(i, edge_cnt, best_ov) or tried >= max_trials:
                        break
                    if satisfied(j, edge_cnt, best_ov) and edge_cnt[i] > 0:
                        continue
                tried += 1; trials += 1
                have.add((min(i, j), max(i, j)))
                res = self._align_pair(raws, i, j)
                if res is None:
                    continue
                e = self._make_edge(i, j, *res)
                edges.append(e)
                if (not e.valid and e.consistent
                        and e.median < cfg['weak_med_max'] and e.overlap > cfg['weak_overlap_min']):
                    weak_edges.append(e)
                    print('    对(%2d,%2d): 中位 %.3f, 重叠 %.0f%% [弱边缓存]'
                          % (i, j, e.median, 100 * e.overlap))
                else:
                    tag = 'OK' if e.valid else ('假阳性' if not e.consistent else 'weak')
                    print('    对(%2d,%2d): 中位 %.3f, 重叠 %.0f%% %s'
                          % (i, j, e.median, 100 * e.overlap, tag))
                if e.valid:
                    edge_cnt[i] += 1; edge_cnt[j] += 1
                    best_ov[i] = max(best_ov[i], e.overlap)
                    best_ov[j] = max(best_ov[j], e.overlap)
        print('  [pairwise] 尝试 %d 对(全量 %d), 有效边 %d, 弱边 %d, 耗时 %.0fs'
              % (trials, n * (n - 1) // 2, sum(e.valid for e in edges), len(weak_edges), time.time() - t0))
        if cfg.get('source_style_main', False) and cfg.get('source_skip_auto_bridge', True):
            print('  [source-main] 顺序模式跳过自动跨分量架桥')
        else:
            edges = self._bridge_components(raws, edges, centers, radii)
        return edges, weak_edges

    def register_all_pairs(self, raws, files=None):
        cfg = self.cfg
        centers, radii = self._aabb_stats(raws)
        n_jobs = cfg.get('n_jobs', -1)
        if n_jobs == 1 or files is None:
            if n_jobs == 1:
                print('  [pairwise] n_jobs=1, 使用顺序模式')
            return self._register_sequential(raws, centers, radii)
        self._files = files          # ★ 这行必须在 return 之前
        return self._register_parallel(raws, files, centers, radii)

    def _bridge_components(self, raws, edges, centers, radii):
        """原版分量/候选/成功判定；并行worker统一接口+安全回收。"""
        cfg = self.cfg
        n = len(raws)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        for e in edges:
            if e.valid:
                ra, rb = find(e.i), find(e.j)
                if ra != rb:
                    parent[ra] = rb
        comp = {}
        for k in range(n):
            comp.setdefault(find(k), []).append(k)
        comps = sorted(comp.values(), key=len, reverse=True)
        if len(comps) <= 1:
            return edges
        print('  [pairwise] 图被切成 %d 个分量(大小 %s), 尝试架桥(并行)...'
              % (len(comps), [len(c) for c in comps]))

        main = list(comps[0])
        n_jobs = cfg.get('n_jobs', -1)
        if n_jobs <= 0:
            n_jobs = max(1, min(int(cfg.get('max_auto_jobs', 8)), (os.cpu_count() or 2) - 1))

        for other in comps[1:]:
            pairs = [(i, j) for i in main for j in other
                     if np.linalg.norm(centers[i] - centers[j]) <= 2.0 * (radii[i] + radii[j])]
            pairs.sort(key=lambda ij: (min(abs(ij[0] - ij[1]), n - abs(ij[0] - ij[1])),
                                       np.linalg.norm(centers[ij[0]] - centers[ij[1]])))
            tasks = pairs[:cfg['pair_max_trials'] * 2]
            if not tasks:
                print('    分量 %s 无候选对, 跳过' % other)
                continue
            bridged = False

            def accept_bridge(i, j, res, cache_state=False):
                nonlocal bridged
                if self._pair_cache is not None:
                    self._pair_cache.store('bridge', i, j, res)
                if res is None:
                    return False
                e = self._make_edge(i, j, *res)
                edges.append(e)
                self._edge_stage[id(e)] = 'bridge-cache' if cache_state else 'bridge'
                if e.valid:
                    print('    架桥成功: (%d,%d) 中位 %.3f, 重叠 %.0f%%'
                          % (i, j, e.median, 100 * e.overlap))
                    bridged = True
                    return True
                return False

            if n_jobs > 1 and self._files is not None:
                # 缓存命中的pair提前尝试（仅显式 --reuse-pairs 读取旧缓存）。
                pending = []
                for i, j in tasks:
                    if self._pair_cache is not None and ('bridge', i, j) in self._pair_cache.saved:
                        if accept_bridge(i, j, self._pair_cache.saved[('bridge', i, j)], True):
                            break
                    else:
                        pending.append((i, j))
                if not bridged:
                    run_pairs_safe(self._files, cfg, pending, accept_bridge,
                                   'bridge', min(n_jobs, len(pending)) if pending else 1,
                                   stop_on_success=True)
            else:
                # 保留原顺序模式的完整注册算法。
                for i, j in tasks:
                    try:
                        res = self._align_pair(raws, i, j)
                    except Exception as exc:
                        print('    [bridge-error] (%d,%d): %s' % (i, j, exc))
                        continue
                    if res is None:
                        continue
                    e = self._make_edge(i, j, *res)
                    edges.append(e)
                    self._edge_stage[id(e)] = 'bridge-sequential'
                    if e.valid:
                        print('    架桥成功: (%d,%d) 中位 %.3f, 重叠 %.0f%%'
                              % (i, j, e.median, 100 * e.overlap))
                        bridged = True
                        break

            if bridged:
                main = main + list(other)
                # 原版并查集重建语义不变。
                parent = list(range(n))
                for e in edges:
                    if e.valid:
                        ra, rb = find(e.i), find(e.j)
                        if ra != rb:
                            parent[ra] = rb
            else:
                print('    分量 %s 架桥失败, 稍后走分量级抢救' % other)
        return edges

# ================================================================
#  模块2: 全局优化(分批 multiway)
# ================================================================
class GlobalOptimizer:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def pick_anchor(n, edges):
        deg = np.zeros(n, int)
        for e in edges:
            if e.valid:
                deg[e.i] += 1; deg[e.j] += 1
        return int(np.argmax(deg)) if deg.sum() else 0

    @staticmethod
    def _edge_residual(T_edge, P_i, P_j):
        D = np.linalg.inv(T_edge) @ (np.linalg.inv(P_j) @ P_i)
        return float(np.linalg.norm(D - np.eye(4)))

    def _graph_error(self, edges, idx_of, pose_graph):
        P = [np.asarray(nd.pose) for nd in pose_graph.nodes]
        res = [self._edge_residual(e.T_ij, P[idx_of[e.i]], P[idx_of[e.j]])
               for e in edges if e.valid and e.i in idx_of and e.j in idx_of]
        return float(np.mean(res)) if res else np.inf

    def build_pose_graph(self, n, edges):
        pose_graph = o3d.pipelines.registration.PoseGraph()
        anchor = self.pick_anchor(n, edges)
        T_world = {anchor: np.eye(4)}
        adj = {k: [] for k in range(n)}
        for e in edges:
            if e.valid:
                adj[e.i].append((e.j, e.T_ij))
                adj[e.j].append((e.i, np.linalg.inv(e.T_ij)))
        dq = deque([anchor])
        while dq:
            u = dq.popleft()
            for v, T_uv in adj[u]:
                if v not in T_world:
                    T_world[v] = T_world[u] @ np.linalg.inv(T_uv)
                    dq.append(v)
        reachable = sorted(T_world.keys())
        idx_of = {old: new for new, old in enumerate(reachable)}
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for e in edges:
            if e.valid:
                a, b = find(e.i), find(e.j)
                if a != b:
                    parent[a] = b
        comp = {}
        for k in range(n):
            comp.setdefault(find(k), []).append(k)
        if len(comp) > 1:
            sizes = sorted(comp.values(), key=len, reverse=True)
            print('  [global] 注意: %d 个连通分量(大小 %s)' % (len(comp), [len(s) for s in sizes]))
            for members in sizes[1:]:
                print('    分量 %s 与主图断开, 暂落选(后续分量级抢救)' % members)
        for k in reachable:
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T_world[k]))
        for e in edges:
            if not e.valid or e.i not in idx_of or e.j not in idx_of:
                continue
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                idx_of[e.i], idx_of[e.j], e.T_ij, e.information, uncertain=False))
        return pose_graph, T_world, reachable, anchor, idx_of

    def optimize(self, n, edges, raws):
        cfg = self.cfg
        pose_graph, T_world, reachable, anchor, idx_of = self.build_pose_graph(n, edges)
        if len(reachable) < 2:
            print('  [global] 有效连通帧不足, 回退')
            return None, reachable
        print('  [global] 位姿图: %d 节点 %d 边, 锚点=帧%d'
              % (len(pose_graph.nodes), len(pose_graph.edges), anchor))
        err0 = self._graph_error(edges, idx_of, pose_graph)
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=min(cfg['overlap_dist'], 0.5),
            edge_prune_threshold=0.25,
            reference_node=0,
            preference_loop_closure=0.0)
        crit = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        o3d.pipelines.registration.global_optimization(pose_graph, method, crit, option)
        err1 = self._graph_error(edges, idx_of, pose_graph)
        print('  [global] 边一致性残差: 优化前 %.4f -> 优化后 %.4f' % (err0, err1))
        if err1 > err0 * 1.5:
            print('  [global] 警告: 全局优化使一致性变差, 回退 BFS 初始位姿')
            Ms = {k: T_world[k] for k in reachable}
        else:
            inv_idx = {v: k for k, v in idx_of.items()}
            Ms = {inv_idx[nk]: np.asarray(nd.pose) for nk, nd in enumerate(pose_graph.nodes)}
        print('  [global] multiway 联合 ICP 抛光(%d 帧位姿同时优化)...' % len(reachable))
        Ms = self._multiway_icp(raws, reachable, Ms)
        return Ms, reachable

    def _multiway_icp(self, raws, use, Ms):
        cfg = self.cfg
        use = list(use)
        batch_size = cfg.get('mw_batch_size', 50)
        for thr in cfg['mw_icp_thr']:
            for _ in range(3):
                moved = 0.0
                for batch_start in range(0, len(use), batch_size):
                    batch = use[batch_start:batch_start + batch_size]
                    world = {}
                    for k in batch:
                        p = _score_cloud(raws, k, cfg); p.transform(Ms[k])
                        world[k] = np.asarray(p.points)
                    for k in batch:
                        others = np.vstack([world[j] for j in batch if j != k])
                        mo = o3d.geometry.PointCloud()
                        mo.points = o3d.utility.Vector3dVector(others)
                        mo = mo.voxel_down_sample(cfg['score_voxel'])
                        src = voxel_down(raws[k], cfg['score_voxel'])
                        T, _ = icp_custom(src, mo, thr, cfg['mw_keep_ratio'],
                                          max_iter=40, T_init=Ms[k])
                        moved += float(np.linalg.norm(T[:3, 3] - Ms[k][:3, 3])
                                       + np.linalg.norm(T[:3, :3] - Ms[k][:3, :3]))
                        Ms[k] = T
                        world[k] = (T[:3, :3] @ np.asarray(src.points).T).T + T[:3, 3]
                if moved < 1e-4:
                    break
        return Ms

# ================================================================
#  模块2.5: 模型锚定配准
# ================================================================
class ScanAnchorRegistrator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scan_pcd = None
        self.scan_down = None
        self.T_main_to_scan = None

    def load_scan(self, path):
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        self.scan_pcd = o3d.geometry.PointCloud()
        self.scan_pcd.points = mesh.vertices
        self.scan_pcd.normals = mesh.vertex_normals
        self.scan_down = voxel_down(self.scan_pcd, self.cfg['scan_voxel'])
        print('  [锚定] 口扫模型: %d 顶点, 降采样 %d 点'
              % (len(self.scan_pcd.points), len(self.scan_down.points)))
        return True

    def register_to_scan(self, src_pcd, T_init=None, tag=''):
        cfg = self.cfg
        cands = []
        if T_init is not None:
            cands.append(T_init)
        else:
            f_down, f_feat = fpfh_feats(src_pcd, cfg['voxel_fpfh'], cfg['fpfh_radius'])
            s_down, s_feat = fpfh_feats(self.scan_pcd, cfg['voxel_fpfh'], cfg['fpfh_radius'])
            try:
                cands.append(ransac_init(f_down, f_feat, s_down, s_feat,
                                         cfg['ransac_dist'], cfg['ransac_iter']))
            except Exception:
                return None
        best = None
        for Tc in cands:
            T = icp_pp(voxel_down(src_pcd, cfg['icp_voxel']), self.scan_down,
                       8.0, keep_ratio=0.9, max_iter=40, T_init=Tc)
            _, ov_c = evaluate_pair(src_pcd, self.scan_pcd, T,
                                    cfg['score_voxel'], cfg['overlap_dist'])
            if best is None or ov_c > best[1]:
                best = (T, ov_c)
        T = best[0]
        for thr in [4.0, 2.0, 1.0]:
            T, _ = icp_custom(voxel_down(src_pcd, cfg['icp_voxel']),
                              self.scan_down, thr, cfg['keep_ratio'], T_init=T)
        med, ov = evaluate_pair(src_pcd, self.scan_pcd, T,
                                cfg['score_voxel'], cfg['overlap_dist'])
        print('    %s -> 口扫: 中位 %.3f, 重叠 %.0f%%' % (tag, med, 100 * ov))
        return T, med, ov

    def anchor_main_cloud(self, mc, cfg):
        print('  [锚定] 主图(%d点)配准到口扫模型...' % len(mc.points))
        res = self.register_to_scan(mc, T_init=None, tag='主图')
        if res is not None and res[1] < cfg['scan_accept_median']:
            self.T_main_to_scan = res[0]
            print('  [锚定] 自动成功(中位 %.3f)' % res[1])
        else:
            print('  [锚定] 自动失败或残差过大, 转手动(主图<->口扫)')
            aligner = ManualAligner(cfg)
            Tm = aligner.manual_align(mc, self.scan_pcd, tag=' 主图->口扫')
            if Tm is None:
                print('  [锚定] 手动也失败, 锚定模式放弃')
                return False
            self.T_main_to_scan = Tm
        print('  [锚定] 主图已锚定到口扫坐标系')
        return True

    def apply_to_main(self, Ms, use):
        assert self.T_main_to_scan is not None
        for k in use:
            Ms[k] = self.T_main_to_scan @ Ms[k]
        return Ms

    def register(self, raws, files):
        cfg = self.cfg
        n = len(raws)
        edges, weak_edges = PairwiseRegistrator(cfg).register_all_pairs(raws, files)
        go = GlobalOptimizer(cfg)
        Ms, use = go.optimize(n, edges, raws)
        if Ms is None or len(use) < 2:
            return None, [], edges, weak_edges
        merger = Merger(cfg)
        _, mg_clean, _ = merger.merge_and_clean(raws, Ms, use)
        if not self.anchor_main_cloud(mg_clean, cfg):
            return None, [], edges, weak_edges
        Ms = self.apply_to_main(Ms, use)
        failed = [k for k in range(n) if k not in set(use)]
        print('  [锚定] 失败帧对口扫配准(重残差轻重叠, 接受标准: 中位<%.2fmm)...'
              % cfg['scan_accept_median'])
        for k in failed:
            res = self.register_to_scan(raws[k], T_init=None, tag='帧%d' % k)
            if res is not None:
                T_frame_to_scan, med, ov = res
                if med < cfg['scan_accept_median']:
                    Ms[k] = T_frame_to_scan
                    use.append(k)
                else:
                    print('    帧%d 残差过大(%.3f), 标记待抢救' % (k, med))
        use.sort()
        print('  [锚定] 对口扫直接成功 %d/%d 帧' % (len(use), n))
        return Ms, use, edges, weak_edges

# ================================================================
#  模块3: 旧版星型拓扑(A/B 对照)
# ================================================================
class StarRegistrator:
    def __init__(self, cfg):
        self.cfg = cfg

    def register(self, raws):
        cfg = self.cfg
        n = len(raws)
        sizes = [len(r.points) for r in raws]
        cand_refs = ([cfg['ref_frame']] if cfg.get('ref_frame') is not None
                     else list(np.argsort(sizes)[::-1][:cfg.get('ref_cands', 3)]))
        best = None
        for ref in cand_refs:
            print('  --- 尝试参考帧 %d (%d点) ---' % (ref, sizes[ref]))
            Ms, good, medians = self._star_pass(raws, ref)
            key = (len(good), -float(np.mean(list(medians.values()))) if medians else -9)
            print('  参考帧%d: 成功 %d/%d' % (ref, len(good), n))
            if best is None or key > best[0]:
                best = (key, ref, Ms, good)
            if len(good) / n >= cfg.get('ref_early_stop', 0.9):
                break
        _, ref, Ms, good = best
        print('  选定参考帧 %d, 成功 %d/%d' % (ref, len(good), n))
        return Ms, sorted(good), ref

    def _star_pass(self, raws, ref):
        cfg = self.cfg
        n = len(raws)
        rd, rf = fpfh_feats(raws[ref], cfg['voxel_fpfh'], cfg['fpfh_radius'])
        r_icp = voxel_down(raws[ref], cfg['icp_voxel'])
        Ms = {ref: np.eye(4)}; good = [ref]; medians = {}
        for k in range(n):
            if k == ref:
                continue
            sd, sf = fpfh_feats(raws[k], cfg['voxel_fpfh'], cfg['fpfh_radius'])
            try:
                T0 = ransac_init(sd, sf, rd, rf, cfg['ransac_dist'], cfg['ransac_iter'])
            except Exception as e:
                print('    帧%d RANSAC失败: %s' % (k, e)); continue
            s_icp = voxel_down(raws[k], cfg['icp_voxel'])
            T = T0
            for thr in cfg['pair_icp_thr']:
                T, _ = icp_custom(s_icp, r_icp, thr, cfg['keep_ratio'], T_init=T)
            med, ov = evaluate_pair(raws[k], raws[ref], T, cfg['score_voxel'], cfg['overlap_dist'])
            Ms[k] = T; medians[k] = med
            tag = 'OK' if (ov > 0.5 and med < 0.5) else 'weak'
            if tag == 'OK':
                good.append(k)
            print('    帧%2d->ref%d: 中位%.3f 重叠%.0f%% [%s]' % (k, ref, med, 100 * ov, tag))
        return Ms, good, medians

# ================================================================
#  模块4: 人工配准(PyVista 射线拾取)
# ================================================================
pv = None  # Loaded only when manual picking is requested.

class ManualAligner:
    SRC_COLOR = 'red'
    TGT_COLOR = 'darkgreen'

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _o3d_to_pv(pcd):
        pts = np.asarray(pcd.points)
        global pv
        if pv is None:
            import pyvista as pv
        pdata = pv.PolyData(pts)
        if pcd.has_colors():
            pdata['colors'] = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        return pdata

    def _pick_one(self, pcd, title, color):
        pdata = self._o3d_to_pv(pcd)
        pl = pv.Plotter(title=title)
        if 'colors' in pdata.array_names:
            pl.add_mesh(pdata, scalars='colors', rgb=True,
                        point_size=3.0, render_points_as_spheres=True)
        else:
            pl.add_mesh(pdata, color=color, point_size=3.0,
                        render_points_as_spheres=True)
        picked = []

        def cb(point, *args):
            p = np.asarray(point, float)
            if picked:
                d = np.linalg.norm(p - np.array(picked), axis=1).min()
                if d < 0.5:
                    print('  [忽略] 与已选点太近 (%.2f mm), 请点别处' % d)
                    return
            picked.append(p)
            pl.add_mesh(pv.Sphere(radius=0.4, center=p), color='red',
                        pickable=False, name=f'mk{len(picked)}')
            pl.add_text(f'已选 {len(picked)} 点', name='cnt', font_size=12)

        pl.enable_point_picking(callback=cb, picker='point',
                                left_clicking=False,
                                show_message='右键点选特征点, 选完直接关窗')
        pl.show()
        return np.array(picked) if picked else np.zeros((0, 3))

    def manual_align(self, src_raw, tgt_pcd, tag=''):
        cfg = self.cfg
        tgt_show = o3d.geometry.PointCloud(tgt_pcd)
        tgt_show.paint_uniform_color([0.0, 0.55, 0.2])
        src_show = voxel_down(src_raw, 1.0)
        src_show.paint_uniform_color([1.0, 0.2, 0.2])

        attempt = 0
        while True:
            attempt += 1
            if attempt > 5:
                print('  尝试次数过多, 放弃'); return None
            print('  [人工配准%s 第%d次] 先弹【源(红)】窗口打点(>=4个,建议5~6个分散),关闭窗口'
                  % (tag, attempt))
            Ps = self._pick_one(src_show, '源(红): 右键点选特征点, 选完关窗', self.SRC_COLOR)
            print('  [人工配准%s] 再弹【目标主云(深绿)】窗口, 按【相同顺序】点对应点' % tag)
            Pt = self._pick_one(tgt_show, '目标主云(深绿): 按相同顺序点对应点, 选完关窗', self.TGT_COLOR)
            if len(Ps) != len(Pt) or len(Ps) < 4:
                print('    源与目标必须各选相同数量的点，且至少4对')
                continue
            if min(np.linalg.matrix_rank(Ps - Ps.mean(axis=0)),
                   np.linalg.matrix_rank(Pt - Pt.mean(axis=0))) < 2:
                print('    对应点太接近共线，请在不同结构上分散选点')
                continue
            T = kabsch(Ps, Pt)
            src_fine = voxel_down(src_raw, cfg['score_voxel'])
            med, ov = evaluate_pair(src_raw, tgt_pcd, T, cfg['score_voxel'], cfg['overlap_dist'])
            limit = float(cfg['manual_refine_max_shift'])
            if limit > 0:
                candidate = T.copy()
                for thr in (2.0, 1.0, 0.5):
                    candidate, _ = icp_custom(src_fine, tgt_pcd, thr, keep_ratio=0.75,
                                              max_iter=60, T_init=candidate)
                if np.isfinite(candidate).all():
                    bounds = src_raw.get_axis_aligned_bounding_box().get_box_points()
                    corners = np.asarray(bounds)
                    shift = corners @ (candidate[:3, :3] - T[:3, :3]).T
                    shift += candidate[:3, 3] - T[:3, 3]
                    max_shift = float(np.linalg.norm(shift, axis=1).max())
                    med_new, ov_new = evaluate_pair(src_raw, tgt_pcd, candidate,
                                                    cfg['score_voxel'], cfg['overlap_dist'])
                    if max_shift <= limit and med_new <= med and ov_new >= ov:
                        T, med, ov = candidate, med_new, ov_new
                        print('    采用局部精修，源包围盒最大位移 %.3f' % max_shift)
                    else:
                        print('    局部精修未通过位移或重叠检查，保留手动点选位姿')
            print('  人工配准%s: 中位 %.3f mm, 重叠 %.0f%%' % (tag, med, 100 * ov))
            try:
                s = voxel_down(src_raw, 1.0); s.transform(T)
                s.paint_uniform_color([1.0, 0.2, 0.2])
                t = o3d.geometry.PointCloud(tgt_pcd)
                t.paint_uniform_color([0.0, 0.55, 0.2])
                o3d.visualization.draw_geometries([s, t],
                    window_name='叠加检查%s: 红=源 深绿=主云' % tag)
            except Exception as e:
                print('  (无法弹叠加图: %s)' % e)
            print('  结构对上(无重影)选 y; 明显错位选 r 重选; 放弃选 n')
            try:
                cmd = input('  接受该配准? [y/r/n] ').strip().lower()
            except EOFError:
                cmd = 'n'
            if cmd == 'y':
                return T
            if cmd == 'r':
                continue
            return None

# ================================================================
#  模块5: 合并 + 一致性清洗 + 可选薄壳过滤
# ================================================================
class Merger:
    def __init__(self, cfg):
        self.cfg = cfg

    def merge_and_clean(self, raws, Ms, use):
        cfg = self.cfg
        print('  [merge] 合并 + 一致性清洗...')
        pts, cols = [], []
        for k in use:
            p = _score_cloud(raws, k, cfg); p.transform(Ms[k])
            pts.append(np.asarray(p.points))
            cols.append(np.asarray(p.colors) if p.has_colors()
                          else np.tile([0.7, 0.7, 0.7], (len(p.points), 1)))
        mf = o3d.geometry.PointCloud()
        mf.points = o3d.utility.Vector3dVector(np.vstack(pts))
        mf.colors = o3d.utility.Vector3dVector(np.vstack(cols))
        mf = mf.voxel_down_sample(cfg['merge_voxel'])
        P = np.asarray(mf.points)
        tree = cKDTree(P)
        cnt = np.zeros(len(P), dtype=np.int16)
        per_frame = []
        for k in use:
            p = _score_cloud(raws, k, cfg); p.transform(Ms[k])
            t = np.asarray(p.points); per_frame.append(t)
            idx = tree.query_ball_point(t, r=cfg['cover_radius'])
            hits = [x for x in idx if len(x) > 0]
            if hits:
                cnt[np.unique(np.concatenate(hits).astype(np.int64))] += 1
        keep = cnt >= cfg['cover_min']
        mc = o3d.geometry.PointCloud()
        mc.points = o3d.utility.Vector3dVector(P[keep])
        mc.colors = o3d.utility.Vector3dVector(np.asarray(mf.colors)[keep])
        print('  [merge] 全量 %d 点; 干净版(>=%d视角) %d 点' % (len(P), cfg['cover_min'], keep.sum()))
        if cfg.get('shell_filter', False):
            mc = self.remove_shell(mc, cfg)
        return mf, mc, per_frame

    def remove_shell(self, mc, cfg):
        r = cfg.get('shell_radius', 1.0)
        gap_thr = cfg.get('shell_gap', 0.12)
        ratio_thr = cfg.get('shell_ratio', 0.35)
        P = np.asarray(mc.points)
        print('  [shell] 薄壳过滤: r=%.2f gap=%.2f ratio=%.2f, %d 点...'
              % (r, gap_thr, ratio_thr, len(P)))
        if not mc.has_normals():
            mc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
        N = np.asarray(mc.normals)
        tree = cKDTree(P)
        nbrs = tree.query_ball_point(P, r=r)
        keep = np.ones(len(P), bool)
        for i in range(len(P)):
            idx = nbrs[i]
            if len(idx) < 30:
                continue
            h = (P[idx] - P[i]) @ N[i]
            hs = np.sort(h)
            gaps = np.diff(hs)
            pos = np.arange(1, len(hs))
            valid = (pos >= 10) & (pos <= len(hs) - 10)
            if not valid.any():
                continue
            jj = np.where(valid)[0]
            j = jj[int(np.argmax(gaps[jj]))]
            if gaps[j] < gap_thr:
                continue
            c_lo, c_hi = j + 1, len(hs) - j - 1
            weak_lo = c_lo <= c_hi
            weak, strong = min(c_lo, c_hi), max(c_lo, c_hi)
            if weak / strong > ratio_thr:
                continue
            hp = 0.0
            in_weak = (hp <= hs[j]) if weak_lo else (hp > hs[j])
            if in_weak:
                keep[i] = False
        mc2 = mc.select_by_index(np.where(keep)[0])
        print('  [shell] 过滤: %d -> %d 点(删 %d)' % (len(P), keep.sum(), len(P) - keep.sum()))
        return mc2

# ================================================================
#  失败帧抢救(弱边级联)
# ================================================================
def component_local_poses(comp, edges):
    """用组内有效边传播局部位姿，并返回用于审计的 BFS 树边。"""
    cset = set(comp)
    deg = {k: 0 for k in comp}
    adj = {k: [] for k in comp}
    for e in edges or []:
        if e.valid and e.i in cset and e.j in cset:
            adj[e.i].append((e.j, e.T_ij))
            adj[e.j].append((e.i, np.linalg.inv(e.T_ij)))
            deg[e.i] += 1; deg[e.j] += 1
    anchor_ = max(comp, key=lambda k: deg[k])
    T_loc = {anchor_: np.eye(4)}
    dq = deque([anchor_])
    tree_edges = set()
    while dq:
        u = dq.popleft()
        for v, T_uv in adj[u]:
            if v not in T_loc:
                T_loc[v] = T_loc[u] @ np.linalg.inv(T_uv)
                tree_edges.add((min(u, v), max(u, v)))
                dq.append(v)
    return T_loc, anchor_, tree_edges


def rescue_failed_frames(raws, Ms, use, cfg, edges=None, weak_edges=None):
    aligner = ManualAligner(cfg)
    merger = Merger(cfg)
    use = list(use)

    outdir = cfg.get('_outdir', '.')
    snapshots = []
    confirmed_manual_poses = {}
    cfg.setdefault('_manual_bridge_frames', [])
    step_i = [0]

    def build_merged(frames):
        pts = []
        for k in frames:
            p = _score_cloud(raws, k, cfg); p.transform(Ms[k])
            pts.append(np.asarray(p.points))
        mg = o3d.geometry.PointCloud()
        mg.points = o3d.utility.Vector3dVector(np.vstack(pts))
        return mg.voxel_down_sample(cfg['score_voxel'])

    def update_main():
        nonlocal mg
        mg = build_merged(use)

    anchor = None
    mg = build_merged(use)
    if cfg.get('scan_model_path'):
        anchor = ScanAnchorRegistrator(cfg)
        anchor.load_scan(cfg['scan_model_path'])
        _, mg_clean, _ = merger.merge_and_clean(raws, Ms, use)
        if anchor.anchor_main_cloud(mg_clean, cfg):
            Ms = anchor.apply_to_main(Ms, use)
            update_main()
            print('  [锚定] 主图已切换到口扫坐标系, 失败帧将直接对口扫配准')
        else:
            anchor = None

    failed = [k for k in range(len(raws)) if k not in set(use)]
    n0 = len(use)
    frozen_initial = ({k: np.asarray(Ms[k]).copy() for k in use}
                      if cfg.get('freeze_seed_main', False) else {})
    if not failed or len(use) < 2:
        return Ms, sorted(use), 0
    # update_main() 会创建新对象；手动点选始终使用此初始快照。
    manual_target = mg
    target_path = os.path.join(outdir, 'manual_target_before_rescue.ply')
    if o3d.io.write_point_cloud(target_path, manual_target):
        print('  [rescue] 固定的手动配准目标：%s' % target_path)
    print('[rescue] %d 帧未自动接入: %s' % (len(failed), failed))

    weak_map = {}
    if weak_edges:
        for e in weak_edges:
            weak_map.setdefault(e.i, []).append(e)
            weak_map.setdefault(e.j, []).append(e)
        print('  [rescue] 弱边缓存: %d 条(将作为抢救初值)' % len(weak_edges))

    def save_snapshot_and_merged(accepted_frames):
        i = step_i[0]; step_i[0] += 1
        try:
            mg_cur = build_merged(use)
            path = os.path.join(outdir, 'merged_step%02d.ply' % i)
            o3d.io.write_point_cloud(path, mg_cur)
            print('      [存档] %s (主图%d帧)' % (path, len(use)))
        except Exception as e:
            print('      [存档失败] %s' % e)
        snapshots.append(list(accepted_frames))
        try:
            np.savez(os.path.join(outdir, 'snapshot_step%02d.npz' % i),
                     accepted=np.array(accepted_frames),
                     use_frames=np.array(use))
        except Exception:
            pass

    def undo_last():
        if not snapshots:
            print('      [撤销] 没有可撤销的手动操作')
            return False
        frames = snapshots.pop()
        cfg['_manual_bridge_frames'] = [k for k in cfg['_manual_bridge_frames'] if k not in frames]
        for k in frames:
            confirmed_manual_poses.pop(k, None)
            Ms.pop(k, None)
            if k in use:
                use.remove(k)
            if k not in failed:
                failed.append(k)
        failed.sort()
        print('      [撤销] 已回滚 %s, 主图回到 %d 帧' % (frames, len(use)))
        return True

    def merge_with_poses(frames, poses):
        pts = []
        for k in frames:
            p = _score_cloud(raws, k, cfg); p.transform(poses[k])
            pts.append(np.asarray(p.points))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.vstack(pts))
        return pc.voxel_down_sample(cfg['score_voxel'])

    def edge_init_for_group(comp, T_loc):
        use_set = set(use)
        cset = set(comp)
        for e in edges or []:
            if not e.valid:
                continue
            if e.j in use_set and e.i in cset and e.i in T_loc and e.j in Ms:
                W_i = Ms[e.j] @ e.T_ij
                return W_i @ np.linalg.inv(T_loc[e.i])
            if e.i in use_set and e.j in cset and e.j in T_loc and e.i in Ms:
                W_j = Ms[e.i] @ np.linalg.inv(e.T_ij)
                return W_j @ np.linalg.inv(T_loc[e.j])
        return None

    def edge_init_for_frame(k):
        use_set = set(use)
        for e in edges or []:
            if not e.valid:
                continue
            if e.j in use_set and e.i == k and e.j in Ms:
                return Ms[e.j] @ e.T_ij
            if e.i in use_set and e.j == k and e.i in Ms:
                return Ms[e.i] @ np.linalg.inv(e.T_ij)
        return None

    def weak_init_for_frame(k):
        use_set = set(use)
        for e in weak_map.get(k, []):
            nbr = e.j if e.i == k else e.i
            if nbr in use_set and nbr in Ms:
                T_k_nbr = e.T_ij if e.i == k else np.linalg.inv(e.T_ij)
                return Ms[nbr] @ T_k_nbr
        return None

    def pca_init_for_group(sub_pcd, mg):
        try:
            c_sub = np.asarray(sub_pcd.points).mean(axis=0)
            _, _, V_sub = np.linalg.svd(np.asarray(sub_pcd.points) - c_sub, full_matrices=False)
            c_main = np.asarray(mg.points).mean(axis=0)
            _, _, V_main = np.linalg.svd(np.asarray(mg.points) - c_main, full_matrices=False)
            R = V_main.T @ V_sub
            if np.linalg.det(R) < 0:
                V_main[-1] *= -1
                R = V_main.T @ V_sub
            t = c_main - R @ c_sub
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
            return T
        except Exception:
            return None

    def align_cloud_auto(src_pcd, mg, T_init=None, src_frame_idx=None):
        tgt = anchor.scan_pcd if (anchor and anchor.T_main_to_scan is not None) else mg
        if anchor and anchor.T_main_to_scan is not None and T_init is not None:
            T_init = anchor.T_main_to_scan @ T_init
        if T_init is None and src_frame_idx is not None:
            T_init = weak_init_for_frame(src_frame_idx)
        return _rescue_align_fast(src_pcd,tgt,cfg,T_init)

    def refine_frames_to_main(frames, T_map, mg):
        tgt = anchor.scan_pcd if (anchor and anchor.T_main_to_scan is not None) else mg
        tgt_icp = voxel_down(tgt, cfg['icp_voxel'])
        out = {}
        for k in frames:
            T = T_map[k]
            s_icp = voxel_down(raws[k], cfg['icp_voxel'])
            thr_seq = [1.0, 0.5] if (anchor and anchor.T_main_to_scan is not None) else [2.0, 1.0]
            for thr in thr_seq:
                T, _ = icp_custom(s_icp, tgt_icp, thr, 0.7, max_iter=60, T_init=T)
            med, ov = evaluate_pair(raws[k], tgt, T, cfg['score_voxel'], cfg['overlap_dist'])
            out[k] = T
            print('      帧%2d 精修后: 中位 %.3f, 重叠 %.0f%%' % (k, med, 100 * ov))
        return out

    def cascade_rescue(seed_frames, max_depth=2):
        queue = deque([(k, 0) for k in seed_frames])
        seen = set(seed_frames)
        n_cascaded = 0
        while queue:
            k, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for e in weak_map.get(k, []):
                nbr = e.j if e.i == k else e.i
                if nbr in seen or nbr not in singles or nbr in declined:
                    continue
                seen.add(nbr)
                T_k_nbr = e.T_ij if e.i == k else np.linalg.inv(e.T_ij)
                T_init = Ms[k] @ T_k_nbr
                res = align_cloud_auto(raws[nbr], mg, T_init=T_init, src_frame_idx=nbr)
                if res is not None:
                    T, med, ov = res
                    if ov > cfg['rescue_accept_overlap'] and med < cfg['rescue_accept_median']:
                        refined = refine_frames_to_main([nbr], {nbr: T}, mg)
                        Ms[nbr] = refined[nbr]
                        use.append(nbr); use.sort()
                        singles.remove(nbr)
                        if nbr in failed:
                            failed.remove(nbr)
                        update_main()
                        n_cascaded += 1
                        queue.append((nbr, depth + 1))
                        print('      [级联] 帧%d -> 帧%d: 中位 %.3f, 重叠 %.0f%%, 救回'
                              % (k, nbr, med, 100 * ov))
        return n_cascaded

    def group_failed():
        if not edges:
            return [], list(failed)
        fset = set(failed)
        adj = {k: [] for k in failed}
        for e in edges:
            if e.valid and e.i in fset and e.j in fset:
                adj[e.i].append(e.j); adj[e.j].append(e.i)
        seen = set(); groups = []; singles = []
        for k in failed:
            if k in seen:
                continue
            comp, dq = [], deque([k]); seen.add(k)
            while dq:
                u = dq.popleft(); comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v); dq.append(v)
            comp.sort()
            if len(comp) > 1:
                groups.append(comp)
            else:
                singles.append(comp[0])
        groups.sort(key=len, reverse=True)
        return groups, singles

    groups, singles = group_failed()
    if groups:
        print('  [rescue] 失败帧分组: %d 个小分量 %s + %d 个孤帧 %s'
              % (len(groups), groups, len(singles), singles))
    declined = set()
    acc_ov, acc_med = cfg['rescue_accept_overlap'], cfg['rescue_accept_median']

    refined_components = {}
    for round_i in range(cfg.get('rescue_rounds', 99)):
        progress = False
        for comp in list(groups):
            T_loc, comp_anchor, _ = component_local_poses(comp, edges)
            if cfg.get('graph_refine', True) and tuple(comp) not in refined_components:
                T_loc = refine_component_graph(T_loc, edges, comp_anchor, cfg)
                refined_components[tuple(comp)] = T_loc
            elif tuple(comp) in refined_components:
                T_loc = refined_components[tuple(comp)]
            sub_pcd = merge_with_poses(comp, T_loc)
            G0 = edge_init_for_group(comp, T_loc)
            res = align_cloud_auto(sub_pcd, mg, T_init=G0)
            if res is None:
                continue
            T_sub, med, ov = res
            src_tag = '边传播' if G0 is not None else 'RANSAC'
            print('    分量%s(%d帧) 自动[%s]: 中位 %.3f, 重叠 %.0f%%'
                  % (comp, len(comp), src_tag, med, 100 * ov))
            if ov > acc_ov and med < acc_med:
                # 使用整个分量通过评分的变换，保持球面等结构的组内相对位姿。
                for k in comp:
                    Ms[k] = T_sub @ T_loc[k]
                use.extend(comp); use.sort()
                for k in comp:
                    if k in failed:
                        failed.remove(k)
                groups.remove(comp)
                update_main()
                progress = True
                print('    -> 分量 %s 作为整体刚体接入, 主图扩至 %d 帧' % (comp, len(use)))
        for k in list(singles):
            T_init = edge_init_for_frame(k)
            res = align_cloud_auto(raws[k], mg, T_init=T_init, src_frame_idx=k)
            if res is None:
                continue
            T, med, ov = res
            src_tag = '边传播' if T_init is not None else 'RANSAC'
            print('    帧%2d 自动[%s]: 中位 %.3f, 重叠 %.0f%%' % (k, src_tag, med, 100 * ov))
            if ov > acc_ov and med < acc_med:
                refined = refine_frames_to_main([k], {k: T}, mg)
                Ms[k] = refined[k]
                use.append(k); use.sort()
                if k in failed:
                    failed.remove(k)
                singles.remove(k)
                update_main()
                progress = True
                print('    -> 帧%d 救回' % k)
        if not groups and not singles:
            break
        if progress:
            continue
        if not cfg.get('manual_rescue', True):
            break
        target = None
        undo_happened = False
        for comp in sorted(groups, key=len, reverse=True):
            if tuple(comp) in declined:
                continue
            try:
                cmd = input('    自动无果, 人工架桥分量%s(%d帧)? [y=打点 / u=撤销上一次 / 回车跳过] '
                            % (comp, len(comp))).strip().lower()
            except EOFError:
                cmd = ''
            if cmd == 'u':
                undo_last(); undo_happened = True; break
            if cmd == 'y':
                target = ('g', comp); break
            declined.add(tuple(comp))
        if not undo_happened and target is None:
            for k in singles:
                if k in declined:
                    continue
                try:
                    cmd = input('    人工架桥帧%d? [y=打点 / u=撤销上一次 / 回车跳过] ' % k).strip().lower()
                except EOFError:
                    cmd = ''
                if cmd == 'u':
                    undo_last(); undo_happened = True; break
                if cmd == 'y':
                    target = ('s', k); break
                declined.add(k)
        if undo_happened:
            break
        if target is None:
            print('  [rescue] 手动架桥全部跳过, 结束')
            break
        if target[0] == 'g':
            comp = target[1]
            T_loc, comp_anchor, _ = component_local_poses(comp, edges)
            if cfg.get('graph_refine', True) and tuple(comp) not in refined_components:
                T_loc = refine_component_graph(T_loc, edges, comp_anchor, cfg)
                refined_components[tuple(comp)] = T_loc
            elif tuple(comp) in refined_components:
                T_loc = refined_components[tuple(comp)]
            sub_pcd = merge_with_poses(comp, T_loc)
            try:
                Tm = aligner.manual_align(sub_pcd, manual_target, tag=' 分量%s' % comp)
            except Exception as e:
                Tm = None
                print('    (无显示环境, 跳过人工: %s)' % e)
                break
            if Tm is not None:
                # 整个分量使用预览时确认的刚体变换，保留内部相对位姿。
                for k in comp:
                    Ms[k] = Tm @ T_loc[k]
                    confirmed_manual_poses[k] = Ms[k].copy()
                cfg['_manual_bridge_frames'] = sorted(set(cfg['_manual_bridge_frames']) | set(comp))
                use.extend(comp); use.sort()
                for k in comp:
                    if k in failed:
                        failed.remove(k)
                groups.remove(comp)
                update_main()
                save_snapshot_and_merged(comp)
                n_cascaded = cascade_rescue(comp)
                if n_cascaded > 0:
                    print('      [级联] 手动架桥后自动救回 %d 个邻居帧' % n_cascaded)
                print('    -> 手动架桥成功: 分量%s接入, 主图%d帧, 自动重试剩余分量'
                      % (comp, len(use)))
            else:
                declined.add(tuple(comp))
        else:
            k = target[1]
            try:
                Tm = aligner.manual_align(raws[k], manual_target, tag=' 帧%d' % k)
            except Exception as e:
                Tm = None
                print('    (无显示环境, 跳过人工: %s)' % e)
                break
            if Tm is not None:
                Ms[k] = Tm.copy()
                confirmed_manual_poses[k] = Ms[k].copy()
                cfg['_manual_bridge_frames'] = sorted(set(cfg['_manual_bridge_frames']) | {k})
                use.append(k); use.sort()
                if k in failed:
                    failed.remove(k)
                singles.remove(k)
                update_main()
                save_snapshot_and_merged([k])
                n_cascaded = cascade_rescue([k])
                if n_cascaded > 0:
                    print('      [级联] 手动架桥后自动救回 %d 个邻居帧' % n_cascaded)
                print('    -> 手动架桥成功: 帧%d接入, 自动重试剩余' % k)
            else:
                declined.add(k)

    left = [k for k in range(len(raws)) if k not in set(use)]
    if left:
        print('  [rescue] 最终仍失败: %s' % left)

    if len(use) > n0 and len(use) >= 2:
        if cfg.get('freeze_seed_main', False):
            print('[rescue] 主图冻结：跳过救援后的全体 multiway 抛光')
        else:
            print('[rescue] 全体 %d 帧统一 multiway 抛光...' % len(use))
            Ms = GlobalOptimizer(cfg)._multiway_icp(raws, use, Ms)
    for k, transform in confirmed_manual_poses.items():
        if k in use:
            Ms[k] = transform.copy()
    if frozen_initial:
        for _k, _old in frozen_initial.items():
            Ms[_k] = _old.copy()
        print('[rescue] 已核对并恢复最初主分量 %d 帧的原始位姿' % len(frozen_initial))
    return Ms, sorted(use), len(use) - n0

# ================================================================
#  局部区域守门
# ================================================================
def _consistency_gate_legacy(raws, Ms, use, cfg):
    if not cfg.get('gate_enable', True) or len(use) < 3:
        return Ms, use
    use = list(use)
    block_size = cfg.get('gate_block_size', 15.0)
    min_block_pts = cfg.get('gate_block_min_pts', 200)
    mad_k = cfg.get('gate_mad_k', 3.0)

    for rd in range(cfg.get('gate_max_rounds', 3)):
        if len(use) < 2:
            return Ms, sorted(use)
        down_w = {}
        for k in use:
            p = _score_cloud(raws, k, cfg); p.transform(Ms[k])
            down_w[k] = np.asarray(p.points)
        others_all = np.vstack(list(down_w.values()))
        tree_all = cKDTree(others_all)

        drift_ratio = {}
        for k in use:
            P = down_w[k]
            d, _ = tree_all.query(P, k=1)
            ov_mask = d < cfg['overlap_dist']
            if ov_mask.sum() < 100 or (~ov_mask).sum() < 100:
                continue
            res_in = float(np.median(d[ov_mask]))
            drift_out = float(np.median(d[~ov_mask]))
            drift_ratio[k] = drift_out / max(res_in, 1e-3)
        if drift_ratio:
            vals = np.array(list(drift_ratio.values()))
            med_c = np.median(vals)
            mad_v = np.median(np.abs(vals - med_c)) + 1e-9
            thr_drift = med_c + mad_k * 1.4826 * mad_v
            drift_outliers = [k for k, r in drift_ratio.items() if r > thr_drift]
            if drift_outliers:
                print('  [守门-漂移检测] 漂移比中位 %.1f, 阈值 %.1f, 剔除帧: %s'
                      % (med_c, thr_drift,
                         ['帧%d(比%.0f)' % (k, drift_ratio[k]) for k in drift_outliers]))
                for k in drift_outliers:
                    use.remove(k)
                if len(use) >= 2:
                    Ms = GlobalOptimizer(cfg)._multiway_icp(raws, use, Ms)
                down_w = {k: down_w[k] for k in use}
                others_all = np.vstack(list(down_w.values()))
                tree_all = cKDTree(others_all)

        bad_frames = {}
        for k in use:
            P = down_w[k]
            if len(P) < min_block_pts:
                continue
            mins = P.min(axis=0)
            block_idx = np.floor((P - mins) / block_size).astype(int)
            from collections import defaultdict
            blocks = defaultdict(list)
            for i, b in enumerate(block_idx):
                blocks[tuple(b)].append(i)
            max_res = 0.0; worst_center = None
            for b, idxs in blocks.items():
                if len(idxs) < min_block_pts:
                    continue
                pts_b = P[idxs]
                d, _ = tree_all.query(pts_b, k=1)
                res95 = float(np.percentile(d, 95))
                if res95 > max_res:
                    max_res = res95
                    worst_center = (np.array(b) * block_size + mins + block_size / 2)
            if max_res > 0:
                bad_frames[k] = (max_res, worst_center)

        if not bad_frames:
            print('  [局部守门 第%d轮] 无离群区块, 通过' % (rd + 1))
            break

        res_vals = np.array([v[0] for v in bad_frames.values()])
        med_center = np.median(res_vals)
        mad = np.median(np.abs(res_vals - med_center)) + 1e-9
        thr = med_center + mad_k * 1.4826 * mad
        outliers = {k: v for k, v in bad_frames.items() if v[0] > thr}
        print('  [局部守门 第%d轮] 区块95分位残差中位 %.3f, 阈值 %.3f, 离群帧: %s'
              % (rd + 1, med_center, thr,
                 ['帧%d(%.2fmm@%.1f,%.1f,%.1f)' % (k, v[0], v[1][0], v[1][1], v[1][2])
                  for k, v in outliers.items()]))

        if not outliers:
            print('  [局部守门] 全部帧通过')
            break

        removed = []
        for k, (res95, center) in outliers.items():
            mo = o3d.geometry.PointCloud()
            mo.points = o3d.utility.Vector3dVector(others_all)
            mo = mo.voxel_down_sample(cfg['score_voxel'])
            src = voxel_down(raws[k], cfg['score_voxel'])
            T = Ms[k]
            for t in cfg.get('gate_refine_thr', [1.0, 0.5, 0.3, 0.15]):
                T, _ = icp_custom(src, mo, t, keep_ratio=0.8, max_iter=150, T_init=T)
            P2 = (T[:3, :3] @ np.asarray(voxel_down(raws[k], cfg['score_voxel']).points).T).T + T[:3, 3]
            mins = P2.min(axis=0)
            block_idx = np.floor((P2 - mins) / block_size).astype(int)
            from collections import defaultdict
            blocks = defaultdict(list)
            for i, b in enumerate(block_idx):
                blocks[tuple(b)].append(i)
            max_res2 = 0.0
            for b, idxs in blocks.items():
                if len(idxs) < min_block_pts:
                    continue
                d, _ = cKDTree(others_all).query(P2[idxs], k=1)
                max_res2 = max(max_res2, float(np.percentile(d, 95)))
            print('    帧%2d 加强精修: 最大区块95分位 %.2f -> %.2f mm' % (k, res95, max_res2))
            if max_res2 <= thr:
                Ms[k] = T
            else:
                removed.append(k)

        if removed:
            print('    [局部守门] 剔除 %d 帧: %s' % (len(removed), removed))
            for k in removed:
                use.remove(k)
            if len(use) >= 2:
                print('    [局部守门] 其余 %d 帧重新抛光...' % len(use))
                Ms = GlobalOptimizer(cfg)._multiway_icp(raws, use, Ms)
    return Ms, sorted(use)


def _consistency_gate_enforce(raws, Ms, use, cfg):
    if not cfg.get('gate_enable', True) or len(use) < 3:
        return Ms, use
    use = list(use)
    block_size = cfg.get('gate_block_size', 15.0)
    min_block_pts = cfg.get('gate_block_min_pts', 200)
    mad_k = cfg.get('gate_mad_k', 3.0)

    for rd in range(cfg.get('gate_max_rounds', 3)):
        if len(use) < 2:
            return Ms, sorted(use)
        down_w = {}
        for k in use:
            p = _score_cloud(raws, k, cfg); p.transform(Ms[k])
            down_w[k] = np.asarray(p.points)
        # 建立严格的leave-one-out目标：当前帧不能查询到自身点云。
        # 只有在显式 --gate-enforce 时才执行此耗时的剔除流程。
        other_clouds = {}
        other_trees = {}
        for _k in use:
            _q = np.vstack([down_w[_j] for _j in use if _j != _k and len(down_w[_j])])
            other_clouds[_k] = _q
            other_trees[_k] = cKDTree(_q)

        drift_ratio = {}
        for k in use:
            P = down_w[k]
            d, _ = other_trees[k].query(P, k=1)
            ov_mask = d < cfg['overlap_dist']
            if ov_mask.sum() < 100 or (~ov_mask).sum() < 100:
                continue
            res_in = float(np.median(d[ov_mask]))
            drift_out = float(np.median(d[~ov_mask]))
            drift_ratio[k] = drift_out / max(res_in, 1e-3)
        if drift_ratio:
            vals = np.array(list(drift_ratio.values()))
            med_c = np.median(vals)
            mad_v = np.median(np.abs(vals - med_c)) + 1e-9
            thr_drift = med_c + mad_k * 1.4826 * mad_v
            drift_outliers = [k for k, r in drift_ratio.items() if r > thr_drift]
            if drift_outliers:
                print('  [守门-漂移检测] 漂移比中位 %.1f, 阈值 %.1f, 剔除帧: %s'
                      % (med_c, thr_drift,
                         ['帧%d(比%.0f)' % (k, drift_ratio[k]) for k in drift_outliers]))
                for k in drift_outliers:
                    use.remove(k)
                if len(use) < 2:
                    return Ms, sorted(use)
                if len(use) >= 2:
                    Ms = GlobalOptimizer(cfg)._multiway_icp(raws, use, Ms)
                down_w = {k: down_w[k] for k in use}
                other_clouds = {}
                other_trees = {}
                for _k in use:
                    _q = np.vstack([down_w[_j] for _j in use if _j != _k and len(down_w[_j])])
                    other_clouds[_k] = _q
                    other_trees[_k] = cKDTree(_q)

        bad_frames = {}
        for k in use:
            P = down_w[k]
            if len(P) < min_block_pts:
                continue
            mins = P.min(axis=0)
            block_idx = np.floor((P - mins) / block_size).astype(int)
            from collections import defaultdict
            blocks = defaultdict(list)
            for i, b in enumerate(block_idx):
                blocks[tuple(b)].append(i)
            max_res = 0.0; worst_center = None
            for b, idxs in blocks.items():
                if len(idxs) < min_block_pts:
                    continue
                pts_b = P[idxs]
                d, _ = other_trees[k].query(pts_b, k=1)
                res95 = float(np.percentile(d, 95))
                if res95 > max_res:
                    max_res = res95
                    worst_center = (np.array(b) * block_size + mins + block_size / 2)
            if max_res > 0:
                bad_frames[k] = (max_res, worst_center)

        if not bad_frames:
            print('  [局部守门 第%d轮] 无离群区块, 通过' % (rd + 1))
            break

        res_vals = np.array([v[0] for v in bad_frames.values()])
        med_center = np.median(res_vals)
        mad = np.median(np.abs(res_vals - med_center)) + 1e-9
        thr = med_center + mad_k * 1.4826 * mad
        outliers = {k: v for k, v in bad_frames.items() if v[0] > thr}
        print('  [局部守门 第%d轮] 区块95分位残差中位 %.3f, 阈值 %.3f, 离群帧: %s'
              % (rd + 1, med_center, thr,
                 ['帧%d(%.2fmm@%.1f,%.1f,%.1f)' % (k, v[0], v[1][0], v[1][1], v[1][2])
                  for k, v in outliers.items()]))

        if not outliers:
            print('  [局部守门] 全部帧通过')
            break

        removed = []
        for k, (res95, center) in outliers.items():
            mo = o3d.geometry.PointCloud()
            mo.points = o3d.utility.Vector3dVector(other_clouds[k])
            mo = mo.voxel_down_sample(cfg['score_voxel'])
            src = voxel_down(raws[k], cfg['score_voxel'])
            T = Ms[k]
            for t in cfg.get('gate_refine_thr', [1.0, 0.5, 0.3, 0.15]):
                T, _ = icp_custom(src, mo, t, keep_ratio=0.8, max_iter=150, T_init=T)
            P2 = (T[:3, :3] @ np.asarray(voxel_down(raws[k], cfg['score_voxel']).points).T).T + T[:3, 3]
            mins = P2.min(axis=0)
            block_idx = np.floor((P2 - mins) / block_size).astype(int)
            from collections import defaultdict
            blocks = defaultdict(list)
            for i, b in enumerate(block_idx):
                blocks[tuple(b)].append(i)
            max_res2 = 0.0
            for b, idxs in blocks.items():
                if len(idxs) < min_block_pts:
                    continue
                d, _ = other_trees[k].query(P2[idxs], k=1)
                max_res2 = max(max_res2, float(np.percentile(d, 95)))
            print('    帧%2d 加强精修: 最大区块95分位 %.2f -> %.2f mm' % (k, res95, max_res2))
            if max_res2 <= thr:
                Ms[k] = T
            else:
                removed.append(k)

        if removed:
            print('    [局部守门] 剔除 %d 帧: %s' % (len(removed), removed))
            for k in removed:
                use.remove(k)
            if len(use) >= 2:
                print('    [局部守门] 其余 %d 帧重新抛光...' % len(use))
                Ms = GlobalOptimizer(cfg)._multiway_icp(raws, use, Ms)
    return Ms, sorted(use)


def consistency_gate_local(raws, Ms, use, cfg):
    """默认只输出定量报告，不再用包含自身点的KDTree误判漂移。
    enforce是可选的旧启发式(已修复自匹配)，有改变结果的风险；
    legacy供A/B比较旧结果，不推荐日常使用。
    """
    if not cfg.get('gate_enable', True):
        return Ms, use
    mode = str(cfg.get('gate_mode', 'audit')).lower()
    if mode in ('off', 'none'):
        print('  [守门] 已禁用')
        return Ms, use
    if mode == 'audit':
        print('  [守门] audit模式：本轮不移动/删除帧，结束后使用其它帧的点云做定量审计')
        return Ms, use
    if mode == 'enforce':
        if cfg.get('freeze_seed_main', False):
            raise ValueError('已启用 --freeze-main，不能同时启用 --gate-enforce；请改用 --gate-audit')
        print('  [守门] enforce模式：严格leave-one-out，可能修改位姿及丢弃帧')
        return _consistency_gate_enforce(raws, Ms, use, cfg)
    if mode == 'legacy':
        print('  [守门] legacy模式：旧版含自身的查询；仅为与上一版A/B复现')
        return _consistency_gate_legacy(raws, Ms, use, cfg)
    raise ValueError('未知gate_mode: %s' % mode)

# ================================================================
#  诊断与输出工具
# ================================================================
def leave_one_out_scores(raws, Ms, use, cfg):
    scores = {}
    down_w = {}
    for k in use:
        p = _score_cloud(raws, k, cfg); p.transform(Ms[k])
        down_w[k] = np.asarray(p.points)
    for k in use:
        others = np.vstack([down_w[j] for j in use if j != k])
        d, _ = cKDTree(others).query(down_w[k], k=1)
        scores[k] = (float(np.percentile(d, 95)), float((d < cfg['overlap_dist']).mean()))
    return scores

def print_scores(scores, cfg):
    meds = [v[0] for v in scores.values()]
    print('  最终: 95分位残差 %.3f~%.3f(均%.3f)mm, 最低重叠 %.0f%%'
          % (min(meds), max(meds), np.mean(meds), 100 * min(v[1] for v in scores.values())))
    worst = sorted(scores.items(), key=lambda kv: -kv[1][0])[:8]
    print('  留一95分位残差最高的帧(薄壳/错位嫌疑):')
    for k, (med, ov) in worst:
        print('    帧%2d: 95分位 %.3f mm, 重叠 %.0f%%' % (k, med, 100 * ov))

def export_by_frame(raws, Ms, use, cfg, outdir):
    import colorsys
    path = os.path.join(outdir, 'merged_by_frame.ply')
    pts, cols = [], []
    legend = {}
    m = max(len(use) - 1, 1)
    for i, k in enumerate(use):
        h = i / m
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
        p = _score_cloud(raws, k, cfg); p.transform(Ms[k])
        pts.append(np.asarray(p.points))
        cols.append(np.tile([r, g, b], (len(p.points), 1)))
        legend[k] = int(h * 360)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.vstack(pts))
    pc.colors = o3d.utility.Vector3dVector(np.vstack(cols))
    o3d.io.write_point_cloud(path, pc)
    print('  [诊断] 已导出 %s (每帧一色)' % path)
    print('  [诊断] 图例(帧号->色相, 0=红 60=黄 120=绿 180=青 240=蓝 300=品红):')
    line = '    ' + '  '.join('%d:%d°' % (k, legend[k]) for k in use)
    print(line)
    return path

def ring_closure_check(Ms, use, cfg):
    diffs = []
    us = sorted(use)
    for a, b in zip(us, us[1:] + [us[0]]):
        if b not in Ms or a not in Ms:
            continue
        D = np.linalg.inv(Ms[a]) @ Ms[b]
        diffs.append((abs(b - a), float(np.linalg.norm(D[:3, 3]))))
    if not diffs:
        return True, 0.0
    vals = np.array([v for _, v in diffs])
    med = np.median(vals) + 1e-9
    worst = float(vals.max())
    ok = worst < med * 5.0
    print('  [环闭合自检] 相邻帧平移: 中位 %.2f mm, 最大 %.2f mm, 比值 %.1f -> %s'
          % (med, worst, worst / med, '通过' if ok else '渐进累积误差'))
    return ok, worst

# ================================================================
#  Source-style main: 与救援分量共用 component_local_poses
#  一样，只通过单个连通分量内的有效两两变换构建点云。
#  该实验不声称局部BFS一定正确：尤其是无闭环的假边仍可能造成漂移。
# ================================================================
def build_source_style_main(n, edges, cfg):
    parent = list(range(n))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for e in edges:
        if not e.valid:
            continue
        a, b = find(e.i), find(e.j)
        if a != b:
            parent[a] = b

    comps = {}
    for k in range(n):
        comps.setdefault(find(k), []).append(k)
    components = sorted(comps.values(), key=lambda c: (-len(c), c[0]))
    if not components:
        return None, [], -1

    # 不默认假定帧数最多就配准正确：可指定一帧或分量序号。
    sf = cfg.get('source_seed_frame')
    if sf is not None:
        sf = int(sf)
        if not (0 <= sf < n):
            raise ValueError('source_seed_frame 不在 [0,%d] 范围内' % (n-1))
        chosen = next((idx for idx, comp in enumerate(components) if sf in comp), None)
        assert chosen is not None
    else:
        chosen = int(cfg.get('source_seed_component', 0))
        if not (0 <= chosen < len(components)):
            raise ValueError('source_seed_component=%d 越界; 分量数=%d'
                             % (chosen, len(components)))

    outdir = cfg.get('_outdir', '.')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'source_style_components.csv')
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['component', 'frame_count', 'valid_edges', 'median_pair_median',
                    'median_pair_overlap', 'is_selected', 'frame_ids'])
        for ci, comp in enumerate(components):
            cs = set(comp)
            ce = [e for e in edges if e.valid and e.i in cs and e.j in cs]
            w.writerow([ci, len(comp), len(ce),
                        float(np.median([e.median for e in ce])) if ce else '',
                        float(np.median([e.overlap for e in ce])) if ce else '',
                        int(ci == chosen), json.dumps(comp)])
    print('  [source-main] 各分量帧数：%s' % [len(c) for c in components])
    print('  [source-main] 分量清单：%s' % path)

    comp = components[chosen]
    print('  [source-main] 选择分量 %d，帧数=%d，成员=%s' % (chosen, len(comp), comp))
    if len(comp) < 2:
        print('  [source-main] 所选分量只有1帧，无法形成多帧主图')
        return None, comp, comp[0]

    T_loc, anchor, tree_edges = component_local_poses(comp, edges)

    # 独立审计：非树边与BFS相对位姿不一致，可标记风险但不擅自自动删边。
    edge_audit = os.path.join(outdir, 'source_style_internal_edge_audit.csv')
    with open(edge_audit, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['i', 'j', 'is_bfs_tree_edge', 'median', 'overlap',
                    'translation_closure_error', 'rotation_closure_error_deg'])
        for e in edges:
            if not e.valid or e.i not in T_loc or e.j not in T_loc:
                continue
            try:
                expected = np.linalg.inv(T_loc[e.j]) @ T_loc[e.i]
                D = np.linalg.inv(e.T_ij) @ expected
                terr = float(np.linalg.norm(D[:3, 3]))
                r = float(np.clip((np.trace(D[:3,:3])-1)/2,-1,1))
                rerr = float(np.degrees(np.arccos(r)))
            except Exception:
                terr = rerr = float('nan')
            w.writerow([e.i, e.j, int((min(e.i,e.j),max(e.i,e.j)) in tree_edges),
                        e.median, e.overlap, terr, rerr])
    print('  [source-main] 内部边闭环审计：%s' % edge_audit)
    if cfg.get('graph_refine', True):
        T_loc = refine_component_graph(T_loc, edges, anchor, cfg)
    print('  [source-main] 主图位姿已构建；闭环校正状态见 graph_refinement.json')
    return T_loc, sorted(T_loc), anchor

def export_source_style_main_cloud(raws, Ms, use, cfg):
    """按 rescue_failed_frames.merge_with_poses/build_merged 原式导出主图。
    不进行 cover_min 清洗，便于观察纯粹的配准几何结果。
    """
    if Ms is None or not use:
        return None
    pts = []
    for k in use:
        p = _score_cloud(raws, k, cfg)
        p.transform(Ms[k])
        pts.append(np.asarray(p.points))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(pts))
    cloud = cloud.voxel_down_sample(cfg['score_voxel'])
    path = os.path.join(cfg.get('_outdir', '.'), 'source_style_main_before_rescue.ply')
    o3d.io.write_point_cloud(path, cloud)
    print('  [source-main] 与“源分量”完全相同的合并点云：%s' % path)
    return cloud

# ================================================================
#  主流程
# ================================================================
def run(files, cfg):
    cfg['_files'] = files
    n = len(files)
    print('[1] 注册 %d 帧(懒加载, 按需读盘)...' % n)
    raws = LazyFrameLoader(files, capacity=cfg.get('lazy_capacity', 48))

    if cfg.get('scan_model_path'):
        print('[2] 配准策略 = 模型锚定(用完整口扫做基准)')
        anchor = ScanAnchorRegistrator(cfg)
        anchor.load_scan(cfg['scan_model_path'])
        Ms, use, edges, weak_edges = anchor.register(raws, files)
        if Ms is None or len(use) < 2:
            print('  锚定模式失败, 回退 multiway')
            cfg['scan_model_path'] = ''
            edges, weak_edges = None, None
        else:
            Ms, use, n_rescued = rescue_failed_frames(raws, Ms, use, cfg,
                                                      edges=edges, weak_edges=weak_edges)
            print('  配准成功 %d/%d 帧' % (len(use), n))
            if len(use) < 2:
                return raws, Ms, use, {}, 0, None, None, None
            mf, mc, per_frame_t = Merger(cfg).merge_and_clean(raws, Ms, use)
            scores = leave_one_out_scores(raws, Ms, use, cfg)
            print_scores(scores, cfg)
            return raws, Ms, use, scores, 0, mf, mc, per_frame_t

    edges, weak_edges = None, None
    ref = -1
    if cfg['strategy'] == 'multiway':
        print('[2] 配准策略 = pairwise + global multiway ICP')
        edges, weak_edges = PairwiseRegistrator(cfg).register_all_pairs(raws, files)
        if cfg.get('source_style_main', False):
            Ms, use, ref = build_source_style_main(n, edges, cfg)
        else:
            go = GlobalOptimizer(cfg)
            Ms, use = go.optimize(n, edges, raws)
            ref = go.pick_anchor(n, edges) if use else -1
        if cfg.get('source_style_main', False) and Ms is not None and len(use) >= 2:
            export_source_style_main_cloud(raws, Ms, use, cfg)
        if cfg.get('snapshot_initial', False) and Ms is not None and len(use) >= 2:
            try:
                snapshot_dir = os.path.join(cfg.get('_outdir', '.'), 'initial_main_before_rescue')
                os.makedirs(snapshot_dir, exist_ok=True)
                export_by_frame(raws, Ms, use, cfg, snapshot_dir)
            except Exception as exc:
                print('  [snapshot] 导出失败(不影响配准): %s' % exc)
        if cfg.get('initial_only', False):
            if Ms is None or len(use) < 2:
                print('  [initial-only] 可用主图不足2帧，未进入后续流程')
                return raws, Ms, use, {}, ref, None, None, None
            print('  [initial-only] 仅输出救援/守门前的选定主图 (%d帧)' % len(use))
            mf, mc, per_frame_t = Merger(cfg).merge_and_clean(raws, Ms, use)
            return raws, Ms, use, {}, ref, mf, mc, per_frame_t
        if Ms is None or len(use) < 2:
            print('  multiway 失败, 回退 star')
            cfg['strategy'] = 'star'
    if cfg['strategy'] == 'star':
        print('[2] 配准策略 = star(旧版星型拓扑)')
        Ms, use, ref = StarRegistrator(cfg).register(raws)

    Ms, use, n_rescued = rescue_failed_frames(raws, Ms, use, cfg,
                                              edges=edges, weak_edges=weak_edges)
    print('  配准成功 %d/%d 帧' % (len(use), n))
    if len(use) < 2:
        print('  成功帧不足, 无法继续')
        return raws, Ms, use, {}, ref, None, None, None

    ring_closure_check(Ms, use, cfg)
    Ms, use = consistency_gate_local(raws, Ms, use, cfg)
    if len(use) < 2:
        print('  守门后帧不足, 无法继续')
        return raws, Ms, use, {}, ref, None, None, None

    mf, mc, per_frame_t = Merger(cfg).merge_and_clean(raws, Ms, use)
    scores = leave_one_out_scores(raws, Ms, use, cfg)
    print_scores(scores, cfg)
    return raws, Ms, use, scores, ref, mf, mc, per_frame_t

def save_outputs(outdir, mf, mc, Ms, use, scores, ref, cfg):
    o3d.io.write_point_cloud(os.path.join(outdir, 'merged_cloud.ply'), mf)
    o3d.io.write_point_cloud(os.path.join(outdir, 'merged_cloud_clean.ply'), mc)
    np.savez(os.path.join(outdir, 'transforms.npz'),
             **{('M_%02d' % k): Ms[k] for k in use},
             use_frames=np.array(use), ref_frame=np.int64(ref),
             strategy=np.str_(cfg['strategy']),
             score_median=np.array([scores[k][0] for k in use]) if scores else np.array([]), # 兼容旧版: 实际是P95
             score_p95=np.array([scores[k][0] for k in use]) if scores else np.array([]),
             score_overlap=np.array([scores[k][1] for k in use]) if scores else np.array([]))


# ================================================================
#  v9.3 定量精度审计：不修改 Ms / use，不作为自动剔帧依据
# ================================================================
def _metric_percentile(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else None


def _metric_stats(distances):
    a = np.asarray(distances, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(count=0, median=None, mean=None, rmse=None, p90=None, p95=None, max=None)
    return dict(count=int(len(a)), median=float(np.median(a)), mean=float(a.mean()),
                rmse=float(np.sqrt(np.mean(a * a))), p90=_metric_percentile(a, 90),
                p95=_metric_percentile(a, 95), max=float(a.max()))


def _metric_sample_world(raws, Ms, use, cfg):
    """只为评分制作确定性的点云子集，不动原配准数据和变换。"""
    voxel = float(cfg.get('metrics_voxel', cfg['score_voxel']))
    cap = max(100, int(cfg.get('metrics_points_per_frame', 2500)))
    world = {}
    for k in use:
        p = _score_cloud(raws, k, cfg) if voxel == cfg['score_voxel'] else voxel_down(raws[k], voxel)
        P = np.asarray(p.points).copy()
        P = P[np.isfinite(P).all(axis=1)]
        if len(P) > cap:
            # 固定随机种子，可复现采样；避免原始PLY点顺序造成采样偏斜。
            rng = np.random.default_rng(1000003 + int(k))
            P = P[np.sort(rng.choice(len(P), size=cap, replace=False))]
        M = np.asarray(Ms[k], dtype=float)
        world[k] = P @ M[:3, :3].T + M[:3, 3]
    return world


def _metric_directed(src, target_tree, threshold):
    """只统计给定阈值内的跨帧对应点；未匹配点不被当作大配准误差。"""
    if target_tree is None or len(src) == 0:
        return np.zeros(0, dtype=float), 0.0, 0
    d, _ = target_tree.query(src, k=1, distance_upper_bound=threshold, workers=1)
    matched = d[np.isfinite(d)]
    return matched, float(len(matched) / len(src)), int(len(src))


def _metric_rotation_degrees(R):
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _metric_saved_edges(path, files, use_set):
    if not os.path.isfile(path):
        return []
    result = []
    try:
        with open(path, 'r', encoding='utf-8-sig', newline='') as fh:
            for r in csv.DictReader(fh):
                try:
                    i, j = int(r['i']), int(r['j'])
                    if i not in use_set or j not in use_set or not int(r['valid']):
                        continue
                    # 防止引用不同数据目录下残留的旧审计CSV。
                    if (r.get('file_i') != os.path.basename(files[i]) or
                        r.get('file_j') != os.path.basename(files[j])):
                        continue
                    T = np.asarray(json.loads(r['T_ij_rowmajor']), dtype=float).reshape(4, 4)
                    if not np.isfinite(T).all():
                        continue
                    result.append((i, j, T, float(r['median']), float(r['overlap'])))
                except (KeyError, ValueError, IndexError, TypeError):
                    continue
    except (OSError, UnicodeError) as ex:
        print('  [metrics] 边审计读取失败：%s' % ex)
    return result


def _metrics_triangles(edge_records, outdir):
    """对原始pair测量做真正的三边闭环检查，不使用文件序号推断闭环。"""
    E = {}
    nbr = {}
    for i, j, T, _, _ in edge_records:
        key = (min(i, j), max(i, j))
        if key in E:
            continue
        E[key] = T if i < j else np.linalg.inv(T)
        nbr.setdefault(i, set()).add(j)
        nbr.setdefault(j, set()).add(i)

    def trans(a, b):
        M = E[min(a, b), max(a, b)]
        return M if a < b else np.linalg.inv(M)

    path = os.path.join(outdir, 'registration_metrics_cycles.csv')
    rows = []
    for i in sorted(nbr):
        for j in sorted(x for x in nbr[i] if x > i):
            for k in sorted(x for x in nbr[i].intersection(nbr.get(j, set())) if x > j):
                D = np.linalg.inv(trans(i, k)) @ (trans(j, k) @ trans(i, j))
                rows.append((i, j, k, float(np.linalg.norm(D[:3, 3])),
                             _metric_rotation_degrees(D[:3, :3])))
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['i', 'j', 'k', 'translation_closure_input_units', 'rotation_closure_deg'])
        w.writerows(rows)
    return rows


def _metrics_external_reference(world, cfg):
    path = str(cfg.get('metrics_reference_ply', '') or '')
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError('参考点云不存在: %s' % path)
    ref = o3d.io.read_point_cloud(path)
    rp = np.asarray(ref.points)
    if len(rp) == 0:
        raise ValueError('参考PLY没有点: %s' % path)
    # 只用于独立评价：要求参考PLY原本已与结果在同一坐标系，不对它自动ICP。
    ref = voxel_down(ref, float(cfg.get('metrics_voxel', 0.5)))
    target = cKDTree(np.asarray(ref.points))
    P = np.vstack([world[k] for k in sorted(world) if len(world[k])])
    d, _ = target.query(P, k=1, workers=1)
    return {'reference_path': os.path.abspath(path), 'sampled_point_count': len(P),
            'one_way_distances': _metric_stats(d),
            'within_overlap_threshold': float((d <= float(cfg.get('metrics_overlap_dist', 1.0))).mean()),
            'alignment_assumption': 'reference model already shares the same world coordinate system'}


def write_registration_metrics(raws, Ms, use, files, cfg, outdir):
    """逐帧、逐边、闭环、汇总；所有量只读，不调用ICP也不修改配准结果。"""
    if not cfg.get('metrics_enable', True) or len(use) < 2:
        return None
    os.makedirs(outdir, exist_ok=True)
    use = sorted(int(k) for k in use)
    threshold = float(cfg.get('metrics_overlap_dist', cfg['overlap_dist']))
    if threshold <= 0:
        raise ValueError('metrics_overlap_dist必须大于0')
    unit = str(cfg.get('metrics_unit', 'mm'))
    print('  [metrics] 以 %.3f %s 为重叠阈值，对 %d 帧做独立定量评估（不修改位姿）...'
          % (threshold, unit, len(use)))
    world = _metric_sample_world(raws, Ms, use, cfg)
    frame_trees = {k: cKDTree(P) for k, P in world.items() if len(P)}
    all_matched = []
    cross_distances = _cross_frame_distances(world)
    frame_rows = []
    bbox = {k: (P.min(axis=0), P.max(axis=0)) for k, P in world.items() if len(P)}

    for k in use:
        P = world[k]
        # AABB加阈值余量的保守预筛，不遗漏阈值内的真实近邻；不依赖帧号连续。
        if k not in bbox:
            frame_rows.append([k, os.path.basename(files[k]), 0, 0, 0., 1., '', '', '', '', '', '', 0])
            continue
        b0, b1 = bbox[k]
        candidates = [j for j in use if j != k and j in bbox
                      and np.all(bbox[j][0] <= b1 + threshold)
                      and np.all(bbox[j][1] >= b0 - threshold)]
        d = cross_distances[k]
        matched = d[d < threshold]
        overlap, nsrc = float(len(matched)/max(len(P), 1)), len(P)
        st = _metric_stats(matched)
        all_matched.append(matched)
        frame_rows.append([k, os.path.basename(files[k]), nsrc, st['count'],
                           overlap, 1.0-overlap, st['median'], st['mean'], st['rmse'],
                           st['p90'], st['p95'], st['max'], len(candidates)])

    frame_path = os.path.join(outdir, 'registration_metrics_frames.csv')
    with open(frame_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'file', 'sample_count', 'matched_count',
                    'cross_frame_overlap_fraction', 'unmatched_fraction',
                    'matched_median_'+unit, 'matched_mean_'+unit,
                    'matched_rmse_'+unit, 'matched_p90_'+unit,
                    'matched_p95_'+unit, 'matched_max_'+unit, 'candidate_neighbor_frames'])
        w.writerows(frame_rows)

    edge_path = os.path.join(outdir, 'registration_metrics_edges.csv')
    edge_records = _metric_saved_edges(os.path.join(outdir, 'pairwise_edge_audit.csv'),
                                       files, set(use)) if cfg.get('metrics_edges_enable', True) else []
    edge_rows = []
    all_edge_matched = []
    for i, j, T, original_med, original_ov in edge_records:
        if i not in frame_trees or j not in frame_trees:
            continue
        dij, oij, _ = _metric_directed(world[i], frame_trees[j], threshold)
        dji, oji, _ = _metric_directed(world[j], frame_trees[i], threshold)
        both = np.concatenate([dij, dji])
        all_edge_matched.append(both)
        st = _metric_stats(both)
        # T_ij和最终位姿导出的相对变换的一致性：不是几何真值误差。
        try:
            predicted = np.linalg.inv(Ms[j]) @ Ms[i]
            D = np.linalg.inv(T) @ predicted
            terr, rerr = float(np.linalg.norm(D[:3, 3])), _metric_rotation_degrees(D[:3, :3])
        except np.linalg.LinAlgError:
            terr, rerr = None, None
        edge_rows.append([i, j, original_med, original_ov, oij, oji,
                          min(oij, oji), st['count'], st['median'], st['rmse'],
                          st['p95'], terr, rerr])
    with open(edge_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['i', 'j', 'original_pair_median', 'original_pair_overlap',
                    'overlap_i_to_j', 'overlap_j_to_i', 'bidirectional_min_overlap',
                    'matched_count', 'symmetric_matched_median_'+unit,
                    'symmetric_matched_rmse_'+unit, 'symmetric_matched_p95_'+unit,
                    'posegraph_translation_residual_'+unit, 'posegraph_rotation_residual_deg'])
        w.writerows(edge_rows)

    cycles = (_metrics_triangles(edge_records, outdir)
              if cfg.get('metrics_triangles_enable', True) else [])
    all_vals = np.concatenate([a for a in all_matched if len(a)]) if any(len(a) for a in all_matched) else np.array([])
    fcount = int(sum(int(row[2]) for row in frame_rows))
    mcount = int(sum(int(row[3]) for row in frame_rows))
    summary = {
        'description': 'sampled cross-frame nearest-neighbor registration residual (NOT absolute accuracy)',
        'input_frame_count': len(files), 'registered_frame_count': len(use),
        'units_assumed': unit, 'source_voxel_input_units': float(cfg.get('metrics_voxel', 0.5)),
        'overlap_threshold_input_units': threshold,
        'sample_cap_per_frame': int(cfg.get('metrics_points_per_frame', 2500)),
        'sampled_points_total': fcount, 'matched_points_total': mcount,
        'cross_frame_overlap_weighted': mcount / max(fcount, 1),
        'cross_frame_overlap_frame_mean': float(np.mean([float(r[4]) for r in frame_rows])),
        'matched_only_distance': _metric_stats(all_vals),
        'registered_pair_count_with_metrics': len(edge_rows),
        'pairwise_matched_only_distance': _metric_stats(np.concatenate([a for a in all_edge_matched if len(a)])
                                                       if any(len(a) for a in all_edge_matched) else np.array([])),
        'cycle_triangle_count': len(cycles),
        'cycle_translation_error': _metric_stats([c[3] for c in cycles]),
        'cycle_rotation_error_deg': _metric_stats([c[4] for c in cycles]),
        'warnings': [
            'NN distances are precision proxies; no ground-truth accuracy is established.',
            'Metrics are measured on downsampled per-frame point subsets.',
            'Matched residuals exclude points outside the overlap threshold; consult unmatched_fraction.',
            'Symmetric pairwise metrics only use valid original pairwise edges still present in use.',
            'Repeated, symmetrical or featureless surfaces may show small NN residuals even when misregistered.',
            'Input coordinate units are user-assumed; confirm PLY really uses '+unit+'.'
        ]
    }
    if cfg.get('metrics_reference_ply'):
        summary['aligned_reference_comparison'] = _metrics_external_reference(world, cfg)
    summary_path = os.path.join(outdir, 'registration_metrics_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    report_path = os.path.join(outdir, 'registration_metrics_README.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('配准定量评估（仅诊断，不改变已得到的位姿）\n')
        f.write('成功帧 %d/%d；各帧样本总数 %d\n' % (len(use), len(files), fcount))
        f.write('跨帧样本重叠率: %.2f%%（阈值 %.3f %s）\n'
                % (100*summary['cross_frame_overlap_weighted'], threshold, unit))
        d = summary['matched_only_distance']
        f.write('重叠区样本最近邻残差: median=%s, RMSE=%s, P95=%s %s\n'
                % (d['median'], d['rmse'], d['p95'], unit))
        f.write('有效边审计 %d 对；三角闭环 %d 个\n' % (len(edge_rows), len(cycles)))
        f.write('注意：残差不等于实物尺寸误差，也不等于绝对精度；无重叠部分不应被当作配准失败。\n')
        f.write('查看 registration_metrics_frames.csv 找低重叠或残差大的帧；\n')
        f.write('查看 registration_metrics_edges.csv 找相互矛盾的配准边；\n')
        f.write('查看 registration_metrics_cycles.csv 找闭环不一致；\n')
        f.write('数据单位请自行校验，当前标称为 %s。\n' % unit)
    print('  [metrics] 重叠率 %.1f%%；重叠内 median=%s RMSE=%s P95=%s %s'
          % (100*summary['cross_frame_overlap_weighted'],
             summary['matched_only_distance']['median'],
             summary['matched_only_distance']['rmse'],
             summary['matched_only_distance']['p95'], unit))
    print('  [metrics] 报告: %s' % summary_path)
    return summary


def write_run_manifest(files, cfg, outdir, use, ref, elapsed_s=None):
    """记录可复现实验设置。只记录大小和修改时间，不重新Hash所有大型PLY。"""
    payload = {
        'script_version': 'v9_6_4_geometry_cuda64',
        'created_at_local': time.strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': sys.version,
        'open3d_version': getattr(o3d, '__version__', 'unknown'),
        'input_folder': os.path.abspath(os.path.dirname(files[0])) if files else '',
        'inputs': [{'name': os.path.basename(p), 'size_bytes': os.stat(p).st_size,
                    'mtime_ns': os.stat(p).st_mtime_ns} for p in files],
        'config': {k: v for k, v in cfg.items() if not k.startswith('_')},
        'accepted_frames': sorted([int(k) for k in use]), 'reference_frame': int(ref),
        'elapsed_seconds': elapsed_s
    }
    path = os.path.join(outdir, 'registration_run_manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path

def remerge(files, cfg, outdir, exclude):
    data = np.load(os.path.join(outdir, 'transforms.npz'), allow_pickle=True)
    use = [int(k) for k in data['use_frames']]
    Ms = {k: data['M_%02d' % k] for k in use}
    print('[remerge] 加载 %d 帧位姿' % len(use))
    raws = [o3d.io.read_point_cloud(f) for f in files]
    if exclude:
        ex = set(exclude)
        use = [k for k in use if k not in ex]
        Ms = {k: Ms[k] for k in use}
        print('[remerge] 排除帧 %s, 剩 %d 帧' % (sorted(ex), len(use)))
    ring_closure_check(Ms, use, cfg)
    Ms, use = consistency_gate_local(raws, Ms, use, cfg)
    mf, mc, per_frame_t = Merger(cfg).merge_and_clean(raws, Ms, use)
    scores = leave_one_out_scores(raws, Ms, use, cfg)
    print_scores(scores, cfg)
    save_outputs(outdir, mf, mc, Ms, use, scores, -1, cfg)
    export_by_frame(raws, Ms, use, cfg, outdir)
    write_registration_metrics(raws, Ms, use, files, cfg, outdir)
    write_run_manifest(files, cfg, outdir, use, -1)
    write_performance_report(cfg, files, use, outdir)
    print('输出目录: %s' % outdir)
    return mf, mc

def main():
    ap = argparse.ArgumentParser(description='多视角点云融合 v9.6.4: 四球几何引导与CUDA双精度')
    ap.add_argument('folder', help='存放 ply 的文件夹')
    ap.add_argument('--backend',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--sphere-guide',default=None,help='Measured scan of the same object with four spherical markers; enables geometry-guided registration')
    ap.add_argument('--sphere-radius',type=float,default=5.8,help='Approximate marker radius for detection, in input units')
    ap.add_argument('--calibrate-depth',action='store_true',help='Opt in to independently validated global depth calibration; changes output input geometry')
    ap.add_argument('--surface-guide',action='store_true',help='Guide contains the same complete static object; also constrain actual surface geometry')
    ap.add_argument('--surface-reference',default=None,help='Optional measured surface model of the same static object, already in sphere-guide coordinates')
    ap.add_argument('--sphere-output-frame',type=int,default=None,help='Optional registered input frame defining output coordinates')
    ap.add_argument('--strategy', choices=['multiway', 'star'], default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--set', nargs='*', action='extend', default=[], metavar='KEY=VALUE')
    manual_mode = ap.add_mutually_exclusive_group()
    manual_mode.add_argument('--manual-bridge', action='store_true', help='自动接入失败后允许手动选点架桥；默认关闭，人工等待计入总耗时')
    manual_mode.add_argument('--no-rescue','--no-manual-rescue', action='store_true',dest='no_rescue',help='关闭手动点选救援；自动接入仍然运行')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--exclude', nargs='*', type=int, default=[])
    ap.add_argument('--remerge', action='store_true')
    ap.add_argument('--scan-model', default=None,
                    help='完整口扫模型路径: 沿用旧版模型锚定流程')
    ap.add_argument('--reuse-pairs', action='store_true',
                    help='从匹配文件/参数的缓存读取已完成的pair，避免重算RANSAC')
    ap.add_argument('--no-precheck', action='store_true',
                    help='跳过启动前PLY完整性检查（不建议）')
    ap.add_argument('--precheck-only', action='store_true',
                    help='只检查PLY文件完整性后退出')
    main_mode = ap.add_mutually_exclusive_group()
    main_mode.add_argument('--source-main', action='store_true',
                           help='以组内 BFS 构建主图，默认启用')
    main_mode.add_argument('--global-main', action='store_true',
                           help='沿用旧的全局位姿图构建初始主图，用于对照')
    ap.add_argument('--seed-frame', type=int, default=None,
                    help='选取包含指定帧号的源分量作为主图；可用于选择已知质量较好的源分量')
    ap.add_argument('--seed-component', type=int, default=None,
                    help='按分量大小排序(0=最大)，选择指定编号的源分量')
    ap.add_argument('--initial-only', action='store_true',
                    help='只运行初始pairwise + GlobalOptimizer + 合并，不进入救援/守门')
    ap.add_argument('--snapshot-initial', action='store_true',
                    help='导出全局PoseGraph+multiway后、rescue前的单帧着色点云')
    ap.add_argument('--metrics-only', action='store_true',
                    help='直接用现有transforms.npz生成定量误差报告；不重新配准或改动点云')
    ap.add_argument('--no-metrics', action='store_true', help='禁用最后的独立定量报告')
    ap.add_argument('--reference-ply', default=None,
                    help='仅审计用：已在相同坐标系的参考PLY，不进行额外对齐')
    polish = ap.add_mutually_exclusive_group()
    polish.add_argument('--freeze-main', action='store_true',
                        help='冻结初始主图（默认），跳过救援后的全体抛光')
    polish.add_argument('--rescue-polish', action='store_true',
                        help='允许救援后的全体抛光，可能改变主图和手动确认的位姿')
    ap.add_argument('--gate-enforce', action='store_true',
                    help='启用修复自身匹配错误后的自动守门（可能改变最终结果）')
    ap.add_argument('--gate-legacy', action='store_true',
                    help='按旧版原样运行守门，以便A/B对照；含自身匹配缺陷')
    ap.add_argument('--gate-off', action='store_true', help='彻底关闭局部守门')
    ap.add_argument('--output-dir', default=None, help='Output directory; keeps prior results intact')
    ap.add_argument('--refine-existing', default=None,
                    help='Directory with existing transforms.npz and manifest; excludes original pair registration time')
    ap.add_argument('--prepared-dir', default=None, help='Optional compatible prepared_frames cache directory')
    ap.add_argument('--baseline-cache', default=None,
                    help='Explicitly reuse original v9.3 pair results for graph-only A/B; not a cold-start benchmark')
    ap.add_argument('--jobs', type=int, default=None,
                    help='并行进程数(默认 CPU核数-1; 1=回退顺序模式)')
    args = ap.parse_args()

    print('[fusion v9.6.4 geometry-guided CUDA float64] 组内配准 + 分量刚体接入')
    cfg = dict(DEFAULT_CFG)
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding='utf-8') as f:
            cfg.update(json.load(f))
    if args.strategy:
        cfg['strategy'] = args.strategy
    if args.scan_model:
        cfg['scan_model_path'] = args.scan_model
    if args.jobs is not None:
        cfg['n_jobs'] = args.jobs
    if args.reuse_pairs:
        cfg['reuse_pairs'] = True
    if args.no_metrics:
        cfg['metrics_enable'] = False
    if args.reference_ply:
        cfg['metrics_reference_ply'] = args.reference_ply
    if args.freeze_main or args.rescue_polish:
        cfg['freeze_seed_main'] = args.freeze_main
    if args.gate_enforce:
        cfg['gate_mode'] = 'enforce'
    if args.gate_legacy:
        cfg['gate_mode'] = 'legacy'
    if args.gate_off:
        cfg['gate_mode'] = 'off'
    if sum((args.gate_enforce, args.gate_legacy, args.gate_off)) > 1:
        ap.error('--gate-enforce / --gate-legacy / --gate-off 不能同时使用')
    if args.no_precheck:
        cfg['precheck_ply'] = False
    if args.snapshot_initial:
        cfg['snapshot_initial'] = True
    if args.source_main or args.global_main:
        cfg['source_style_main'] = args.source_main
    if args.seed_frame is not None:
        cfg['source_seed_frame'] = args.seed_frame
    if args.seed_component is not None:
        cfg['source_seed_component'] = args.seed_component
    if args.seed_frame is not None and args.seed_component is not None:
        ap.error('--seed-frame 和 --seed-component 不可同时指定')
    if args.initial_only:
        cfg['initial_only'] = True
    for kv in args.set:
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        cfg[k] = v

    cfg['sphere_guide_path']=args.sphere_guide
    cfg['sphere_radius']=args.sphere_radius
    cfg['calibrate_depth']=args.calibrate_depth
    cfg['surface_guide']=args.surface_guide or bool(args.surface_reference)
    cfg['surface_reference_path']=args.surface_reference
    cfg['sphere_output_frame']=args.sphere_output_frame
    cfg['compute_backend'] = args.backend
    if args.manual_bridge:
        cfg['manual_rescue'] = True
    elif args.no_rescue:
        cfg['manual_rescue'] = False
    cfg['_manual_bridge_frames'] = []
    if cfg.get('manual_rescue', False):
        import importlib.util
        if any(importlib.util.find_spec(name) is None for name in ('pyvista', 'vtk')):
            ap.error('手动架桥需要项目Python中安装PyVista和VTK')
        print('[manual-bridge] 已开启：自动救援失败后询问是否手动选点；人工等待计入总耗时', flush=True)
    if args.refine_existing and args.backend=='cuda':
        cfg.setdefault('gpu_frame_refine',True)
    _validate_cuda_startup(cfg)
    cfg['_started_at'] = _SCRIPT_STARTED_AT
    cfg['_baseline_cache'] = args.baseline_cache
    cfg['_refine_existing'] = args.refine_existing
    cfg['_prepared_dir'] = args.prepared_dir
    files = _geometry_input_files(args.folder)
    assert files, '文件夹内没有 ply'
    outdir = args.output_dir or os.path.join(args.folder, 'multiview_output_v964')
    os.makedirs(outdir, exist_ok=True)
    cfg['_outdir'] = outdir
    if args.remerge or args.metrics_only:
        stored = Path(outdir, 'transforms.npz')
        if stored.exists():
            with np.load(stored, allow_pickle=False) as saved:
                if 'depth_parameters' in saved:
                    ap.error('Calibrated outputs require depth correction before rigid transforms; use full registration for this version')
    if cfg.get('freeze_seed_main', False) and cfg.get('gate_mode') in ('enforce', 'legacy'):
        ap.error('冻结初始主图与会移动帧的守门模式冲突，请使用默认 audit 或显式启用 --rescue-polish')

    if args.metrics_only:
        if args.no_metrics:
            ap.error('--metrics-only 不能与 --no-metrics 同时使用')
        if args.remerge or args.initial_only or args.exclude:
            ap.error('--metrics-only 不支持 --remerge / --initial-only / --exclude')
        transform_file = os.path.join(outdir, 'transforms.npz')
        if not os.path.isfile(transform_file):
            ap.error('找不到已保存的 %s；请先运行原程序并生成位姿' % transform_file)
        # 防止文件列表变更后仍用旧帧索引评估错误PLY。
        manifest_path = os.path.join(outdir, 'registration_run_manifest.json')
        if os.path.isfile(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as _fh:
                prior = json.load(_fh)
            prior_names = [v['name'] for v in prior.get('inputs', [])]
            if prior_names and prior_names != [os.path.basename(f) for f in files]:
                ap.error('当前PLY文件列表/顺序和已保存的manifest不同，不能直接用旧位姿评估')
        else:
            print('[metrics-only] 注意：旧结果没有文件列表manifest，请确认PLY的顺序与产生位姿时完全一致')
        data = np.load(transform_file, allow_pickle=False)
        use = [int(k) for k in data['use_frames']]
        Ms = {k: np.asarray(data['M_%02d' % k]).copy() for k in use}
        raws = LazyFrameLoader(files, capacity=cfg.get('lazy_capacity', 48))
        write_registration_metrics(raws, Ms, use, files, cfg, outdir)
        print('[metrics-only] 已基于现有位姿生成报告；未执行任何ICP/配准/融合')
        return

    if args.precheck_only:
        bad = precheck_point_cloud_files(files)
        if bad:
            raise SystemExit('PLY预检失败：%d个文件异常，请先修复，不会跳过以免帧索引错位' % len(bad))
        print('[precheck] 所有文件通过')
        return

    if cfg.get('precheck_ply', True) and not args.remerge and not cfg.get('fused_precheck', True):
        bad = precheck_point_cloud_files(files)
        if bad:
            raise SystemExit('PLY预检失败：%d个文件异常，请先修复，不会跳过以免帧索引错位' % len(bad))

    if args.remerge:
        mf, mc = remerge(files, cfg, outdir, args.exclude)
        if args.show:
            try:
                o3d.visualization.draw_geometries([mc], window_name='合并结果(干净版)')
            except Exception as e:
                print('(无显示环境: %s)' % e)
        return

    t_main_begin = cfg['_started_at']
    raws, Ms, use, scores, ref, mf, mc, per_frame_t = run(files, cfg)
    if mf is None:
        sys.exit(2)

    if args.exclude:
        ex = set(args.exclude)
        use = [k for k in use if k not in ex]
        Ms = {k: Ms[k] for k in use}
        print('[exclude] 排除帧 %s, 重新合并 (%d 帧)' % (sorted(ex), len(use)))
        mf, mc, per_frame_t = Merger(cfg).merge_and_clean(raws, Ms, use)
        scores = leave_one_out_scores(raws, Ms, use, cfg)
        print_scores(scores, cfg)

    save_outputs(outdir, mf, mc, Ms, use, scores, ref, cfg)
    export_by_frame(raws, Ms, use, cfg, outdir)
    write_registration_metrics(raws, Ms, use, files, cfg, outdir)
    write_run_manifest(files, cfg, outdir, use, ref, time.monotonic()-t_main_begin)
    write_performance_report(cfg, files, use, outdir)
    print('输出目录: %s' % outdir)

    if args.show:
        try:
            o3d.visualization.draw_geometries([mc], window_name='合并结果(干净版)')
        except Exception as e:
            print('(无显示环境, 跳过可视化: %s)' % e)

# ================================================================
# v9.4: exact nearest-neighbor acceleration and checked loop refinement
# ================================================================
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

DEFAULT_CFG.update(
    lazy_capacity=2, worker_raw_capacity=2, worker_max_tasks=0,
    pair_tree_cache_capacity=2, prepared_cache_capacity=3,
    graph_refine=True, graph_loss_scale=0.15, graph_max_iter=60,
    graph_sample_cap=2500, graph_geom_tolerance=1e-6,
    random_seed=1729, pair_fast_init=True,
    fused_precheck=True, time_target_seconds=300.,
    shared_preprocess=True, max_auto_jobs=8,
)

_ICP_TREE_CACHE = OrderedDict()


def _target_tree(pcd):
    key = id(pcd)
    pts = np.asarray(pcd.points)
    if key not in _ICP_TREE_CACHE or not np.array_equal(_ICP_TREE_CACHE[key][1].data, pts):
        _ICP_TREE_CACHE[key] = (pcd, cKDTree(pts, copy_data=True))
    _ICP_TREE_CACHE.move_to_end(key)
    while len(_ICP_TREE_CACHE) > 4:
        _ICP_TREE_CACHE.popitem(last=False)
    return _ICP_TREE_CACHE[key][1]


def p2plane_step(P, Q, Qn, keep_ratio=1.0, huber=1.345):
    """Centred SE(3) update; suppress directions unobservable on flat surfaces."""
    r = np.einsum('ij,ij->i', P - Q, Qn)
    if keep_ratio < 1.:
        count = max(6, int(len(r) * keep_ratio))
        take = np.argpartition(np.abs(r), min(count, len(r))-1)[:count]
        P, Qn, r = P[take], Qn[take], r[take]
    if len(r) < 6:
        return np.eye(4)
    scale = huber * np.abs(r).mean()
    if scale < 1e-12:
        return np.eye(4)
    w = np.minimum(1., scale / np.maximum(np.abs(r), 1e-30))
    pivot = np.average(P, axis=0, weights=w)
    A = np.column_stack((np.cross(P - pivot, Qn), Qn))
    H = A.T @ (w[:, None] * A)
    b = -A.T @ (w * r)
    norm = np.sqrt(np.maximum(np.diag(H), 1e-20))
    Hn = H / norm[:, None] / norm[None, :]
    eig, vec = np.linalg.eigh(Hn)
    observable = eig > max(eig[-1] * 1e-8, 1e-12)
    inv_eig = np.zeros(6)
    inv_eig[observable] = 1. / eig[observable]
    dx = (vec @ (inv_eig * (vec.T @ (b / norm)))) / norm
    R = Rotation.from_rotvec(dx[:3]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pivot + dx[3:] - R @ pivot
    return T


def icp_custom(src, tgt, thr, keep_ratio=0.6, max_iter=60, conv=1e-4, T_init=None):
    src_pts = np.asarray(src.points)
    if not len(src_pts) or not len(tgt.points):
        return (np.eye(4) if T_init is None else T_init.copy()), np.inf
    if not tgt.has_normals():
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=thr * 2, max_nn=30))
    if thr <= 2. and not src.has_normals():
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=thr * 2, max_nn=30))
    tgt_pts, tgt_n = np.asarray(tgt.points), np.asarray(tgt.normals)
    tree = _target_tree(tgt)
    T = np.eye(4) if T_init is None else T_init.copy()
    prev = np.inf
    for _ in range(max_iter):
        P = src_pts @ T[:3, :3].T + T[:3, 3]
        # eps=0: exact search, same correspondences as unrestricted query then dist<thr.
        dist, idx = tree.query(P, k=1, distance_upper_bound=thr, eps=0.)
        m = np.isfinite(dist)
        safe_idx = np.minimum(idx, len(tgt_pts)-1)
        if thr <= 2.:
            sn = np.asarray(src.normals) @ T[:3, :3].T
            m &= np.abs(np.einsum('ij,ij->i', sn, tgt_n[safe_idx])) > .5
        valid = np.flatnonzero(m)
        if len(valid) < 10:
            break
        ds = dist[valid]
        count = min(len(ds), max(10, int(len(ds) * keep_ratio)))
        cutoff = np.partition(ds, count-1)[count-1]
        valid = valid[ds <= cutoff]
        Pk, Qk, Nk = P[valid], tgt_pts[idx[valid]], tgt_n[idx[valid]]
        delta = p2plane_step(Pk, Qk, Nk)
        if not np.isfinite(delta).all():
            break
        cur = np.abs(np.einsum('ij,ij->i', Pk-Qk, Nk)).mean()
        T = delta @ T
        if abs(prev-cur) < conv:
            prev = cur
            break
        prev = cur
    return T, float(prev)


def _prepared_key(path, cfg):
    st = os.stat(path)
    values = [os.path.abspath(path), st.st_size, st.st_mtime_ns,
              getattr(o3d, '__version__', ''), 'prepared-940-1',
              {k: cfg[k] for k in ('voxel_fpfh', 'fpfh_radius', 'icp_voxel',
                                   'score_voxel', 'hp_voxel', 'hp_norm_radius')}]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _prepared_path(path, cfg):
    directory = cfg.get('_prepared_dir') or os.path.join(cfg['_outdir'], 'prepared_frames')
    return os.path.join(directory, _prepared_key(path, cfg)+'.npz')


def _score_cloud(raws, k, cfg):
    files = cfg.get('_files')
    if files:
        path = _prepared_path(files[k], cfg)
        if os.path.isfile(path):
            with np.load(path, allow_pickle=False) as data:
                p = _cloud_from_arrays(data['score_p'])
                colors = data['score_c']
                if len(colors):
                    p.colors = o3d.utility.Vector3dVector(colors)
                return p
    return voxel_down(raws[k], cfg['score_voxel'])


def _prepare_frame(task):
    path, cfg = task
    dest = _prepared_path(path, cfg)
    if os.path.isfile(dest):
        try:
            with np.load(dest, allow_pickle=False) as data:
                if 'bounds' in data and 'hp_n' in data:
                    return data['bounds'].copy()
        except (OSError, ValueError, EOFError):
            pass
    raw = safe_read_point_cloud(path)
    if cfg.get('compute_backend')=='cuda' and cfg.get('gpu_pipeline',True):
        _save_raw_gpu_cache(path,raw,cfg)
    if len(raw.points) < 10:
        raise ValueError('Too few points: '+path)
    sd, sf = fpfh_feats(raw, cfg['voxel_fpfh'], cfg['fpfh_radius'])
    mid = voxel_down(raw, cfg['icp_voxel'])
    mid.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4., max_nn=30))
    score = voxel_down(raw, cfg['score_voxel'])
    hp = voxel_down(raw, cfg['hp_voxel'])
    hp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg['hp_norm_radius'], max_nn=50))
    box = raw.get_axis_aligned_bounding_box()
    bounds = np.stack((box.get_min_bound(), box.get_max_bound()))
    dest_tmp = dest + '.%d.tmp.npz' % os.getpid()
    np.savez(dest_tmp, feat_p=np.asarray(sd.points), feat_n=np.asarray(sd.normals),
             feat_f=np.asarray(sf.data), mid_p=np.asarray(mid.points),
             mid_n=np.asarray(mid.normals), score_p=np.asarray(score.points),
             score_c=np.asarray(score.colors), hp_p=np.asarray(hp.points),
             hp_n=np.asarray(hp.normals), bounds=bounds)
    os.replace(dest_tmp, dest)
    return bounds


def _auto_jobs(cfg):
    requested = int(cfg.get('n_jobs', -1))
    if requested > 0:
        return requested
    jobs = max(1, min(int(cfg.get('max_auto_jobs', 8)), (os.cpu_count() or 2)-1))
    try:
        import psutil
        # Allow roughly 450MB/worker, reserving 900MB for the coordinator.
        jobs = min(jobs, max(1, int((psutil.virtual_memory().available-900_000_000)/450_000_000)))
    except ImportError:
        jobs = min(jobs, 4)
    return jobs


_original_register = PairwiseRegistrator.register_all_pairs
_original_feats = PairwiseRegistrator._feats
_original_mid = PairwiseRegistrator._icp_cloud
_original_hp = PairwiseRegistrator._icp_cloud_hp
_original_align = PairwiseRegistrator._align_pair


def _get_prepared(self, k):
    if not hasattr(self, '_prepared'):
        self._prepared = OrderedDict()
    if k not in self._prepared:
        files = self.cfg.get('_files')
        if not files:
            return None
        path = _prepared_path(files[k], self.cfg)
        if not os.path.isfile(path):
            return None
        with np.load(path, allow_pickle=False) as data:
            self._prepared[k] = {key: data[key] for key in data.files if key not in ('score_c',)}
    self._prepared.move_to_end(k)
    while len(self._prepared) > int(self.cfg.get('prepared_cache_capacity', 3)):
        self._prepared.popitem(last=False)
    return self._prepared[k]


def _cloud_from_arrays(points, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def _prepared_feats(self, raws, k):
    data = _get_prepared(self, k)
    if data is None:
        return _original_feats(self, raws, k)
    if k not in self._feat:
        f = o3d.pipelines.registration.Feature()
        f.data = data['feat_f']
        self._feat[k] = _cloud_from_arrays(data['feat_p'], data['feat_n']), f
    self._feat.move_to_end(k)
    while len(self._feat) > 24:
        self._feat.popitem(last=False)
    return self._feat[k]


def _prepared_cloud(self, raws, k, high=False):
    data = _get_prepared(self, k)
    if data is None:
        return (_original_hp if high else _original_mid)(self, raws, k)
    cache = self._hp if high else self._icp
    prefix = 'hp' if high else 'mid'
    if k not in cache:
        cache[k] = _cloud_from_arrays(data[prefix+'_p'], data[prefix+'_n'])
    cache.move_to_end(k)
    while len(cache) > (2 if high else 12):
        cache.popitem(last=False)
    return cache[k]


def _seeded_align(self, raws, i, j):
    o3d.utility.random.seed((int(self.cfg.get('random_seed', 1729)) + 1000003*i + j) % 2147483647)
    return _original_align(self, raws, i, j)


def _pair_initial_transform(self, raws, i, j, sd, sf, td, tf):
    cfg = self.cfg
    if cfg.get('pair_fast_init', True):
        try:
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                sd, td, sf, tf, o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=cfg['ransac_dist']))
            T, _ = icp_custom(self._icp_cloud(raws, i), self._icp_cloud(raws, j),
                              2., .8, max_iter=80, T_init=np.asarray(result.transformation))
            med, ov = self._evaluate_cached(raws, i, j, T, cfg['score_voxel'], cfg['overlap_dist'])
            rev_med, rev_ov = self._evaluate_cached(raws, j, i, np.linalg.inv(T), cfg['score_voxel'], cfg['overlap_dist'])
            # Accelerated initializer is used only with strong symmetric support.
            # The entire original 0.2mm refinement still follows this initializer.
            if max(med, rev_med) <= .4 and min(ov, rev_ov) >= .55:
                return T
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
    o3d.utility.random.seed((int(cfg.get('random_seed', 1729)) + 1000003*i + j) % 2147483647)
    return ransac_init(sd, sf, td, tf, cfg['ransac_dist'], cfg['ransac_iter'])


def _cached_register(self, raws, files=None):
    cfg = self.cfg
    cfg['_files'] = files
    # Explicit diagnostic path: old pair transformations, without claiming they are new cached results.
    if cfg.get('_baseline_cache'):
        with open(cfg['_baseline_cache'], encoding='utf8') as fp:
            head = json.loads(next(fp))
            old_files = head.get('spec', {}).get('files', [])
            # Older headers only retain a signature; require adjacent manifest for provenance.
            manifest = Path(cfg['_baseline_cache']).with_name('registration_run_manifest.json')
            if not manifest.exists():
                raise ValueError('Baseline cache requires its original run manifest')
            prior = json.loads(manifest.read_text(encoding='utf8'))
            current = [(os.path.basename(p), os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in files]
            previous = [(v['name'], v['size_bytes'], v['mtime_ns']) for v in prior['inputs']]
            if current != previous:
                raise ValueError('Baseline cache input list/stat does not match current PLYs')
            records = {}
            for line in fp:
                item = json.loads(line)
                if item.get('type') == 'pair' and item.get('status') == 'ok':
                    records[(int(item['i']), int(item['j']))] = item
        edges = [self._make_edge(i, j, np.asarray(d['T']).reshape(4, 4), d['med'], d['ov'], d['consistent'])
                 for (i, j), d in sorted(records.items())]
        self._export_audit(edges, files)
        cfg['_benchmark_kind'] = 'baseline-pairs-graph-only'
        return edges, [e for e in edges if not e.valid and e.consistent]
    if files and cfg.get('shared_preprocess', True):
        import multiprocessing as mp
        jobs = _auto_jobs(cfg)
        cfg['n_jobs'] = jobs
        os.makedirs(os.path.join(cfg['_outdir'], 'prepared_frames'), exist_ok=True)
        cfg['_prepared_cache_files_before_run']=sum(os.path.isfile(_prepared_path(p,cfg)) for p in files)
        start = time.monotonic()
        print('  [prepare] %d frames, %d workers; read/validate/features/normals once' % (len(files), jobs), flush=True)
        if jobs == 1:
            bounds = [_prepare_frame((p, cfg)) for p in files]
        else:
            with mp.get_context('spawn').Pool(jobs) as pool:
                bounds = []
                for b in pool.imap(_prepare_frame, [(p, cfg) for p in files], chunksize=1):
                    bounds.append(b)
                    if len(bounds) % 20 == 0:
                        print('    [prepare] %d/%d %.1fs' % (len(bounds), len(files), time.monotonic()-start), flush=True)
        cfg['_prepare_seconds'] = time.monotonic()-start
        bounds = np.asarray(bounds)
        centers = bounds.mean(axis=1)
        radii = np.linalg.norm(bounds[:, 1]-bounds[:, 0], axis=1)/2
        self._files = files
        # Same candidate set for jobs=1 and jobs>1.
        return self._register_parallel(raws, files, centers, radii)
    return _original_register(self, raws, files)


PairwiseRegistrator._feats = _prepared_feats
PairwiseRegistrator._icp_cloud = lambda self, raws, k: _prepared_cloud(self, raws, k)
PairwiseRegistrator._icp_cloud_hp = lambda self, raws, k: _prepared_cloud(self, raws, k, True)
PairwiseRegistrator._align_pair = _seeded_align
PairwiseRegistrator.register_all_pairs = _cached_register


def _cross_frame_distances(world):
    """Exact leave-one-frame-out NN with ONE labelled tree; no self matches."""
    keys = sorted(world)
    if len(keys) < 2:
        return {k: np.full(len(world[k]), np.inf) for k in keys}
    points = np.vstack([world[k] for k in keys])
    labels = np.concatenate([np.full(len(world[k]), k, np.int32) for k in keys])
    tree = cKDTree(points)
    result = {}
    for k in keys:
        P = world[k]
        dist_out = np.full(len(P), np.inf)
        pending = np.arange(len(P))
        neighbors = min(2, len(points))
        while len(pending) and neighbors:
            # Chunk bounds memory even for many coincident points from the same frame.
            unresolved = []
            chunk = max(1, 200000 // neighbors)
            for start in range(0, len(pending), chunk):
                take = pending[start:start+chunk]
                d, ix = tree.query(P[take], k=neighbors, eps=0.)
                d, ix = np.asarray(d).reshape(len(take), neighbors), np.asarray(ix).reshape(len(take), neighbors)
                d[labels[ix] == k] = np.inf
                best = d.min(axis=1)
                found = np.isfinite(best)
                dist_out[take[found]] = best[found]
                unresolved.append(take[~found])
            pending = np.concatenate(unresolved)
            if neighbors == len(points):
                break
            if neighbors >= 32 and len(pending):
                # Avoid quadratic k-NN work for isolated, dense same-frame patches.
                seeds = [p[::max(1, len(p)//200)] for j, p in world.items() if j != k and len(p)]
                if not seeds:
                    break
                seed = np.vstack(seeds)
                upper = cKDTree(seed).query(P[pending], k=1)[0].max()
                lo = P[pending].min(axis=0) - np.nextafter(upper, np.inf)
                hi = P[pending].max(axis=0) + np.nextafter(upper, np.inf)
                candidate = []
                for j, p in world.items():
                    if j == k or not len(p):
                        continue
                    mask = ((p >= lo) & (p <= hi)).all(axis=1)
                    if mask.any():
                        candidate.append(p[mask])
                if candidate:
                    dist_out[pending] = cKDTree(np.vstack(candidate)).query(P[pending], k=1, eps=0.)[0]
                break
            neighbors = min(neighbors*2, len(points))
        result[k] = dist_out
    return result


def leave_one_out_scores(raws, Ms, use, cfg):
    world = {}
    for k in use:
        p = _score_cloud(raws, k, cfg)
        world[k] = np.asarray(p.points) @ Ms[k][:3, :3].T + Ms[k][:3, 3]
    distances = _cross_frame_distances(world)
    return {k: (float(np.percentile(d, 95)), float((d < cfg['overlap_dist']).mean()))
            for k, d in distances.items()}


def _geometry_score(poses, samples, threshold):
    world = {k: p @ poses[k][:3, :3].T + poses[k][:3, 3] for k, p in samples.items()}
    distances = _cross_frame_distances(world)
    d = np.concatenate(list(distances.values()))
    # Clipped loss also penalises lost coverage instead of hiding unmatched points.
    clipped = np.minimum(d, threshold)
    matched = d[d < threshold]
    return dict(median=float(np.median(clipped)), p95=float(np.percentile(clipped, 95)),
                mean=float(clipped.mean()), overlap=float((d < threshold).mean()),
                matched_median=float(np.median(matched)) if len(matched) else threshold,
                matched_p95=float(np.percentile(matched, 95)) if len(matched) else threshold,
                matched_rmse=float(np.sqrt(np.mean(matched**2))) if len(matched) else threshold)


def refine_component_graph(poses, edges, anchor, cfg):
    valid = [e for e in edges if e.valid and e.i in poses and e.j in poses]
    if len(valid) < len(poses):
        return poses  # No independent loop constraints.
    files = cfg.get('_files')
    if not files:
        return poses
    start = time.monotonic()
    samples = {}
    centers, radii = {}, {}
    cap = int(cfg.get('graph_sample_cap', 2500))
    for k in sorted(poses):
        path = _prepared_path(files[k], cfg)
        if os.path.isfile(path):
            with np.load(path, allow_pickle=False) as data:
                p = data['score_p']
        else:
            p = np.asarray(safe_read_point_cloud(files[k]).voxel_down_sample(cfg['score_voxel']).points)
        centers[k] = p.mean(axis=0)
        radii[k] = max(1., float(np.sqrt(np.mean(np.sum((p-centers[k])**2, axis=1)))))
        if len(p) > cap:
            take = np.random.default_rng(1000003+k).choice(len(p), cap, replace=False)
            p = p[np.sort(take)]
        samples[k] = p
    keys = [k for k in sorted(poses) if k != anchor]
    offsets = {k: i*6 for i, k in enumerate(keys)}
    x0 = np.concatenate([np.r_[Rotation.from_matrix(poses[k][:3, :3]).as_rotvec(), poses[k][:3, 3]] for k in keys])
    def unpack(x):
        out = {anchor: poses[anchor].copy()}
        for k in keys:
            a = offsets[k]
            T = np.eye(4)
            T[:3, :3] = Rotation.from_rotvec(x[a:a+3]).as_matrix()
            T[:3, 3] = x[a+3:a+6]
            out[k] = T
        return out
    weights = np.sqrt([min(10., max(.1, e.overlap/max(.05, e.median))) for e in valid])
    node_keys = sorted(poses)
    node_index = {k:i for i,k in enumerate(node_keys)}
    free_indices = np.array([node_index[k] for k in keys])
    source_indices = np.array([node_index[e.i] for e in valid])
    target_indices = np.array([node_index[e.j] for e in valid])
    rotations0 = np.stack([poses[k][:3, :3] for k in node_keys])
    translations0 = np.stack([poses[k][:3, 3] for k in node_keys])
    edge_rotations = np.stack([e.T_ij[:3, :3] for e in valid])
    edge_centers = np.stack([centers[e.i] for e in valid])
    edge_radii = np.array([radii[e.i] for e in valid])
    expected_centers = np.stack([e.T_ij[:3, :3] @ centers[e.i] + e.T_ij[:3, 3] for e in valid])
    def fun(x):
        parameters = x.reshape(-1, 6)
        rotations = rotations0.copy()
        translations = translations0.copy()
        rotations[free_indices] = Rotation.from_rotvec(parameters[:, :3]).as_matrix()
        translations[free_indices] = parameters[:, 3:]
        Ri, Rj = rotations[source_indices], rotations[target_indices]
        relative_R = np.einsum('nji,njk->nik', Rj, Ri)
        relative_t = np.einsum('nji,nj->ni', Rj, translations[source_indices]-translations[target_indices])
        out = np.empty((len(valid), 6))
        out[:, :3] = np.einsum('nij,nj->ni', relative_R, edge_centers) + relative_t - expected_centers
        delta_R = np.einsum('nji,njk->nik', edge_rotations, relative_R)
        out[:, 3:] = Rotation.from_matrix(delta_R).as_rotvec()*edge_radii[:, None]
        return (out*weights[:, None]).ravel()
    sparsity = lil_matrix((6*len(valid), len(x0)), dtype=np.int8)
    for row, e in enumerate(valid):
        for k in (e.i, e.j):
            if k != anchor:
                a = offsets[k]
                sparsity[6*row:6*row+6, a:a+6] = 1
    fit = least_squares(fun, x0, jac_sparsity=sparsity.tocsr(), method='trf',
                        loss='cauchy', f_scale=float(cfg.get('graph_loss_scale', .15)),
                        x_scale='jac', max_nfev=int(cfg.get('graph_max_iter', 60)),
                        ftol=1e-7, xtol=1e-7, gtol=1e-7)
    candidate = unpack(fit.x)
    before = _geometry_score(poses, samples, cfg['overlap_dist'])
    after = _geometry_score(candidate, samples, cfg['overlap_dist'])
    tol = float(cfg.get('graph_geom_tolerance', .002))
    accepted = (np.isfinite(fit.x).all() and after['median'] <= before['median']+tol
                and after['p95'] <= before['p95']+tol
                and after['mean'] <= before['mean']+tol
                and after['overlap'] >= before['overlap']-.001)
    report = dict(accepted=bool(accepted), before=before, after=after,
                  edge_rms_before=float(np.sqrt(np.mean(fun(x0)**2))),
                  edge_rms_after=float(np.sqrt(np.mean(fun(fit.x)**2))),
                  elapsed_seconds=time.monotonic()-start, iterations=fit.nfev,
                  anchor=anchor, frames=len(poses), edges=len(valid),
                  note='Sampled geometric safeguard; not a guarantee of absolute accuracy.')
    Path(cfg['_outdir'], 'graph_refinement.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('  [graph] checked loop refinement: %s, median %.5f -> %.5f, P95 %.5f -> %.5f (%0.1fs)'
          % ('accepted' if accepted else 'reverted', before['median'], after['median'], before['p95'], after['p95'], report['elapsed_seconds']), flush=True)
    return candidate if accepted else poses


_old_merge = Merger.merge_and_clean


def _fast_merge(self, raws, Ms, use):
    """Identical support counts, queried in the reverse direction, without Python neighbor lists."""
    cfg = self.cfg
    per_frame, colors = [], []
    for k in use:
        p = _score_cloud(raws, k, cfg)
        per_frame.append(np.asarray(p.points) @ Ms[k][:3, :3].T + Ms[k][:3, 3])
        colors.append(np.asarray(p.colors).copy() if p.has_colors() else np.full((len(p.points), 3), .7))
    mf = _cloud_from_arrays(np.vstack(per_frame))
    mf.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    del colors
    mf = mf.voxel_down_sample(cfg['merge_voxel'])
    points = np.asarray(mf.points)
    counts = np.zeros(len(points), dtype=np.uint32)
    radius = float(cfg['cover_radius'])
    # <= preserves original query_ball_point boundary semantics.
    def support_one(p):
        tree = cKDTree(p)
        support = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), 200000):
            d, _ = tree.query(points[start:start+200000], k=1,
                              distance_upper_bound=np.nextafter(radius, np.inf), eps=0.)
            support[start:start+len(d)] = (d <= radius)
        return support
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(4, _auto_jobs(cfg))) as executor:
        for support in executor.map(support_one, per_frame):
            counts += support
    mc = mf.select_by_index(np.flatnonzero(counts >= cfg['cover_min']))
    print('  [merge] %d points, clean %d (exact independent-frame support)' % (len(points), len(mc.points)), flush=True)
    if cfg.get('shell_filter', False):
        mc = self.remove_shell(mc, cfg)
    return mf, mc, per_frame


Merger.merge_and_clean = _fast_merge


def write_performance_report(cfg, files, use, outdir):
    elapsed = time.monotonic()-cfg['_started_at']
    target = float(cfg.get('time_target_seconds', 300.))
    report = dict(elapsed_seconds=elapsed, target_seconds=target,
                  target_met=elapsed <= target, inputs=len(files), accepted=len(use),
                  omitted_frames=[k for k in range(len(files)) if k not in set(use)],
                  benchmark_kind=cfg.get('_benchmark_kind', 'full-pipeline'),
                  pair_cache_reuse_requested=bool(cfg.get('reuse_pairs')),
                  prepare_seconds=cfg.get('_prepare_seconds'),
                  prepared_cache_directory=cfg.get('_prepared_dir') or os.path.join(outdir, 'prepared_frames'),
                  note='No time limit truncates registration, reduces resolution, or drops unfinished pairs.')
    Path(outdir, 'performance.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('  [time] %.1fs / %.1fs target: %s' % (elapsed, target, 'MET' if report['target_met'] else 'NOT MET'), flush=True)


_original_run = run


def _prepare_score_only(task):
    path, cfg = task
    dest = _prepared_path(path, cfg)
    if os.path.isfile(dest):
        with np.load(dest, allow_pickle=False) as data:
            if 'score_p' in data and 'score_c' in data:
                return
    p = safe_read_point_cloud(path).voxel_down_sample(cfg['score_voxel'])
    temp = dest + '.%d.tmp.npz' % os.getpid()
    np.savez(temp, score_p=np.asarray(p.points), score_c=np.asarray(p.colors))
    os.replace(temp, dest)


def run(files, cfg):
    prior_dir = cfg.get('_refine_existing')
    if not prior_dir:
        return _original_run(files, cfg)
    cfg['_files'] = files
    cfg['_benchmark_kind'] = 'refine-existing-poses-excludes-original-registration'
    prior = json.loads(Path(prior_dir, 'registration_run_manifest.json').read_text(encoding='utf8'))
    current = [(os.path.basename(p), os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in files]
    old = [(v['name'], v['size_bytes'], v['mtime_ns']) for v in prior['inputs']]
    if current != old:
        raise ValueError('Existing poses do not match current PLY list/sizes/timestamps')
    with np.load(Path(prior_dir, 'transforms.npz'), allow_pickle=False) as data:
        use = [int(k) for k in data['use_frames']]
        poses = {k: np.asarray(data['M_%02d' % k]).copy() for k in use}
        ref = int(data['ref_frame'])
    if len(set(use)) != len(use) or any(k < 0 or k >= len(files) for k in use):
        raise ValueError('Invalid frame IDs in existing transforms')
    if any(not np.isfinite(T).all() or T.shape != (4, 4) for T in poses.values()):
        raise ValueError('Invalid saved transformations')
    if len(use) < 2:
        raise ValueError('Existing result needs at least two registered frames')
    raws = LazyFrameLoader(files, capacity=2)
    original_poses = {k:T.copy() for k,T in poses.items()}
    import multiprocessing as mp
    cache_dir = cfg.get('_prepared_dir') or os.path.join(cfg['_outdir'], 'prepared_frames')
    os.makedirs(cache_dir, exist_ok=True)
    jobs = _auto_jobs(cfg)
    print('  [refine-existing] prepare/check scoring clouds once, workers=%d' % jobs, flush=True)
    with mp.get_context('spawn').Pool(jobs) as pool:
        prepare=_prepare_frame if cfg.get('gpu_frame_refine',False) else _prepare_score_only
        prepare_cfg=dict(cfg,gpu_pipeline=False)
        begin=time.monotonic()
        for prepared_count,_ in enumerate(pool.imap(prepare, [(files[k],prepare_cfg) for k in use]),1):
            if prepared_count%20==0 or prepared_count==len(use):
                print('[refine-prepare] %d/%d %.1fs'%(prepared_count,len(use),time.monotonic()-begin),flush=True)
        cfg['_prepare_seconds']=time.monotonic()-begin
    records = _metric_saved_edges(str(Path(prior_dir, 'pairwise_edge_audit.csv')), files, set(use))
    reg = PairwiseRegistrator(cfg)
    edges = [reg._make_edge(i, j, T, med, ov, True) for i, j, T, med, ov in records]
    reg._export_audit(edges, files)
    # Anchor each original disconnected component independently; preserve rigid group placement.
    parent = {k:k for k in use}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for e in edges:
        a, b = find(e.i), find(e.j)
        if a != b:
            parent[a] = b
    groups = {}
    for k in use:
        groups.setdefault(find(k), []).append(k)
    checks, proposals = [], []
    for gi, group in enumerate(sorted(groups.values(), key=lambda x:(-len(x), x[0]))):
        subset = {k: poses[k] for k in group}
        degree = {k:0 for k in group}
        for e in edges:
            if e.i in degree and e.j in degree:
                degree[e.i] += 1
                degree[e.j] += 1
        anchor = ref if ref in subset else max(group, key=lambda k:degree[k])
        candidate = refine_component_graph(subset, edges, anchor, cfg)
        if any(not np.array_equal(candidate[k], subset[k]) for k in group):
            proposals.append((gi, candidate))
        poses.update(candidate)
        report = Path(cfg['_outdir'], 'graph_refinement.json')
        if len(group) > 1 and report.exists():
            details = json.loads(report.read_text(encoding='utf8'))
            if details['anchor'] == anchor and details['frames'] == len(group):
                checks.append(dict(component=gi, frame_ids=group, **details))
    local_samples = _metric_sample_world(raws, {k:np.eye(4) for k in use}, use, cfg)
    before = _geometry_score(original_poses, local_samples, cfg['overlap_dist'])
    poses = original_poses.copy()
    after = before
    tol = float(cfg.get('graph_geom_tolerance', .002))
    accepted = False
    global_checks = []
    for gi, proposal in proposals:
        trial = poses.copy()
        trial.update(proposal)
        score = _geometry_score(trial, local_samples, cfg['overlap_dist'])
        keep = (all(score[key] <= after[key]+tol for key in
                    ('median', 'mean', 'p95', 'matched_median', 'matched_p95', 'matched_rmse'))
                and score['overlap'] >= after['overlap']-.001)
        global_checks.append(dict(component=gi, accepted=bool(keep), before=after, proposed=score))
        for check in checks:
            if check['component'] == gi:
                check['accepted_global'] = bool(keep)
        if keep:
            poses, after, accepted = trial, score, True
    Path(cfg['_outdir'], 'component_refinement.json').write_text(json.dumps(checks, indent=2), encoding='utf8')
    Path(cfg['_outdir'], 'refine_existing_validation.json').write_text(json.dumps(
        dict(accepted=accepted, before=before, after=after, frames_preserved=len(use),
             component_checks=global_checks,
             note='Same deterministic samples and identical frame set; clipped at overlap threshold.'), indent=2), encoding='utf8')
    if not accepted:
        poses = original_poses
        print('  [refine-existing] joint geometric check failed; restored original poses', flush=True)
    print('  [refine-existing] preserved all %d accepted frames; %d original components' % (len(use), len(groups)), flush=True)
    poses=_joint_refine_existing(raws,poses,use,cfg,ref)
    mf, mc, per_frame = Merger(cfg).merge_and_clean(raws, poses, use)
    scores = leave_one_out_scores(raws, poses, use, cfg)
    print_scores(scores, cfg)
    return raws, poses, use, scores, ref, mf, mc, per_frame


"""Exact float64 CUDA primitives; no downsampling or approximate neighbours."""
from collections import OrderedDict
import time
import numpy as np
from scipy.spatial.transform import Rotation


class ExactCudaBackend:
    def __init__(self, cache_bytes=1_500_000_000, chunk_size=65536, search_mode='float64', specialized=True):
        import os, importlib.util
        from pathlib import Path
        self._dll_handles = []
        spec = importlib.util.find_spec('cupy')
        if spec is not None:
            root = Path(spec.origin).parent.parent/'nvidia'/'cu13'
            if root.exists():
                os.environ.setdefault('CUDA_PATH',str(root))
                for folder in [root/'bin',root/'bin'/'x86_64']:
                    if folder.is_dir() and hasattr(os,'add_dll_directory'):
                        self._dll_handles.append(os.add_dll_directory(str(folder)))
                        os.environ['PATH']=str(folder)+os.pathsep+os.environ.get('PATH','')
        import cupy as cp
        from cupyx.scipy.spatial import KDTree
        self.cp, self.KDTree = cp, KDTree
        self.cache = OrderedDict()
        self.cache_bytes = int(cache_bytes)
        self.chunk_size = int(chunk_size)
        self.search_mode = search_mode
        self.specialized = specialized
        self._knn3 = None
        self.stats = dict(queries=0, icp_calls=0, iterations=0, tree_builds=0)
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA GPU available')
        cp.cuda.Device(0).use()

    def sync(self):
        self.cp.cuda.get_current_stream().synchronize()

    def cloud(self, obj, normals=False, tree=False):
        cp = self.cp
        points = np.asarray(obj.points) if hasattr(obj, 'points') else np.asarray(obj)
        key = id(obj)
        item = self.cache.get(key)
        if item is None or item['host'].shape != points.shape or not np.array_equal(item['host'], points):
            item = dict(owner=obj, host=points.copy(), p=cp.asarray(points, dtype=cp.float64), n=None, tree=None)
            self.cache[key] = item
        self.cache.move_to_end(key)
        if normals:
            host_normals = np.asarray(obj.normals)
            if item['n'] is None or not np.array_equal(item.get('host_n'), host_normals):
                item['host_n'] = host_normals.copy()
                item['n'] = cp.asarray(host_normals, dtype=cp.float64)
        if tree and item['tree'] is None:
            item['tree'] = self.KDTree(item['p'], copy_data=True)
            self.stats['tree_builds'] += 1
        while len(self.cache) > 2 and sum(v['host'].nbytes*4 for v in self.cache.values()) > self.cache_bytes:
            self.cache.popitem(last=False)
        return item

    def query(self, target, points, bound=np.inf, k=1):
        return self.query64(target, points, bound, k)

    def query64(self, target, points, bound=np.inf, k=1):
        cp = self.cp
        item = target if isinstance(target, dict) else self.cloud(target, tree=True)
        if item['tree'] is None:
            item['tree'] = self.KDTree(item['p'], copy_data=True)
            self.stats['tree_builds'] += 1
        points = cp.asarray(points, dtype=cp.float64)
        if self.specialized and k == 1 and points.shape[1] == 3:
            return self.query3d(item, points, bound)
        shape = (len(points),) if k == 1 else (len(points), k)
        distances = cp.empty(shape, dtype=cp.float64)
        indices = cp.empty(shape, dtype=cp.int64)
        for start in range(0, len(points), self.chunk_size):
            end = min(len(points), start+self.chunk_size)
            d, ix = item['tree'].query(points[start:end], k=k, eps=0., distance_upper_bound=bound)
            distances[start:end], indices[start:end] = d, ix
        self.stats['queries'] += 1
        return distances, indices

    def query3d(self, item, points, bound=np.inf):
        """Exact stack-free Euclidean 3D traversal specialised to float64.

        CuPy provides the balanced heap-shaped tree; this kernel specialises
        its traversal to squared Euclidean distance, removing general-p pow().
        No approximate pruning, FP32 coordinates or fast-math are used.
        """
        cp=self.cp
        if self._knn3 is None:
            self._knn3=cp.RawKernel(r'''
            extern "C" __global__ void exact_knn3(
                const double* points, const double* tree,
                const long long* original, const long long n,
                const long long nq, const double bound,
                double* distances, long long* indices) {
                long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
                if(tid>=nq) return;
                const double px=points[3*tid],py=points[3*tid+1],pz=points[3*tid+2];
                double best=bound*bound;
                long long best_id=n,prev=-1,node=0;
                while(node>=0) {
                    long long parent=(node+1)/2-1;
                    if(node>=n) { prev=node;node=parent;continue; }
                    long long left=2*node+1,right=left+1;
                    const double x=tree[3*node],y=tree[3*node+1],z=tree[3*node+2];
                    if(prev<left) {
                        const double dx=px-x,dy=py-y,dz=pz-z;
                        const double dd=(dx*dx+dy*dy)+dz*dz;
                        long long id=original[node];
                        if(dd<best || (dd==best && id<best_id)) { best=dd;best_id=id; }
                    }
                    int axis=(63-__clzll((unsigned long long)(node+1)))%3;
                    double delta=axis==0?px-x:(axis==1?py-y:pz-z);
                    long long near=delta>0?right:left;
                    long long far=delta>0?left:right;
                    long long next;
                    if(prev==near) next=(far<n && delta*delta<=best)?far:parent;
                    else if(prev==far) next=parent;
                    else next=left<n?near:parent;
                    prev=node;node=next;
                }
                double distance=sqrt(best);
                if(best_id==n || !(distance<bound)) { distance=__longlong_as_double(0x7ff0000000000000LL);best_id=n; }
                distances[tid]=distance;indices[tid]=best_id;
            }
            ''','exact_knn3',options=('--std=c++17','--fmad=false'))
        points=cp.ascontiguousarray(points,dtype=cp.float64)
        d=cp.empty(len(points),dtype=cp.float64);ix=cp.empty(len(points),dtype=cp.int64)
        for start in range(0,len(points),self.chunk_size):
            stop=min(len(points),start+self.chunk_size);count=stop-start
            self._knn3(((count+127)//128,),(128,),
                       (points[start:stop],item['tree'].tree,item['tree'].index,
                        np.int64(len(item['p'])),np.int64(count),float(bound),d[start:stop],ix[start:stop]))
        self.stats['queries']+=1
        return d,ix

    def plane_step(self, P, Q, N):
        return self.plane_step_with_loss(P,Q,N)[0]

    def plane_step_with_loss(self,P,Q,N):
        cp = self.cp
        r = cp.sum((P-Q)*N, axis=1)
        mean_absolute=cp.abs(r).mean()
        scale = 1.345 * mean_absolute
        weights = cp.minimum(1., scale/cp.maximum(cp.abs(r), 1e-30))
        weights=cp.where(scale<1e-12,1.,weights)
        pivot = cp.sum(P*weights[:, None], axis=0)/weights.sum()
        A = cp.concatenate((cp.cross(P-pivot, N), N), axis=1)
        H = A.T @ (weights[:, None]*A)
        b = -A.T @ (weights*r)
        packed=cp.asnumpy(cp.concatenate((H.reshape(-1),b,pivot,mean_absolute.reshape(1))))
        H,b,pivot=packed[:36].reshape(6,6),packed[36:42],packed[42:45]
        loss=float(packed[45])
        if 1.345*loss<1e-12:return np.eye(4),loss
        norms = np.sqrt(np.maximum(np.diag(H), 1e-20))
        Hn = H/norms[:, None]/norms[None, :]
        eigen, vectors = np.linalg.eigh(Hn)
        observable = eigen > max(eigen[-1]*1e-8, 1e-12)
        inverse = np.zeros(6)
        inverse[observable] = 1./eigen[observable]
        update = (vectors @ (inverse*(vectors.T @ (b/norms))))/norms
        R = Rotation.from_rotvec(update[:3]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pivot+update[3:]-R@pivot
        return T,loss

    def icp(self, src, tgt, thr, keep_ratio=.6, max_iter=60, conv=1e-4, T_init=None):
        cp = self.cp
        source = self.cloud(src, normals=(thr <= 2.))
        target = self.cloud(tgt, normals=True, tree=True)
        T = np.eye(4) if T_init is None else np.asarray(T_init, dtype=np.float64).copy()
        previous = np.inf
        self.stats['icp_calls'] += 1
        hints=None
        for _ in range(max_iter):
            transform=cp.asarray(T)
            R,t=transform[:3,:3],transform[:3,3]
            P = source['p']@R.T+t
            if hasattr(self,'query_hint'):
                distances,indices=self.query_hint(target,P,thr,hints)
            else:
                distances, indices = self.query(target, P, thr)
            hints=indices
            valid = cp.isfinite(distances)
            safe = cp.minimum(indices, len(target['p'])-1)
            if thr <= 2.:
                rotated = source['n']@R.T
                valid &= cp.abs(cp.sum(rotated*target['n'][safe], axis=1)) > .5
            take = cp.flatnonzero(valid)
            if len(take) < 10:
                break
            ds = distances[take]
            count = min(len(ds), max(10, int(len(ds)*keep_ratio)))
            cutoff = cp.partition(ds, count-1)[count-1]
            take = take[ds <= cutoff]
            pp, qq, nn = P[take], target['p'][indices[take]], target['n'][indices[take]]
            delta,current = self.plane_step_with_loss(pp, qq, nn)
            if not np.isfinite(delta).all():
                raise FloatingPointError('Non-finite CUDA ICP update')
            T = delta@T
            self.stats['iterations'] += 1
            if abs(previous-current) < conv:
                previous = current
                break
            previous = current
        return T, float(previous)

    def evaluate(self, src_points, target, T, dist_thr, bound=np.inf):
        cp = self.cp
        P = cp.asarray(src_points, dtype=cp.float64)
        P = P@cp.asarray(T[:3, :3]).T+cp.asarray(T[:3, 3])
        d, _ = self.query(target, P, bound=bound)
        if np.isfinite(bound):
            d = cp.minimum(d, bound)
        return float(cp.median(d)), float((d < dist_thr).mean())

    def clear(self):
        self.cache.clear()
        self.cp.get_default_memory_pool().free_all_blocks()

# CUDA backend: identical float64 mathematics, exact neighbours, bounded rejection scores.
_CPU_ICP = icp_custom
_CPU_PAIR_EVALUATE = PairwiseRegistrator._evaluate_cached
_CUDA_BACKEND = None
_CUDA_SETTINGS = dict(enabled=True, min_points=5000, cache_bytes=256000000)


def _gpu_backend():
    global _CUDA_BACKEND
    if _CUDA_BACKEND is None:
        _CUDA_BACKEND = ExactCudaBackend(cache_bytes=_CUDA_SETTINGS['cache_bytes'])
        _CUDA_BACKEND.cp.get_default_memory_pool().set_limit(size=650000000)
    return _CUDA_BACKEND


def _pair_score_bound(cfg):
    # For an even-sized median, either central value can be censored.
    # A cap strictly above 2*all acceptance cutoffs ensures every possibly
    # accepted median is uncensored and exactly equals the original median.
    cutoff = max(2., float(cfg['edge_med_max']), float(cfg['weak_med_max']),
                 float(cfg['overlap_dist']))
    return 2.*cutoff + max(1e-6, 32.*np.spacing(cutoff))


def _bounded_pair_score(self, raws, i, j, T, voxel, dist_thr):
    if voxel != self.cfg['score_voxel']:
        return _CPU_PAIR_EVALUATE(self, raws, i, j, T, voxel, dist_thr)
    if i not in self._score:
        self._score[i] = _score_cloud(raws, i, self.cfg)
    self._score.move_to_end(i)
    while len(self._score) > int(self.cfg.get('pair_score_cache_capacity', 8)):
        self._score.popitem(last=False)
    if j not in self._target_trees:
        self._target_trees[j] = cKDTree(np.asarray(raws[j].points))
    self._target_trees.move_to_end(j)
    while len(self._target_trees) > int(self.cfg.get('pair_tree_cache_capacity', 2)):
        self._target_trees.popitem(last=False)
    points = np.asarray(self._score[i].points)
    query = points@T[:3,:3].T+T[:3,3]
    cap = _pair_score_bound(self.cfg)
    distances, _ = self._target_trees[j].query(query,k=1,eps=0.,distance_upper_bound=cap)
    return float(np.median(np.minimum(distances,cap))),float((distances<dist_thr).mean())


def _bounded_hp_score(src,tgt,T,cfg):
    points=np.asarray(src.voxel_down_sample(.2).points)
    query=points@T[:3,:3].T+T[:3,3]
    cap=_pair_score_bound(cfg)
    distances,_=_target_tree(tgt).query(query,k=1,eps=0.,distance_upper_bound=cap)
    return float(np.median(np.minimum(distances,cap))),float((distances<.3).mean())


def icp_custom(src,tgt,thr,keep_ratio=.6,max_iter=60,conv=1e-4,T_init=None):
    if (not _CUDA_SETTINGS['enabled'] or
            len(src.points)<_CUDA_SETTINGS['min_points']):
        return _CPU_ICP(src,tgt,thr,keep_ratio,max_iter,conv,T_init)
    if not tgt.has_normals():
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=thr*2,max_nn=30))
    if thr<=2. and not src.has_normals():
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=thr*2,max_nn=30))
    try:
        return _gpu_backend().icp(src,tgt,thr,keep_ratio,max_iter,conv,T_init)
    except Exception as exc:
        # A memory fallback preserves the full CPU algorithm and resolution.
        import cupy
        if not isinstance(exc,cupy.cuda.memory.OutOfMemoryError):
            raise
        print('[cuda-memory-fallback] %s'%exc,flush=True)
        _gpu_backend().clear()
        return _CPU_ICP(src,tgt,thr,keep_ratio,max_iter,conv,T_init)


PairwiseRegistrator._evaluate_cached = _bounded_pair_score
_BASE_PW_INIT = _pw_init


def _pw_init(files,cfg):
    _CUDA_SETTINGS.update(enabled=cfg.get('compute_backend','cuda')=='cuda',
                          min_points=int(cfg.get('gpu_min_points',5000)),
                          cache_bytes=int(cfg.get('gpu_cache_bytes',256000000)))
    _BASE_PW_INIT(files,cfg)


_BASE_PW_WORK=_pw_work


def _pw_work(pair):
    result=_BASE_PW_WORK(pair)
    stats=dict(_CUDA_BACKEND.stats) if _CUDA_BACKEND is not None else {}
    stats.update(pid=os.getpid(),backend='cuda-float64' if _CUDA_BACKEND is not None else 'cpu-small-cloud')
    path=Path(_PW['cfg']['_outdir'])/('cuda_worker_%d.json'%os.getpid())
    path.write_text(json.dumps(stats),encoding='utf8')
    return result


_BASE_AUDIT=PairwiseRegistrator._export_audit


def _cuda_export_audit(self,edges,files):
    _BASE_AUDIT(self,edges,files)
    path=Path(self.cfg.get('_outdir','.'))/'pairwise_edge_audit.csv'
    with path.open(encoding='utf-8-sig',newline='') as fp:
        reader=csv.DictReader(fp);fields=list(reader.fieldnames);rows=list(reader)
    fields.append('median_is_lower_bound')
    limit=max(2.,float(self.cfg['edge_med_max']),float(self.cfg['weak_med_max']))
    for row in rows:row['median_is_lower_bound']=int(float(row['median'])>limit)
    with path.open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields);writer.writeheader();writer.writerows(rows)


PairwiseRegistrator._export_audit=_cuda_export_audit
_BASE_PERFORMANCE=write_performance_report


def write_performance_report(cfg,files,use,outdir):
    _BASE_PERFORMANCE(cfg,files,use,outdir)
    path=Path(outdir)/'performance.json';report=json.loads(path.read_text(encoding='utf8'))
    worker_stats=[json.loads(p.read_text(encoding='utf8')) for p in Path(outdir).glob('cuda_worker_*.json')]
    report.update(compute_backend=cfg.get('compute_backend'),gpu_dtype='float64',
                  approximate_neighbours=False,pair_median_rejection_cap=_pair_score_bound(cfg),
                  cuda_worker_stats=worker_stats,
                  note='CUDA 3D exact Euclidean search and float64 arithmetic. Rejected scores may be censored; accepted scores remain exact. No time cutoff drops tasks.')
    pipeline=Path(outdir,'cuda_pipeline_stats.json')
    report['cuda_pipeline_stats']=json.loads(pipeline.read_text(encoding='utf8')) if pipeline.exists() else None
    report['elapsed_within_time_budget']=report['target_met']
    report['prepared_cache_files_before_run']=cfg.get('_prepared_cache_files_before_run')
    report['full_raw_registration_run']=not (cfg.get('_refine_existing') or cfg.get('_baseline_cache') or cfg.get('reuse_pairs') or cfg.get('_prepared_dir') or cfg.get('_prepared_cache_files_before_run',0))
    report['full_registration_target_met']=bool(report['target_met'] and report['full_raw_registration_run'])
    report['absolute_accuracy_verified']=False
    if cfg.get('compute_backend')=='cuda':
        gpu=_gpu_backend()
        name=gpu.cp.cuda.runtime.getDeviceProperties(0)['name']
        report['cuda_device']=name.decode() if isinstance(name,bytes) else str(name)
    path.write_text(json.dumps(report,indent=2),encoding='utf8')


DEFAULT_CFG.update(compute_backend='cuda',gpu_min_points=5000,gpu_cache_bytes=256000000,
                   bounded_pair_scores=True)
PairCache.KEYS = PairCache.KEYS + ('compute_backend','bounded_pair_scores')


def _validate_cuda_startup(cfg):
    _CUDA_SETTINGS.update(enabled=cfg.get('compute_backend','cuda')=='cuda',
                          min_points=int(cfg.get('gpu_min_points',5000)),
                          cache_bytes=int(cfg.get('gpu_cache_bytes',256000000)))
    if not _CUDA_SETTINGS['enabled']:
        print('[backend] CPU, bounded exact pair scores enabled',flush=True)
        return
    gpu=_gpu_backend()
    target=np.array([[0.,0.,0.],[1.,2.,3.],[7.,-3.,2.]],dtype=np.float64)
    d,ix=gpu.query(target,target+.01)
    expected,index=cKDTree(target).query(target+.01)
    np.testing.assert_allclose(gpu.cp.asnumpy(d),expected,rtol=1e-11,atol=1e-11)
    np.testing.assert_array_equal(gpu.cp.asnumpy(ix),index)
    print('[backend] CUDA float64; exact KDTree validated; CPU scores and 6x6 solve',flush=True)



def _raw_cache_paths(path,cfg):
    import scipy
    abi=hashlib.sha256(scipy.__version__.encode()).hexdigest()[:8]
    base=_prepared_path(path,cfg)[:-4]+'_raw_ckd72_'+abi
    return {kind:base+'_'+kind+'.npy' for kind in ['points','nodes','index','parent','bounds']}


def _save_raw_gpu_cache(path,raw,cfg):
    paths=_raw_cache_paths(path,cfg)
    if all(os.path.isfile(p) for p in paths.values()):return
    points=np.asarray(raw.points)
    tree=cKDTree(points)
    record=tree.__getstate__()[0]
    if record.nbytes!=72*tree.size:raise RuntimeError('Unsupported SciPy GPU cache layout')
    dtype=np.dtype({'names':['axis','children','split','start','end','lp','rp','less','greater'],
                   'formats':['i8','i8','f8','i8','i8','i8','i8','i8','i8'],
                   'offsets':list(range(0,72,8)),'itemsize':72})
    nodes=record.view(dtype);branch=np.flatnonzero(nodes['axis']>=0)
    parent=np.full(tree.size,-1,np.int64)
    parent[nodes['less'][branch]]=branch;parent[nodes['greater'][branch]]=branch
    arrays=dict(points=points,nodes=record.view(np.uint8),index=tree.indices,parent=parent)
    arrays['bounds']=_kd_partition_bounds(nodes,tree.mins,tree.maxes)
    for kind,array in arrays.items():
        temporary=paths[kind]+'.%d.tmp.npy'%os.getpid()
        np.save(temporary,array,allow_pickle=False);os.replace(temporary,paths[kind])


_PIPE_PREPARE_FRAME=_prepare_frame


def _prepare_frame(task):
    path,cfg=task
    result=_PIPE_PREPARE_FRAME(task)
    if cfg.get('compute_backend')=='cuda' and cfg.get('gpu_pipeline',True):
        paths=_raw_cache_paths(path,cfg)
        if not all(os.path.isfile(p) for p in paths.values()):
            if all(os.path.isfile(paths[k]) for k in ['points','nodes','index','parent']):
                points=np.load(paths['points'],mmap_mode='r',allow_pickle=False)
                record=np.load(paths['nodes'],mmap_mode='r',allow_pickle=False)
                dtype=np.dtype({'names':['axis','children','split','start','end','lp','rp','less','greater'],
                    'formats':['i8','i8','f8','i8','i8','i8','i8','i8','i8'],
                    'offsets':list(range(0,72,8)),'itemsize':72})
                bounds=_kd_partition_bounds(record.view(dtype),points.min(axis=0),points.max(axis=0))
                temp=paths['bounds']+'.%d.tmp.npy'%os.getpid();np.save(temp,bounds,allow_pickle=False)
                os.replace(temp,paths['bounds'])
            else:
                raw=safe_read_point_cloud(path);_save_raw_gpu_cache(path,raw,cfg)
    return result

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree


def _kd_partition_bounds(nodes,minimum,maximum):
    boxes=np.empty((len(nodes),6),dtype=np.float64)
    boxes[0,:3]=minimum;boxes[0,3:]=maximum
    frontier=np.array([0],dtype=np.int64)
    while len(frontier):
        branch=frontier[nodes['axis'][frontier]>=0]
        if not len(branch):break
        left,right=nodes['less'][branch],nodes['greater'][branch]
        axis=nodes['axis'][branch];split=nodes['split'][branch]
        boxes[left]=boxes[branch];boxes[right]=boxes[branch]
        boxes[left,axis+3]=split;boxes[right,axis]=split
        frontier=np.concatenate((left,right))
    return boxes


class SharedExactCuda(ExactCudaBackend):
    """One CUDA context, immutable shared clouds, CPU-built exact KD nodes."""
    def __init__(self,cache_bytes=1_500_000_000,chunk_size=65536):
        super().__init__(cache_bytes,chunk_size)
        self.lock=threading.RLock()
        self._node_kernel=None
        self.score_sources=OrderedDict()

    def evaluate(self,src_points,target,T,dist_thr,bound=np.inf):
        cp=self.cp;key=id(src_points)
        with self.lock:
            entry=self.score_sources.get(key)
            if entry is None:
                tensor=cp.asarray(src_points,dtype=cp.float64)
                event=cp.cuda.Event();event.record()
                entry=(src_points,tensor,event);self.score_sources[key]=entry
            self.score_sources.move_to_end(key)
            while len(self.score_sources)>2 and sum(v[1].nbytes for v in self.score_sources.values())>128000000:
                self.score_sources.popitem(last=False)
        cp.cuda.get_current_stream().wait_event(entry[2])
        P=entry[1]@cp.asarray(T[:3,:3]).T+cp.asarray(T[:3,3])
        d,_=self.query(target,P,bound)
        if np.isfinite(bound):d=cp.minimum(d,bound)
        return float(cp.median(d)),float((d<dist_thr).mean())

    def cloud(self,obj,normals=False,tree=False):
        cp=self.cp;key=id(obj)
        build=False
        with self.lock:
            item=self.cache.get(key)
            if item is None:
                item=dict(owner=obj,ready=threading.Event(),error=None)
                self.cache[key]=item;build=True
            self.cache.move_to_end(key)
        if build:
            try:
                points=np.asarray(obj.points) if hasattr(obj,'points') else np.asarray(obj)
                item.update(host=points,p=cp.asarray(points,dtype=cp.float64),n=None)
                if hasattr(obj,'has_normals') and obj.has_normals():
                    item['n']=cp.asarray(np.asarray(obj.normals),dtype=cp.float64)
                prebuilt=getattr(obj,'gpu_tree_state',None)
                cpu=None if prebuilt is not None else cKDTree(points)
                raw=prebuilt['nodes'] if prebuilt is not None else cpu.__getstate__()[0]
                dtype=np.dtype({'names':['axis','children','split','start','end','lp','rp','less','greater'],
                    'formats':['i8','i8','f8','i8','i8','i8','i8','i8','i8'],
                    'offsets':list(range(0,72,8)),'itemsize':72})
                size=len(prebuilt['parent']) if prebuilt is not None else cpu.size
                if np.dtype(np.intp).itemsize!=8 or raw.nbytes!=size*72:
                    item['tree']=self.KDTree(item['p'],copy_data=True)
                    item['bytes']=item['p'].nbytes*4
                else:
                    nodes=raw.view(dtype)
                    branch=np.flatnonzero(nodes['axis']>=0)
                    left,right=nodes['less'][branch],nodes['greater'][branch]
                    if not (np.all((left>0)&(left<size)) and np.all((right>0)&(right<size))):
                        raise RuntimeError('Unsupported SciPy KDTree node layout')
                    if prebuilt is None:
                        parent=np.full(size,-1,np.int64);parent[left]=branch;parent[right]=branch
                        index=cpu.indices
                        boxes=_kd_partition_bounds(nodes,cpu.mins,cpu.maxes)
                    else:parent,index,boxes=prebuilt['parent'],prebuilt['index'],prebuilt['bounds']
                    item['tree']=dict(nodes=cp.asarray(raw.view(np.uint8)),
                                      parent=cp.asarray(parent),index=cp.asarray(index),bounds=cp.asarray(boxes))
                    item['bytes']=item['p'].nbytes+raw.nbytes+parent.nbytes+index.nbytes+boxes.nbytes
                    labels=getattr(obj,'frame_labels',None)
                    if labels is not None:
                        labels=np.asarray(labels,dtype=np.int32)
                        leaf=np.flatnonzero(nodes['axis']<0)
                        leaf=leaf[np.argsort(nodes['start'][leaf])]
                        ordered=labels[index];starts=nodes['start'][leaf]
                        lo=np.minimum.reduceat(ordered,starts);hi=np.maximum.reduceat(ordered,starts)
                        summary=np.full(size,-1,np.int32);summary[leaf]=np.where(lo==hi,lo,-1)
                        levels=[];front=np.array([0],np.int64)
                        while len(front):
                            br=front[nodes['axis'][front]>=0]
                            if not len(br):break
                            levels.append(br);front=np.concatenate((nodes['less'][br],nodes['greater'][br]))
                        for br in reversed(levels):
                            a,b=summary[nodes['less'][br]],summary[nodes['greater'][br]]
                            summary[br]=np.where((a==b)&(a>=0),a,-1)
                        item['labels']=cp.asarray(labels);item['summary']=cp.asarray(summary)
                        item['bytes']+=labels.nbytes+summary.nbytes
                if item['n'] is not None:item['bytes']+=item['n'].nbytes
                item['event']=cp.cuda.Event();item['event'].record()
                # Release temporary CPU node/index buffers only after upload.
                item['event'].synchronize()
                self.stats['tree_builds']+=1
                with self.lock:
                    while len(self.cache)>2 and sum(v.get('bytes',0) for v in self.cache.values())>self.cache_bytes:
                        candidates=[k for k,v in self.cache.items() if v['ready'].is_set() and k!=key]
                        if not candidates:break
                        self.cache.pop(candidates[0])
            except Exception as exc:item['error']=exc
            finally:item['ready'].set()
        else:item['ready'].wait()
        if item['error'] is not None:raise item['error']
        cp.cuda.get_current_stream().wait_event(item['event'])
        if normals and item['n'] is None:
            with self.lock:
                if item['n'] is None:
                    if not hasattr(obj,'has_normals') or not obj.has_normals():
                        raise ValueError('Cloud lacks required normals')
                    item['n']=cp.asarray(np.asarray(obj.normals),dtype=cp.float64)
                    item['bytes']+=item['n'].nbytes
                    item['event']=cp.cuda.Event();item['event'].record()
            cp.cuda.get_current_stream().wait_event(item['event'])
        return item

    def query64(self,target,points,bound=np.inf,k=1,hints=None):
        cp=self.cp
        item=target if isinstance(target,dict) else self.cloud(target,tree=True)
        if not isinstance(item['tree'],dict) or k!=1:
            return super().query64(item,points,bound,k)
        if self._node_kernel is None:
            with self.lock:
                if self._node_kernel is None:
                    self._node_kernel=cp.RawKernel(r'''
                    struct Node {long long axis,children;double split;
                        long long start,end;unsigned long long lp,rp;long long less,greater;};
                    extern "C" __global__ void exact_nodes(
                        const double* q,const double* data,const Node* nodes,
                        const long long* parent,const long long* order,const double* boxes,
                        const int* labels,const int* qlabels,const int* summary,int filtered,
                        const long long* hints,int hinted,
                        long long n,long long nq,double bound,double* d,long long* ix) {
                        long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
                        if(t>=nq)return;
                        double x=q[3*t],y=q[3*t+1],z=q[3*t+2],best=bound*bound;
                        long long hit=n,prev=-1,node=0;
                        if(hinted && hints[t]>=0 && hints[t]<n){
                            long long j=hints[t];
                            double dx=x-data[3*j],dy=y-data[3*j+1],dz=z-data[3*j+2];
                            double candidate=(dx*dx+dy*dy)+dz*dz;
                            // An upper bound only: the normal traversal still selects
                            // the winner and resolves equal-distance neighbours.
                            best=fmin(best,nextafter(candidate,__longlong_as_double(0x7ff0000000000000LL)));
                        }
                        int own=filtered?qlabels[t]:-1;
                        while(node>=0){
                            if(filtered && summary[node]==own){prev=node;node=parent[node];continue;}
                            const double* bb=boxes+6*node;
                            double ax=x<bb[0]?bb[0]-x:(x>bb[3]?x-bb[3]:0.);
                            double ay=y<bb[1]?bb[1]-y:(y>bb[4]?y-bb[4]:0.);
                            double az=z<bb[2]?bb[2]-z:(z>bb[5]?z-bb[5]:0.);
                            if((ax*ax+ay*ay)+az*az>best*1.0000000000000072){
                                prev=node;node=parent[node];continue;
                            }
                            const Node& nd=nodes[node];long long next;
                            if(nd.axis<0){
                                for(long long a=nd.start;a<nd.end;a++){
                                    long long j=order[a];
                                    if(filtered && labels[j]==own)continue;
                                    double dx=x-data[3*j],dy=y-data[3*j+1],dz=z-data[3*j+2];
                                    double dd=(dx*dx+dy*dy)+dz*dz;
                                    if(dd<best){best=dd;hit=j;}
                                }
                                next=parent[node];
                            }else{
                                double delta=(nd.axis==0?x:(nd.axis==1?y:z))-nd.split;
                                long long near=delta>0?nd.greater:nd.less;
                                long long far=delta>0?nd.less:nd.greater;
                                if(prev==near)next=delta*delta<=best?far:parent[node];
                                else if(prev==far)next=parent[node];
                                else next=near;
                            }
                            prev=node;node=next;
                        }
                        double distance=sqrt(best);
                        if(hit==n || !(distance<bound)){distance=__longlong_as_double(0x7ff0000000000000LL);hit=n;}
                        d[t]=distance;ix[t]=hit;
                    }
                    ''','exact_nodes',options=('--std=c++17','--fmad=false'))
        points=cp.ascontiguousarray(cp.asarray(points,dtype=cp.float64))
        if not hasattr(self,'_dummy_labels'):self._dummy_labels=cp.zeros(1,dtype=cp.int32)
        distances=cp.empty(len(points),cp.float64);indices=cp.empty(len(points),cp.int64)
        state=item['tree']
        for start in range(0,len(points),self.chunk_size):
            stop=min(len(points),start+self.chunk_size);count=stop-start
            self._node_kernel(((count+127)//128,),(128,),
                (points[start:stop],item['p'],state['nodes'],state['parent'],state['index'],state['bounds'],
                 self._dummy_labels,self._dummy_labels,self._dummy_labels,np.int32(0),
                 self._dummy_labels if hints is None else hints[start:stop],np.int32(hints is not None),
                 np.int64(len(item['p'])),np.int64(count),float(bound),distances[start:stop],indices[start:stop]))
        self.stats['queries']+=1
        return distances,indices

    def query_excluding_frames(self,obj,return_indices=False):
        cp=self.cp;item=self.cloud(obj,tree=True);state=item['tree']
        if not isinstance(state,dict):raise RuntimeError('Labelled CUDA search requires verified SciPy node layout')
        self.query64(item,item['p'][:1])
        distances=cp.empty(len(item['p']),dtype=cp.float64)
        indices=cp.empty(len(item['p']),dtype=cp.int64)
        for start in range(0,len(item['p']),self.chunk_size):
            stop=min(len(item['p']),start+self.chunk_size);count=stop-start
            self._node_kernel(((count+127)//128,),(128,),
                (item['p'][start:stop],item['p'],state['nodes'],state['parent'],state['index'],state['bounds'],
                 item['labels'],item['labels'][start:stop],item['summary'],np.int32(1),
                 self._dummy_labels,np.int32(0),
                 np.int64(len(item['p'])),np.int64(count),float('inf'),distances[start:stop],indices[start:stop]))
        return (distances,indices) if return_indices else distances

    def query_labelled(self,item,points,labels,bound=np.inf):
        cp=self.cp;state=item['tree']
        points=cp.ascontiguousarray(cp.asarray(points,dtype=cp.float64))
        labels=cp.ascontiguousarray(cp.asarray(labels,dtype=cp.int32))
        self.query64(item,item['p'][:1])
        distances=cp.empty(len(points),cp.float64);indices=cp.empty(len(points),cp.int64)
        for start in range(0,len(points),self.chunk_size):
            stop=min(len(points),start+self.chunk_size);count=stop-start
            self._node_kernel(((count+127)//128,),(128,),
                (points[start:stop],item['p'],state['nodes'],state['parent'],state['index'],state['bounds'],
                 item['labels'],labels[start:stop],item['summary'],np.int32(1),
                 self._dummy_labels,np.int32(0),
                 np.int64(len(item['p'])),np.int64(count),float(bound),distances[start:stop],indices[start:stop]))
        return distances,indices

    def query_hint(self,target,points,bound,hints):
        return self.query64(target,points,bound,hints=hints)

    def clear(self):
        with self.lock:self.cache.clear();self.score_sources.clear()
        self.cp.get_default_memory_pool().free_all_blocks()

from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait,FIRST_COMPLETED
import threading
import heapq
import weakref

_PIPE_FEATURES={}


def _pipe_feature_init(files,cfg):
    global _PIPE_FEATURES
    _PIPE_FEATURES=dict(files=files,cfg=cfg,cache=OrderedDict())


def _pipe_features(k):
    state=_PIPE_FEATURES;cache=state['cache']
    if k not in cache:
        with np.load(_prepared_path(state['files'][k],state['cfg']),allow_pickle=False) as d:
            feature=o3d.pipelines.registration.Feature();feature.data=d['feat_f']
            cache[k]=(_cloud_from_arrays(d['feat_p'],d['feat_n']),feature)
    cache.move_to_end(k)
    while len(cache)>24:cache.popitem(last=False)
    return cache[k]


def _pipe_initial(task):
    i,j,kind=task;cfg=_PIPE_FEATURES['cfg']
    o3d.utility.random.seed((int(cfg['random_seed'])+1000003*i+j)%2147483647)
    source,sf=_pipe_features(i);target,tf=_pipe_features(j)
    if kind=='ransac':
        try:T=ransac_init(source,sf,target,tf,cfg['ransac_dist'],cfg['ransac_iter'])
        except (RuntimeError,ValueError):return None
    else:
        try:
            T=np.asarray(o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source,target,sf,tf,o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=cfg['ransac_dist'])).transformation)
        except (RuntimeError,ValueError):return None
    return np.asarray(T).copy()


def _pipe_initial_timed(task):
    begin=time.monotonic()
    result=_pipe_initial(task)
    return result,time.monotonic()-begin


class _RawPoints:
    def __init__(self,points):self.points=points
    def has_normals(self):return False


class _SharedFrames:
    def __init__(self,files,cfg):
        self.files,self.cfg=files,cfg
        self.cache=OrderedDict();self.lock=threading.RLock();self.limit=650000000
        self.live=weakref.WeakValueDictionary();self.sizes={}
    def get(self,k,kind):
        key=k,kind;create=False
        with self.lock:
            item=self.cache.get(key)
            if item is None:
                live=self.live.get(key)
                if live is not None:
                    event=threading.Event();event.set()
                    item=dict(ready=event,value=live,error=None,bytes=self.sizes[key])
                    self.cache[key]=item
            if item is None:
                item=dict(ready=threading.Event(),value=None,error=None,bytes=0)
                self.cache[key]=item;create=True
            self.cache.move_to_end(key)
        if create:
            try:
                if kind=='raw':
                    paths=_raw_cache_paths(self.files[k],self.cfg)
                    if all(os.path.isfile(p) for p in paths.values()):
                        value=_RawPoints(np.load(paths['points'],mmap_mode='r',allow_pickle=False))
                        value.gpu_tree_state={name:np.load(paths[name],mmap_mode='r',allow_pickle=False)
                                              for name in ['nodes','index','parent','bounds']}
                    else:
                        raw=safe_read_point_cloud(self.files[k])
                        value=_RawPoints(np.asarray(raw.points).copy());del raw
                    size=value.points.nbytes
                elif kind=='hpscore':
                    cloud=self.get(k,'hp')
                    value=np.asarray(cloud.voxel_down_sample(.2).points).copy();size=value.nbytes
                else:
                    with np.load(_prepared_path(self.files[k],self.cfg),allow_pickle=False) as d:
                        if kind=='score':
                            value=d['score_p'];size=value.nbytes
                        else:
                            p,n=d[kind+'_p'],d[kind+'_n']
                            value=_cloud_from_arrays(p,n);size=p.nbytes+n.nbytes
                item.update(value=value,bytes=size)
                with self.lock:
                    self.live[key]=value;self.sizes[key]=size
                    while len(self.cache)>2 and sum(v['bytes'] for v in self.cache.values())>self.limit:
                        candidates=[q for q,v in self.cache.items() if q!=key and v['ready'].is_set()]
                        if not candidates:break
                        self.cache.pop(candidates[0])
            except Exception as exc:item['error']=exc
            finally:item['ready'].set()
        else:item['ready'].wait()
        if item['error'] is not None:raise item['error']
        return item['value']
    def __len__(self):return len(self.files)
    def __getitem__(self,k):return self.get(k,'raw')


_PIPE_LOCAL=threading.local()
_PIPE_BASE_BACKEND=_gpu_backend
_PIPE_BASE_INITIAL=_pair_initial_transform
_PIPE_BASE_HP_SCORE=_bounded_hp_score
_PIPE_BASE_TARGET_TREE=_target_tree
_PIPE_TREE_LOCK=threading.RLock()


def _gpu_backend():
    if getattr(_PIPE_LOCAL,'active',False):return _PIPE_LOCAL.engine.gpu
    return _PIPE_BASE_BACKEND()


def _pair_initial_transform(self,raws,i,j,sd,sf,td,tf):
    if getattr(_PIPE_LOCAL,'active',False):return _PIPE_LOCAL.initial.copy()
    return _PIPE_BASE_INITIAL(self,raws,i,j,sd,sf,td,tf)


def _bounded_hp_score(src,tgt,T,cfg):
    if getattr(_PIPE_LOCAL,'active',False):
        p=np.asarray(src.voxel_down_sample(.2).points)
        return _PIPE_LOCAL.engine.gpu.evaluate(p,tgt,T,.3,_pair_score_bound(cfg))
    return _PIPE_BASE_HP_SCORE(src,tgt,T,cfg)


def _target_tree(obj):
    with _PIPE_TREE_LOCK:return _PIPE_BASE_TARGET_TREE(obj)


class _PipelineReg(PairwiseRegistrator):
    def __init__(self,engine):super().__init__(engine.cfg);self.engine=engine;self.exact_scores={}
    def _feats(self,raws,k):return None,None
    def _icp_cloud(self,raws,k):return self.engine.frames.get(k,'mid')
    def _icp_cloud_hp(self,raws,k):return self.engine.frames.get(k,'hp')
    def _evaluate_cached(self,raws,i,j,T,voxel,dist_thr):
        if voxel!=self.cfg['score_voxel']:raise ValueError('Unexpected pair scoring resolution')
        key=(i,j,np.asarray(T).tobytes())
        if key in self.exact_scores:return self.exact_scores[key]
        result=self.engine.gpu.evaluate(self.engine.frames.get(i,'score'),
            self.engine.frames.get(j,'raw'),T,dist_thr,_pair_score_bound(self.cfg))
        if result[0]<=2.:self.exact_scores[key]=result
        return result
    def probe(self,i,j,T):
        radius=float(self.cfg['overlap_dist'])
        bound=max(1.000001,radius+1e-6)
        result=self.engine.gpu.evaluate(self.engine.frames.get(i,'score'),
            self.engine.frames.get(j,'raw'),T,radius,bound)
        if result[0]<=.4:self.exact_scores[i,j,np.asarray(T).tobytes()]=result
        return result
    def overlap(self,i,j,T):
        cp=self.engine.gpu.cp
        source=cp.asarray(self.engine.frames.get(i,'score'),dtype=cp.float64)
        points=source@cp.asarray(T[:3,:3]).T+cp.asarray(T[:3,3])
        radius=float(self.cfg['overlap_dist'])
        d,_=self.engine.gpu.query(self.engine.frames.get(j,'raw'),points,bound=radius)
        return float(cp.isfinite(d).mean())
    def _align_pair(self,raws,i,j):
        cfg=self.cfg;T0=_PIPE_LOCAL.initial.copy()
        med,ov=self._evaluate_cached(raws,i,j,T0,cfg['score_voxel'],cfg['overlap_dist'])
        if ov<.05:return None
        if not cfg.get('pair_high_precision',True):
            T=T0
            for threshold in cfg['pair_icp_thr']:
                T,_=icp_custom(self._icp_cloud(raws,i),self._icp_cloud(raws,j),
                               threshold,cfg['keep_ratio'],T_init=T)
            med,ov=self._evaluate_cached(raws,i,j,T,cfg['score_voxel'],cfg['overlap_dist'])
            reverse=self.overlap(j,i,np.linalg.inv(T))
            consistent=min(ov,reverse)>.35 or (max(ov,reverse)>.6 and min(ov,reverse)>.25)
            return T,med,ov,consistent
        if ov<cfg['hp_enter_overlap'] or med>2.:return T0,med,ov,True
        T,_=icp_custom(self._icp_cloud(raws,i),self._icp_cloud(raws,j),2.,.8,80,T_init=T0)
        overlap=self.overlap(i,j,T)
        if overlap<cfg['hp_enter_overlap']:
            med,overlap=self._evaluate_cached(raws,i,j,T,cfg['score_voxel'],cfg['overlap_dist'])
            return T,med,overlap,True
        source,target=self._icp_cloud_hp(raws,i),self._icp_cloud_hp(raws,j)
        for threshold in cfg['hp_icp_thr']:
            T,_=icp_custom(source,target,threshold,cfg['hp_keep_ratio'],
                           max_iter=cfg['hp_iter'],conv=5e-5,T_init=T)
        # With >50% required fine overlap, an accepted median is necessarily
        # inside the 0.3 radius. Values outside that radius cannot affect it.
        cap=.3 if cfg['hp_min_overlap']>.5 else _pair_score_bound(cfg)
        points=self.engine.frames.get(i,'hpscore')
        med,ov=self.engine.gpu.evaluate(points,target,T,.3,cap)
        if ov<cfg['hp_min_overlap']:
            med,ov=self._evaluate_cached(raws,i,j,T,cfg['score_voxel'],cfg['overlap_dist'])
            return T,med,ov,False
        reverse=self.overlap(j,i,np.linalg.inv(T))
        consistent=min(ov,reverse)>.35 or (max(ov,reverse)>.6 and min(ov,reverse)>.25)
        return T,med,ov,consistent


class _PipelineEngine:
    def __init__(self,files,cfg):
        self.cfg=cfg;self.frames=_SharedFrames(files,cfg)
        self.gpu=SharedExactCuda(cache_bytes=int(cfg.get('gpu_shared_cache_bytes',1500000000)))
        self.gpu.cp.get_default_memory_pool().set_limit(size=int(cfg.get('gpu_memory_limit',4500000000)))
    def process(self,i,j,kind,T):
        cp=self.gpu.cp;cp.cuda.Device(0).use()
        if not hasattr(_PIPE_LOCAL,'stream'):
            _PIPE_LOCAL.stream=cp.cuda.Stream(non_blocking=True)
            _PIPE_LOCAL.reg=_PipelineReg(self)
        _PIPE_LOCAL.active=True;_PIPE_LOCAL.engine=self;reg=_PIPE_LOCAL.reg
        reg.exact_scores={}
        try:
            with _PIPE_LOCAL.stream:
                if kind=='fgr':
                    T,_=icp_custom(self.frames.get(i,'mid'),self.frames.get(j,'mid'),
                                   2.,.8,max_iter=80,T_init=T)
                    med,ov=reg.probe(i,j,T)
                    reverse,reverse_ov=reg.probe(j,i,np.linalg.inv(T))
                    if max(med,reverse)>.4 or min(ov,reverse_ov)<.55:return 'ransac',None
                _PIPE_LOCAL.initial=T
                result=reg._align_pair(self.frames,i,j)
                _PIPE_LOCAL.stream.synchronize()
                return 'done',result
        finally:_PIPE_LOCAL.active=False


_BASE_RUN_PAIRS=run_pairs_safe


def run_pairs_safe(files,cfg,task_list,on_result,stage,jobs,stop_on_success=False):
    if cfg.get('compute_backend')!='cuda' or stage!='pairwise' or not cfg.get('gpu_pipeline',True):
        return _BASE_RUN_PAIRS(files,cfg,task_list,on_result,stage,jobs,stop_on_success)
    import multiprocessing as mp
    tasks=list(dict.fromkeys((int(i),int(j)) for i,j in task_list))
    if not tasks:return False,[]
    start=time.monotonic();engine=_PipelineEngine(files,cfg)
    cpu_jobs=max(1,int(cfg.get('gpu_initial_jobs',jobs)))
    debug_file=None
    if cfg.get('debug_gpu_pipeline',False):
        import faulthandler
        debug_file=open(Path(cfg['_outdir'])/'pipeline_stacks.log','w')
        faulthandler.dump_traceback_later(25,repeat=True,file=debug_file)
    nthreads=max(1,int(cfg.get('gpu_threads',4)))
    print('[cuda-pipeline] %d pairs, %d CPU initializers, %d CUDA streams, shared float64 trees'
          %(len(tasks),cpu_jobs,nthreads),flush=True)
    pending={};completed=0;counts={'fgr':0,'ransac':0}
    cpu_seconds={'fgr':0.,'ransac':0.}
    gpu_ready=[];cpu_repairs=[];next_coarse=0;cpu_busy=0;gpu_busy=0;serial=0
    cpu=ProcessPoolExecutor(max_workers=cpu_jobs,mp_context=mp.get_context('spawn'),
        initializer=_pipe_feature_init,initargs=(files,cfg))
    gpu=ThreadPoolExecutor(max_workers=nthreads)
    normal_exit=False
    try:
        initial_kind='fgr' if cfg.get('pair_fast_init',True) else 'ransac'
        while pending or gpu_ready or cpu_repairs or next_coarse<len(tasks):
            while gpu_ready and gpu_busy<nthreads:
                _,i,j,_,job_kind,T=heapq.heappop(gpu_ready)
                future=gpu.submit(engine.process,i,j,job_kind,T)
                pending[future]=('gpu',i,j,job_kind);gpu_busy+=1
            while cpu_busy<cpu_jobs:
                if cpu_repairs:
                    i,j=heapq.heappop(cpu_repairs);job_kind='ransac'
                elif next_coarse<len(tasks) and len(gpu_ready)<2*cpu_jobs:
                    i,j=tasks[next_coarse];next_coarse+=1;job_kind=initial_kind
                else:break
                future=cpu.submit(_pipe_initial_timed,(i,j,job_kind))
                pending[future]=('cpu',i,j,job_kind);cpu_busy+=1
            done,_=wait(pending,timeout=5.,return_when=FIRST_COMPLETED)
            for future in done:
                where,i,j,kind=pending.pop(future)
                if where=='cpu':
                    cpu_busy-=1
                    T,seconds=future.result();counts[kind]+=1;cpu_seconds[kind]+=seconds
                    if T is None:
                        if kind=='fgr':
                            heapq.heappush(cpu_repairs,(i,j))
                        else:
                            on_result(i,j,None);completed+=1
                    else:
                        serial+=1
                        heapq.heappush(gpu_ready,(0 if kind=='ransac' else 1,i,j,serial,kind,T))
                else:
                    gpu_busy-=1
                    status,result=future.result()
                    if status=='ransac':
                        heapq.heappush(cpu_repairs,(i,j))
                    else:
                        completed+=1
                        if on_result(i,j,result):
                            raise RuntimeError('Unexpected early-stop callback for pairwise stage')
                        if completed%50==0 or completed==len(tasks):
                            print('[cuda-pipeline] %d/%d %.1fs, RANSAC repairs=%d'
                                  %(completed,len(tasks),time.monotonic()-start,counts['ransac']),flush=True)
        normal_exit=True
    finally:
        if not normal_exit:
            for future in pending:future.cancel()
        gpu.shutdown(wait=True,cancel_futures=not normal_exit)
        cpu.shutdown(wait=True,cancel_futures=not normal_exit)
        stats=dict(engine.gpu.stats,completed_pairs=completed,cpu_initializers=cpu_jobs,
                   cuda_streams=nthreads,elapsed_seconds=time.monotonic()-start,
                   cpu_initializer_seconds=cpu_seconds,**counts)
        Path(cfg['_outdir'],'cuda_pipeline_stats.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
        engine.gpu.clear()
        if debug_file is not None:
            faulthandler.cancel_dump_traceback_later();debug_file.close()
    return False,[]


DEFAULT_CFG.update(gpu_pipeline=True,gpu_threads=4,gpu_shared_cache_bytes=1500000000,
                   gpu_memory_limit=4500000000,gpu_initial_jobs=min(16,max(1,(os.cpu_count() or 2)//2)))

_POST_CUDA=None
_POST_BACKEND_BASE=_gpu_backend
_POST_CPU_PP=icp_pp
_POST_CPU_CROSS=_cross_frame_distances
_POST_CPU_MERGE=Merger.merge_and_clean


def _gpu_backend():
    global _POST_CUDA
    if getattr(_PIPE_LOCAL,'active',False):return _PIPE_LOCAL.engine.gpu
    if _POST_CUDA is None:
        _POST_CUDA=SharedExactCuda()
        _POST_CUDA.cp.get_default_memory_pool().set_limit(size=4500000000)
    return _POST_CUDA


def icp_pp(src,tgt,thr,keep_ratio=.8,max_iter=80,conv=1e-4,T_init=None):
    if not _CUDA_SETTINGS['enabled'] or len(src.points)<5000:
        return _POST_CPU_PP(src,tgt,thr,keep_ratio,max_iter,conv,T_init)
    g=_gpu_backend();cp=g.cp
    s,t=g.cloud(src),g.cloud(tgt,tree=True)
    T=np.eye(4) if T_init is None else T_init.copy();previous=np.inf
    for _ in range(max_iter):
        transform=cp.asarray(T);P=s['p']@transform[:3,:3].T+transform[:3,3]
        d,ix=g.query(t,P,thr);valid=cp.flatnonzero(cp.isfinite(d))
        if len(valid)<10:break
        ds=d[valid];count=min(len(ds),max(10,int(len(ds)*keep_ratio)))
        cutoff=cp.partition(ds,count-1)[count-1];valid=valid[ds<=cutoff]
        pp,qq=P[valid],t['p'][ix[valid]]
        a,b=pp.mean(axis=0),qq.mean(axis=0)
        H=(pp-a).T@(qq-b)/len(pp)
        packet=cp.asnumpy(cp.concatenate((H.reshape(-1),a,b,ds[ds<=cutoff].mean().reshape(1))))
        U,_,V=np.linalg.svd(packet[:9].reshape(3,3));R=V.T@U.T
        if np.linalg.det(R)<0:V[-1]*=-1;R=V.T@U.T
        delta=np.eye(4);delta[:3,:3]=R;delta[:3,3]=packet[12:15]-R@packet[9:12]
        T=delta@T;current=float(packet[15])
        if abs(previous-current)<conv:break
        previous=current
    return T


def _gpu_merge(self,raws,Ms,use):
    if not _CUDA_SETTINGS['enabled']:return _POST_CPU_MERGE(self,raws,Ms,use)
    cfg=self.cfg;per_frame=[];colors=[]
    for k in use:
        cloud=_score_cloud(raws,k,cfg)
        per_frame.append(np.asarray(cloud.points)@Ms[k][:3,:3].T+Ms[k][:3,3])
        colors.append(np.asarray(cloud.colors).copy() if cloud.has_colors() else np.full((len(cloud.points),3),.7))
    merged=_cloud_from_arrays(np.vstack(per_frame))
    merged.colors=o3d.utility.Vector3dVector(np.vstack(colors));del colors
    merged=merged.voxel_down_sample(cfg['merge_voxel']);points=np.asarray(merged.points)
    g=_gpu_backend();cp=g.cp;query=cp.asarray(points)
    counts=cp.zeros(len(points),dtype=cp.uint32);radius=float(cfg['cover_radius'])
    for frame in per_frame:
        d,_=g.query(frame,query,np.nextafter(radius,np.inf))
        counts+=(d<=radius)
    keep=cp.asnumpy(counts)>=cfg['cover_min']
    clean=merged.select_by_index(np.flatnonzero(keep))
    print('[cuda-merge] %d points, clean %d; exact independent-frame support'
          %(len(points),len(clean.points)),flush=True)
    if cfg.get('shell_filter',False):clean=self.remove_shell(clean,cfg)
    g.clear()
    return merged,clean,per_frame


Merger.merge_and_clean=_gpu_merge


def _cross_frame_distances(world):
    if not _CUDA_SETTINGS['enabled'] or sum(len(p) for p in world.values())<10000:
        return _POST_CPU_CROSS(world)
    keys=sorted(world)
    if len(keys)<2:return {k:np.full(len(world[k]),np.inf) for k in keys}
    lengths=[len(world[k]) for k in keys]
    owner=_RawPoints(np.vstack([world[k] for k in keys]))
    owner.frame_labels=np.concatenate([np.full(len(world[k]),k,np.int32) for k in keys])
    gpu=_gpu_backend();d=gpu.cp.asnumpy(gpu.query_excluding_frames(owner))
    starts=np.r_[0,np.cumsum(lengths)]
    result={k:d[starts[i]:starts[i+1]] for i,k in enumerate(keys)}
    gpu.clear()
    return result

_RESCUE_TARGET_FEATURES=OrderedDict()


def _rescue_align_fast(source,target,cfg,initial=None):
    g=_gpu_backend() if _CUDA_SETTINGS['enabled'] else None
    source_mid=voxel_down(source,cfg['icp_voxel']);target_mid=voxel_down(target,cfg['icp_voxel'])
    def score(T):
        sampled=source.voxel_down_sample(cfg['score_voxel'])
        if g is not None:return g.evaluate(np.asarray(sampled.points),target,T,cfg['overlap_dist'],_pair_score_bound(cfg))
        query=np.asarray(sampled.points)@T[:3,:3].T+T[:3,3]
        d,_=cKDTree(np.asarray(target.points)).query(query,distance_upper_bound=_pair_score_bound(cfg))
        return float(np.median(np.minimum(d,_pair_score_bound(cfg)))),float((d<cfg['overlap_dist']).mean())
    def refine(T):
        T=icp_pp(source_mid,target_mid,8.,.9,40,T_init=T)
        T=icp_pp(source_mid,target_mid,4.,.85,60,T_init=T)
        for threshold in cfg['pair_icp_thr']:
            T,_=icp_custom(source_mid,target_mid,threshold,cfg['keep_ratio'],T_init=T)
        med,overlap=score(T)
        return T,med,overlap
    def accepted(result):
        return result[1]<cfg['rescue_accept_median'] and result[2]>cfg['rescue_accept_overlap']
    best=None
    if initial is not None:
        best=refine(initial)
        if accepted(best):return best
    sd,sf=fpfh_feats(source,cfg['voxel_fpfh'],cfg['fpfh_radius'])
    key=id(target)
    if key not in _RESCUE_TARGET_FEATURES:
        _RESCUE_TARGET_FEATURES[key]=(target,)+fpfh_feats(target,cfg['voxel_fpfh'],cfg['fpfh_radius'])
    _RESCUE_TARGET_FEATURES.move_to_end(key)
    while len(_RESCUE_TARGET_FEATURES)>2:_RESCUE_TARGET_FEATURES.popitem(last=False)
    _,td,tf=_RESCUE_TARGET_FEATURES[key]
    threshold=max(cfg['ransac_dist'],1.5*cfg['voxel_fpfh'])
    o3d.utility.random.seed(int(cfg['random_seed'])+len(source.points))
    T=None
    try:
        T=np.asarray(o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            sd,td,sf,tf,o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=threshold)).transformation)
    except (RuntimeError,ValueError):pass
    if T is not None:
        result=refine(T)
        if best is None or (result[1]<best[1] and result[2]>=best[2]):best=result
        if accepted(result):return result
    for seed in [1729,104729]:
        o3d.utility.random.seed(seed+len(source.points))
        try:
            T=ransac_init(sd,sf,td,tf,threshold,cfg['ransac_iter'])
        except (RuntimeError,ValueError):continue
        result=refine(T)
        if best is None or (result[1]<best[1] and result[2]>=best[2]):best=result
        if accepted(result):return result
    return best

def _distance_safeguard_values(distances,threshold):
    d=np.asarray(distances);matched=d[d<threshold]
    clipped=np.minimum(d,threshold)
    return dict(median=float(np.median(clipped)),mean=float(clipped.mean()),
        p95=float(np.percentile(clipped,95)),overlap=float((d<threshold).mean()),
        matched_median=float(np.median(matched)) if len(matched) else np.inf,
        matched_p95=float(np.percentile(matched,95)) if len(matched) else np.inf,
        matched_rmse=float(np.sqrt(np.mean(matched**2))) if len(matched) else np.inf)


def _joint_refine_existing(raws,poses,use,cfg,ref):
    if not cfg.get('gpu_frame_refine',False):return poses
    start=time.monotonic();g=_gpu_backend();cp=g.cp
    originals={k:T.copy() for k,T in poses.items()}
    local=_metric_sample_world(raws,{k:np.eye(4) for k in use},use,cfg)
    world={k:local[k]@poses[k][:3,:3].T+poses[k][:3,3] for k in use}
    keys=sorted(world);starts=np.r_[0,np.cumsum([len(world[k]) for k in keys])]
    owner=_RawPoints(np.vstack([world[k] for k in keys]))
    owner.frame_labels=np.concatenate([np.full(len(world[k]),k,np.int32) for k in keys])
    distance,nearest=g.query_excluding_frames(owner,return_indices=True)
    distance,nearest=cp.asnumpy(distance),cp.asnumpy(nearest)
    target_validation=g.cloud(owner,tree=True)
    before={k:_distance_safeguard_values(distance[starts[i]:starts[i+1]],cfg['overlap_dist']) for i,k in enumerate(keys)}
    nearest_labels=owner.frame_labels[nearest]
    frames=_SharedFrames(cfg['_files'],cfg)
    neighbors={}
    for i,k in enumerate(keys):
        labels,counts=np.unique(nearest_labels[starts[i]:starts[i+1]],return_counts=True)
        neighbors[k]=labels[np.argsort(-counts)][:int(cfg.get('gpu_refine_neighbors',6))].tolist()
    changes=[];proposals={};lock=threading.Lock()
    def worker(k):
        cp.cuda.Device(0).use()
        with cp.cuda.Stream(non_blocking=True):
            source=frames.get(k,'hp');points=[];normals=[]
            for j in neighbors[k]:
                cloud=frames.get(j,'hp');T=originals[j]
                points.append(np.asarray(cloud.points)@T[:3,:3].T+T[:3,3])
                normals.append(np.asarray(cloud.normals)@T[:3,:3].T)
            target=_cloud_from_arrays(np.vstack(points),np.vstack(normals))
            candidate=originals[k].copy()
            for threshold in cfg['hp_icp_thr']:
                candidate,_=g.icp(source,target,threshold,cfg['hp_keep_ratio'],cfg['hp_iter'],5e-5,candidate)
            query=local[k]@candidate[:3,:3].T+candidate[:3,3]
            distances,_=g.query_labelled(target_validation,query,np.full(len(query),k,np.int32))
            after=_distance_safeguard_values(cp.asnumpy(distances),cfg['overlap_dist'])
            displacement=float(np.max(np.linalg.norm(query-world[k],axis=1)))
            valid=all(after[name]<=before[k][name]+1e-9 for name in
                      ['median','mean','p95','matched_median','matched_p95','matched_rmse'])
            valid=valid and after['overlap']>=before[k]['overlap']-1e-9
            valid=valid and displacement<=float(cfg.get('gpu_refine_max_displacement',1.))
            with lock:
                changes.append(dict(frame=k,accepted_local=bool(valid),neighbors=neighbors[k],
                                    displacement=displacement,before=before[k],after=after))
                if valid:proposals[k]=candidate
                if len(changes)%20==0:print('[gpu-refine] %d/%d; safe proposals %d; %.1fs'
                    %(len(changes),len(use)-1,len(proposals),time.monotonic()-start),flush=True)
    print('[gpu-refine] full 0.2 resolution; independent-frame targets; %d frames'%(len(use)-1),flush=True)
    with ThreadPoolExecutor(max_workers=int(cfg.get('gpu_refine_threads',2))) as pool:
        list(pool.map(worker,[k for k in use if k!=ref]))
    g.clear()
    baseline=_geometry_score(originals,local,cfg['overlap_dist'])
    checks=[];result=originals
    for alpha in [1.,.5,.25,.125]:
        trial={k:T.copy() for k,T in originals.items()}
        for k,T in proposals.items():
            correction=T@np.linalg.inv(originals[k]);scaled=np.eye(4)
            scaled[:3,:3]=Rotation.from_rotvec(Rotation.from_matrix(correction[:3,:3]).as_rotvec()*alpha).as_matrix()
            scaled[:3,3]=correction[:3,3]*alpha;trial[k]=scaled@originals[k]
        score=_geometry_score(trial,local,cfg['overlap_dist'])
        valid=all(score[name]<=baseline[name]+1e-9 for name in
                  ['median','mean','p95','matched_median','matched_p95','matched_rmse'])
        valid=valid and score['overlap']>=baseline['overlap']-1e-9
        checks.append(dict(alpha=alpha,accepted=bool(valid),score=score))
        if valid:result=trial;break
    report=dict(frames_preserved=len(use),proposals=len(proposals),
        changed_frames=[k for k in use if not np.array_equal(result[k],originals[k])],
        elapsed_seconds=time.monotonic()-start,before=baseline,global_checks=checks,
        per_frame=sorted(changes,key=lambda v:v['frame']),
        note='Initialises from existing poses; not raw registration. Cross-frame NN safeguards are not absolute ground truth.')
    Path(cfg['_outdir'],'gpu_frame_refinement.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('[gpu-refine] %d safe frame updates, %.1fs'%(len(report['changed_frames']),report['elapsed_seconds']),flush=True)
    g.clear()
    return result



from concurrent.futures import as_completed
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist,squareform
import itertools


def fit_sphere(points,initial_center,initial_radius=5.8,threshold=.5):
    center=np.asarray(initial_center,float);radius=float(initial_radius)
    local=points[np.linalg.norm(points-center,axis=1)<initial_radius+2.]
    if len(local)<40:return None
    selection=np.abs(np.linalg.norm(local-center,axis=1)-radius)<1.
    for _ in range(20):
        P=local[selection]
        if len(P)<40:return None
        def fun(x):return np.linalg.norm(P-x[:3],axis=1)-x[3]
        def jac(x):
            delta=x[:3]-P;d=np.maximum(np.linalg.norm(delta,axis=1),1e-20)
            return np.column_stack((delta/d[:,None],-np.ones(len(P))))
        result=least_squares(fun,np.r_[center,radius],jac=jac,loss='linear',max_nfev=35,
                             xtol=1e-11,ftol=1e-11,gtol=1e-11)
        center,radius=result.x[:3],float(result.x[3])
        if not .6*initial_radius<radius<1.4*initial_radius:return None
        new=np.abs(np.linalg.norm(local-center,axis=1)-radius)<threshold
        if np.array_equal(selection,new):break
        selection=new
    P=local[selection];residual=np.linalg.norm(P-center,axis=1)-radius
    N=(P-center)/np.maximum(np.linalg.norm(P-center,axis=1)[:,None],1e-20)
    J=np.column_stack((-N,-np.ones(len(N))))
    eig=np.linalg.eigvalsh(J.T@J)
    return dict(center=center,radius=radius,n=len(P),rmse=float(np.sqrt(np.mean(residual**2))),
                condition=float(eig[-1]/max(eig[0],1e-30)))


def detect_spheres(points,normals,fine_points=None,radius=5.8,max_candidates=20):
    fine=points if fine_points is None else fine_points
    votes=np.vstack((points+radius*normals,points-radius*normals))
    origin=votes.min(axis=0)
    cells=np.floor((votes-origin)/1.).astype(np.int64)
    _,first,count=np.unique(cells,axis=0,return_index=True,return_counts=True)
    order=np.argsort(-count,kind='stable');seeds=[];peaks=[]
    for i in order:
        seed=(cells[first[i]]+.5)+origin
        if count[i]<8:break
        if all(np.linalg.norm(seed-old)>4. for old in seeds):
            seeds.append(seed);peaks.append(int(count[i]))
        if len(seeds)>=max_candidates:break
    result=[]
    for seed,votes in zip(seeds,peaks):
        fitted=fit_sphere(fine,seed,radius)
        if fitted is None or fitted['rmse']>.24 or fitted['condition']>2000 or fitted['n']<80:continue
        if abs(fitted['radius']-radius)>.15*radius:continue
        if all(np.linalg.norm(fitted['center']-old['center'])>4. for old in result):
            fitted['votes']=votes;result.append(fitted)
    return result


def match_spheres(source,target,tolerance=.8):
    # Match geometry observed in the anchor scan. No calibration values here.
    S=np.asarray([x['center'] for x in source]);Q=np.asarray([x['center'] for x in target])
    if len(S)<3 or len(Q)<3:return None
    best=None
    for subset in itertools.combinations(range(len(S)),min(4,len(S),len(Q))):
        A=S[list(subset)];distance=pdist(A)
        if len(A)==len(Q) and np.max(np.abs(np.sort(distance)-np.sort(pdist(Q))))>2.*tolerance:continue
        if distance.min()<pdist(Q).min()-2.*tolerance or distance.max()>pdist(Q).max()+2.*tolerance:continue
        for permutation in itertools.permutations(range(len(Q)),len(A)):
            B=Q[list(permutation)];error=np.sqrt(np.mean((pdist(B)-distance)**2))
            if error>tolerance or (best is not None and error>=best['distance_error']):continue
            a,b=A.mean(axis=0),B.mean(axis=0);U,_,V=np.linalg.svd((A-a).T@(B-b));R=V.T@U.T
            if np.linalg.det(R)<0:V[-1]*=-1;R=V.T@U.T
            T=np.eye(4);T[:3,:3]=R;T[:3,3]=b-R@a
            position=np.sqrt(np.mean(np.sum((A@R.T+T[:3,3]-B)**2,axis=1)))
            if position>tolerance:continue
            best=dict(T=T,distance_error=float(error),position_error=float(position),
                      source_indices=list(subset),target_indices=list(permutation))
    return best


def match_three_or_four(source,target,tolerance=.25):
    best=match_spheres(source,target,tolerance)
    if best is not None:return best
    if len(source)<=3:return None
    for subset in itertools.combinations(range(len(source)),3):
        match=match_spheres([source[k] for k in subset],target,tolerance)
        if match is None:continue
        match['source_indices']=[subset[k] for k in match['source_indices']]
        if best is None or match['distance_error']<best['distance_error']:best=match
    return best


def four_marker_subset(spheres):
    spheres=[s for s in spheres if s['rmse']<.12 and s['condition']<500]
    if len(spheres)<4:return None
    ranked=[]
    for subset in itertools.combinations(range(len(spheres)),4):
        selected=[spheres[k] for k in subset];distance=pdist([v['center'] for v in selected])
        if distance.min()<30.:continue
        radii=np.array([v['radius'] for v in selected])
        if radii.max()-radii.min()>.15*np.median(radii):continue
        ranked.append((-sum(v['rmse'] for v in selected),float(distance.min()),subset))
    if not ranked:return None
    selected=max(ranked)[-1]
    return [spheres[k] for k in selected]

import numpy as np
from scipy.spatial.transform import Rotation


def refine_sphere_pose(points,normals,initial,centers,radii,max_iter=35):
    """Rigid scan pose against spheres fitted to an independent anchor scan.

    Points are never projected, scaled, or replaced. The calibration board
    measurements are deliberately not accepted by this function.
    """
    T=np.asarray(initial,dtype=np.float64).copy();C=np.asarray(centers);R=np.asarray(radii)
    world=points@T[:3,:3].T+T[:3,3];world_n=normals@T[:3,:3].T
    selected=[];labels=[]
    for k,c in enumerate(C):
        delta=world-c;d=np.linalg.norm(delta,axis=1)
        valid=np.abs(d-R[k])<1.
        valid&=np.abs(np.sum(world_n*delta,axis=1)/np.maximum(d,1e-20))>.7
        ix=np.flatnonzero(valid)
        if len(ix)>=40:selected.append(ix);labels.append(np.full(len(ix),k,np.int32))
    if len(selected)<3:return T,dict(accepted=False,reason='fewer than three reliable sphere patches')
    ix=np.concatenate(selected);label=np.concatenate(labels);P=world[ix].copy()
    centers=C[label];radius=R[label];initial_world=P.copy();previous=np.inf
    initial_error=float(np.sqrt(np.mean((np.linalg.norm(P-centers,axis=1)-radius)**2)))
    for iteration in range(max_iter):
        delta=P-centers;distance=np.maximum(np.linalg.norm(delta,axis=1),1e-20)
        N=delta/distance[:,None];residual=distance-radius
        scale=max(.03,1.345*float(np.abs(residual).mean()))
        w=np.minimum(1.,scale/np.maximum(abs(residual),1e-30))
        pivot=np.average(P,axis=0,weights=w)
        A=np.column_stack((np.cross(P-pivot,N),N))
        H=A.T@(w[:,None]*A);b=-A.T@(w*residual)
        norms=np.sqrt(np.maximum(np.diag(H),1e-20));Hn=H/norms[:,None]/norms[None,:]
        eigen,vectors=np.linalg.eigh(Hn);observable=eigen>max(eigen[-1]*1e-8,1e-12)
        inverse=np.zeros(6);inverse[observable]=1./eigen[observable]
        update=(vectors@(inverse*(vectors.T@(b/norms))))/norms
        step=np.eye(4);step[:3,:3]=Rotation.from_rotvec(update[:3]).as_matrix()
        step[:3,3]=pivot+update[3:]-step[:3,:3]@pivot
        candidate=P@step[:3,:3].T+step[:3,3]
        if not np.isfinite(candidate).all():break
        current=float(np.mean(np.abs(np.linalg.norm(candidate-centers,axis=1)-radius)))
        P=candidate;T=step@T
        if abs(previous-current)<1e-7:break
        previous=current
    final_error=float(np.sqrt(np.mean((np.linalg.norm(P-centers,axis=1)-radius)**2)))
    shift=float(np.max(np.linalg.norm(P-initial_world,axis=1)))
    accepted=bool(final_error<=initial_error and shift<1.5 and np.isfinite(T).all())
    return (T if accepted else np.asarray(initial).copy()),dict(accepted=accepted,initial_rmse=initial_error,
        final_rmse=final_error,max_surface_displacement=shift,iterations=iteration+1,
        sphere_patches=len(selected),points=len(P))

import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def two_sphere_transforms(source,target,tolerance=.5,angle_step=6.):
    pairings=[]
    for i,j in itertools.combinations(range(len(source)),2):
        a,b=source[i]['center'],source[j]['center'];a,b=np.asarray(a),np.asarray(b)
        length=np.linalg.norm(b-a)
        for k,l in itertools.combinations(range(len(target)),2):
            c,d=np.asarray(target[k]['center']),np.asarray(target[l]['center'])
            error=abs(length-np.linalg.norm(d-c))
            if error<tolerance:pairings.append((error+source[i]['rmse']+source[j]['rmse'],i,j,k,l))
    result=[]
    for _,i,j,k,l in sorted(pairings)[:3]:
        A=np.array([source[i]['center'],source[j]['center']]);u=A[1]-A[0];u/=np.linalg.norm(u)
        for ids in [(k,l),(l,k)]:
            B=np.array([target[v]['center'] for v in ids]);v=B[1]-B[0];v/=np.linalg.norm(v)
            axis=np.cross(u,v);s=np.linalg.norm(axis);c=float(np.clip(u@v,-1.,1.))
            if s>1e-12:R0=Rotation.from_rotvec(axis/s*np.arctan2(s,c)).as_matrix()
            elif c>0:R0=np.eye(3)
            else:
                aux=np.eye(3)[np.argmin(abs(u))];axis=np.cross(u,aux);axis/=np.linalg.norm(axis)
                R0=Rotation.from_rotvec(axis*np.pi).as_matrix()
            for angle in np.arange(0.,360.,angle_step):
                R=Rotation.from_rotvec(v*np.deg2rad(angle)).as_matrix()@R0
                T=np.eye(4);T[:3,:3]=R;T[:3,3]=B.mean(0)-R@A.mean(0)
                result.append(dict(T=T,source_indices=[i,j],target_indices=list(ids)))
    return result


def rank_two_sphere_candidates(gpu,points,target_cloud,candidates):
    if not candidates:return []
    cp=gpu.cp;transforms=np.array([x['T'] for x in candidates])
    xyz=cp.einsum('ni,bji->bnj',cp.asarray(points),cp.asarray(transforms[:,:3,:3]))
    xyz+=cp.asarray(transforms[:,:3,3])[:,None,:]
    d,_=gpu.query(target_cloud,xyz.reshape(-1,3),bound=4.000001)
    d=cp.minimum(d,4.000001).reshape(len(candidates),len(points))
    score=cp.asnumpy(cp.mean(d,axis=1)-.2*cp.mean(d<1.,axis=1))
    return [dict(candidates[i],coarse_score=float(score[i])) for i in np.argsort(score)[:3]]

_DEPTH_MODEL=None
_DEPTH_INPUTS=set()
_DEPTH_BASE_READ=safe_read_point_cloud
_DEPTH_BASE_KEY=_prepared_key


def depth_correct(points,model):
    a,b,k=model['parameters'];P=np.asarray(points,dtype=np.float64)
    denominator=a+b*P[:,2:3]/model['depth_normalizer']
    if np.any(denominator<=0):raise ValueError('Depth correction outside its valid range')
    return P/denominator*np.array([1.,1.,k])


def depth_correct_normals(points,normals,model):
    a,b,k=model['parameters'];N=np.asarray(normals,dtype=np.float64).copy()
    N[:,2]+=b/model['depth_normalizer']/a*np.sum(np.asarray(points)*N,axis=1)
    N[:,2]/=k
    N/=np.maximum(np.linalg.norm(N,axis=1)[:,None],1e-30)
    return N


def _prepared_key(path,cfg):
    key=_DEPTH_BASE_KEY(path,cfg);model=cfg.get('_depth_model')
    if model and os.path.normcase(os.path.abspath(path))!=os.path.normcase(os.path.abspath(cfg['sphere_guide_path'])):
        key=hashlib.sha256((key+json.dumps(model,sort_keys=True)).encode()).hexdigest()
    return key


def safe_read_point_cloud(path):
    cloud=_DEPTH_BASE_READ(path)
    if _DEPTH_MODEL and os.path.normcase(os.path.abspath(path)) in _DEPTH_INPUTS:
        P=np.asarray(cloud.points)
        if cloud.has_normals():cloud.normals=o3d.utility.Vector3dVector(depth_correct_normals(P,np.asarray(cloud.normals),_DEPTH_MODEL))
        cloud.points=o3d.utility.Vector3dVector(depth_correct(P,_DEPTH_MODEL))
    return cloud


def _geometry_fit_depth(features,spheres):
    from scipy.optimize import least_squares
    target=np.array([s['center'] for s in spheres]);samples=[];groups=set()
    for key,row in sorted(features.items()):
        match=row['match']
        if match is None:continue
        A=np.array([row['spheres'][i]['center'] for i in match['source_indices']]);B=target[match['target_indices']]
        # Exact repeated scans get the same split, even with renamed input files.
        signature=hashlib.sha256(np.round(A[np.lexsort(A.T[::-1])],6).tobytes()).hexdigest()
        group=int(signature[:8],16)%3;groups.add(signature)
        for i,j in itertools.combinations(range(len(A)),2):samples.append((group,A[i],A[j],np.linalg.norm(B[i]-B[j])))
    if len(groups)<30:raise ValueError('Depth calibration needs at least 30 distinct reliable marker views')
    fold=np.array([s[0] for s in samples]);A=np.array([s[1] for s in samples]);B=np.array([s[2] for s in samples]);distance=np.array([s[3] for s in samples])
    train=fold!=0;test=~train
    if train.sum()<60 or test.sum()<30:raise ValueError('Insufficient independent calibration validation samples')
    def residual(x):
        model=dict(parameters=list(x),depth_normalizer=400.)
        return np.linalg.norm(depth_correct(A,model)-depth_correct(B,model),axis=1)-distance
    fit=least_squares(lambda x:residual(x)[train],[1.,0.,1.],loss='soft_l1',f_scale=.1,
        bounds=([.95,-.05,.95],[1.05,.05,1.05]),xtol=1e-12,ftol=1e-12,gtol=1e-12)
    original=residual([1.,0.,1.]);corrected=residual(fit.x);validation={}
    for name,mask in [('train',train),('heldout',test)]:
        validation[name]=dict(samples=int(mask.sum()),before_rmse=float(np.sqrt(np.mean(original[mask]**2))),
            after_rmse=float(np.sqrt(np.mean(corrected[mask]**2))),before_median=float(np.median(abs(original[mask]))),
            after_median=float(np.median(abs(corrected[mask]))))
    check=validation['heldout']
    if not fit.success or check['after_rmse']>=.95*check['before_rmse'] or check['after_median']>check['before_median']:
        raise ValueError('Depth calibration failed independent validation; omit --calibrate-depth to retain rigid geometry')
    return dict(parameters=fit.x.tolist(),depth_normalizer=400.,validation=validation,distinct_training_views=len(groups),
        formula='P_corrected = P / (a + b * P.z / depth_normalizer) * [1,1,k]; world = T_rigid @ P_corrected',
        target='Sphere geometry fitted to measured guide only; no calibration PDF values used',
        changes_input_geometry=True,raw_input_files_modified=False,heldout_used_for_parameter_fitting=False)


def _geometry_calibrated_prepare(task):
    index,path,cfg,previous,row=task
    dest=_prepared_path(path,cfg)
    with np.load(_prepared_path(path,previous),allow_pickle=False) as data:values={key:data[key].copy() for key in data.files}
    for prefix in ['hp','mid','feat','score']:
        P=values[prefix+'_p']
        if prefix+'_n' in values:values[prefix+'_n']=depth_correct_normals(P,values[prefix+'_n'],cfg['_depth_model'])
        values[prefix+'_p']=depth_correct(P,cfg['_depth_model'])
    with np.load(_prepared_path(path,previous),allow_pickle=False) as original:
        values['depth_init_map']=kabsch(original['feat_p'],values['feat_p'])
    feature=_cloud_from_arrays(values['feat_p'],values['feat_n'])
    values['feat_f']=np.asarray(o3d.pipelines.registration.compute_fpfh_feature(feature,
        o3d.geometry.KDTreeSearchParamHybrid(radius=cfg['fpfh_radius'],max_nn=100)).data)
    P=values['hp_p'];values['bounds']=np.stack((P.min(0),P.max(0)))
    temporary=dest+'.%d.tmp.npz'%os.getpid();np.savez(temporary,**values);os.replace(temporary,dest)
    observed=[]
    for old in row['spheres']:
        seed=depth_correct(np.array([old['center']]),cfg['_depth_model'])[0]
        fitted=fit_sphere(P,seed,cfg['sphere_radius'])
        if fitted is not None and fitted['rmse']<=.24 and fitted['condition']<=2000 and fitted['n']>=80:
            observed.append(dict(fitted,center=fitted['center'].tolist(),votes=old.get('votes',0)))
    match=match_three_or_four(observed,cfg['_guide_spheres'],tolerance=.5)
    if match is not None:match=dict(match,T=match['T'].tolist())
    return index,dict(spheres=observed,match=match,bounds=values['bounds'].tolist())


def _geometry_pair_initial_timed(task,model):
    if not model:return _pipe_initial_timed(task)
    # Keep descriptor matching in the original sensor coordinates; map its
    # coarse rigid initializer into corrected coordinates before exact ICP.
    begin=time.monotonic();T,_=_pipe_initial_timed(task)
    if T is None:return None,time.monotonic()-begin
    state=_PIPE_FEATURES;corrected=dict(state['cfg'],_depth_model=model)
    maps=state.setdefault('depth_initial_maps',{})
    for k in task[:2]:
        if k not in maps:
            with np.load(_prepared_path(state['files'][k],corrected),allow_pickle=False) as data:maps[k]=data['depth_init_map']
    i,j,_=task
    return maps[j]@T@np.linalg.inv(maps[i]),time.monotonic()-begin


def _geometry_surface_refine(files,poses,cfg,target):
    begin=time.monotonic();gpu=_gpu_backend();gpu.cloud(target,normals=True,tree=True)
    locked=set(cfg.get('_manual_bridge_frames', []))
    def worker(k):
        if k in locked:
            return k,poses[k].copy(),dict(accepted=False,reason='manual_bridge_locked')
        gpu.cp.cuda.Device(0).use()
        with gpu.cp.cuda.Stream(non_blocking=True):
            with np.load(_prepared_path(files[k],cfg),allow_pickle=False) as data:P,N,score=data['hp_p'],data['hp_n'],data['score_p']
            source=_cloud_from_arrays(P,N);initial=poses[k];T=initial.copy()
            before=gpu.evaluate(score,target,T,1.,_pair_score_bound(cfg))
            for threshold in cfg['hp_icp_thr']:
                T,_=gpu.icp(source,target,threshold,cfg['hp_keep_ratio'],cfg['hp_iter'],5e-5,T)
            after=gpu.evaluate(score,target,T,1.,_pair_score_bound(cfg))
            displacement=float(np.max(np.linalg.norm(P@(T[:3,:3]-initial[:3,:3]).T+T[:3,3]-initial[:3,3],axis=1)))
            accepted=after[0]<=before[0]+1e-9 and after[1]>=before[1]-.005 and displacement<1.5
            return k,T if accepted else initial,dict(accepted=bool(accepted),before=before,after=after,max_displacement=displacement)
    with ThreadPoolExecutor(int(cfg.get('gpu_threads',4))) as pool:result=list(pool.map(worker,sorted(poses)))
    gpu.clear();checks=dict(elapsed_seconds=time.monotonic()-begin,frames={k:v for k,_,v in result})
    Path(cfg['_outdir'],'surface_guide_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    return {k:T for k,T,_ in result}


_DEPTH_BASE_SAVE=save_outputs
def save_outputs(outdir,mf,mc,Ms,use,scores,ref,cfg):
    _DEPTH_BASE_SAVE(outdir,mf,mc,Ms,use,scores,ref,cfg)
    manual_frames=sorted(k for k in set(cfg.get('_manual_bridge_frames', [])) if k in Ms and k in use)
    Path(outdir,'manual_bridge_audit.json').write_text(json.dumps(dict(
        enabled=bool(cfg.get('manual_rescue', False)),confirmed_frames=manual_frames,
        confirmed_final_poses={str(k):np.asarray(Ms[k]).tolist() for k in manual_frames},
        pose_policy='Keep confirmed component rigid; skip later per-frame sphere/surface refinement',
        elapsed_includes_manual_wait=True),indent=2),encoding='utf8')
    if cfg.get('_depth_model'):
        path=Path(outdir,'transforms.npz')
        with np.load(path,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
        arrays.update(depth_parameters=np.asarray(cfg['_depth_model']['parameters']),
            depth_normalizer=np.array(cfg['_depth_model']['depth_normalizer']),
            transform_order=np.array('depth_correction_then_rigid_matrix'))
        np.savez(path,**arrays)


_DEPTH_BASE_PERFORMANCE=write_performance_report
def write_performance_report(cfg,files,use,outdir):
    _DEPTH_BASE_PERFORMANCE(cfg,files,use,outdir)
    path=Path(outdir,'performance.json');report=json.loads(path.read_text(encoding='utf8'))
    report.update(manual_bridge_enabled=bool(cfg.get('manual_rescue', False)),
        manual_bridge_frames=sorted(set(cfg.get('_manual_bridge_frames', []))),
        timing_mode='automatic_with_optional_manual_wait' if cfg.get('manual_rescue', False) else 'fully_automatic',
        surface_guide=bool(cfg.get('surface_guide')),depth_correction_applied=bool(cfg.get('_depth_model')),
        surface_reference_path=cfg.get('surface_reference_path'),
        overlapped_cuda_warmup_seconds=cfg.get('_overlapped_cuda_warmup_seconds'),
        transform_order='depth_correction_then_rigid_matrix' if cfg.get('_depth_model') else 'rigid_matrix_only')
    path.write_text(json.dumps(report,indent=2),encoding='utf8')

def _geometry_prepare(task):
    index,path,cfg=task
    bounds=_prepare_frame((path,dict(cfg,gpu_pipeline=False)))
    settings=[cfg['sphere_radius'],'sphere960-1',.5,25000]
    tag=hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:12]
    dest=Path(cfg['_outdir'],'sphere_features',_prepared_key(path,cfg)+'_'+tag+'.json')
    if dest.exists():
        spheres=json.loads(dest.read_text(encoding='utf8'))
    else:
        with np.load(_prepared_path(path,cfg),allow_pickle=False) as data:P,N=data['hp_p'],data['hp_n']
        step=max(1,len(P)//25000)
        spheres=detect_spheres(P[::step],N[::step],P,radius=cfg['sphere_radius'])
        spheres=[dict(s,center=s['center'].tolist()) for s in spheres]
        temporary=dest.with_name(dest.name+'.%d.tmp'%os.getpid())
        temporary.write_text(json.dumps(spheres),encoding='utf8');os.replace(temporary,dest)
    match=match_three_or_four(spheres,cfg['_guide_spheres'],tolerance=.5)
    if match is not None:match=dict(match,T=match['T'].tolist())
    return index,dict(spheres=spheres,match=match,bounds=np.asarray(bounds).tolist())


class _GeometryPairReg(_PipelineReg):
    def probe(self,i,j,T):
        result=super().probe(i,j,T)
        self.engine.probes[i,j]=result
        return result


class _GeometryPairEngine(_PipelineEngine):
    def __init__(self,files,cfg):
        super().__init__(files,cfg);self.probes={}
    def process(self,i,j,kind,T):
        cp=self.gpu.cp;cp.cuda.Device(0).use()
        if not hasattr(_PIPE_LOCAL,'stream'):_PIPE_LOCAL.stream=cp.cuda.Stream(non_blocking=True)
        if not isinstance(getattr(_PIPE_LOCAL,'reg',None),_GeometryPairReg):
            _PIPE_LOCAL.reg=_GeometryPairReg(self)
        return super().process(i,j,kind,T)


class _GeometryPairPool:
    def __init__(self,cpu,files,cfg):
        self.cpu=cpu;self.cfg=cfg;self.engine=_GeometryPairEngine(files,cfg)
        self.threads=ThreadPoolExecutor(max_workers=int(cfg.get('gpu_threads',4)));self.phases=[]
    def run(self,tasks,kind,on_result):
        tasks=sorted(set(tasks))
        if not tasks:return
        begin=time.monotonic();pending={};completed=0;cpu_seconds=0.;reported=-1
        for i,j in tasks:
            future=self.cpu.submit(_geometry_pair_initial_timed,(i,j,kind),self.cfg.get('_depth_model'));pending[future]=('cpu',i,j)
        print('[geometry-pairs] %s %d pairs; shared CPU pool and CUDA streams'%(kind,len(tasks)),flush=True)
        while pending:
            done,_=wait(pending,timeout=5.,return_when=FIRST_COMPLETED)
            for future in done:
                stage,i,j=pending.pop(future)
                if stage=='cpu':
                    initial,seconds=future.result();cpu_seconds+=seconds
                    if initial is None:on_result(i,j,None);completed+=1
                    else:
                        work=self.threads.submit(self.engine.process,i,j,kind,initial)
                        pending[work]=('gpu',i,j)
                else:
                    status,result=future.result()
                    on_result(i,j,result if status=='done' else None);completed+=1
                if completed!=reported and completed and (completed%100==0 or completed==len(tasks)):
                    reported=completed
                    print('[geometry-pairs] %d/%d %.1fs'%(completed,len(tasks),time.monotonic()-begin),flush=True)
        self.phases.append(dict(kind=kind,pairs=len(tasks),elapsed_seconds=time.monotonic()-begin,cpu_seconds=cpu_seconds))
    def close(self):
        self.threads.shutdown(wait=True)
        stats=dict(self.engine.gpu.stats,phases=self.phases)
        Path(self.cfg['_outdir'],'geometry_pair_stats.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
        self.engine.gpu.clear()


def _geometry_model(files,poses,cfg,frames):
    points=[]
    for k in sorted(poses):
        P=frames.get(k,'score');T=poses[k];points.append(P@T[:3,:3].T+T[:3,3])
    model=_cloud_from_arrays(np.vstack(points)).voxel_down_sample(cfg['score_voxel'])
    model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5,max_nn=40))
    return model


def _geometry_refine_many(files,poses,cfg,spheres):
    centers=np.array([s['center'] for s in spheres]);radii=np.array([s['radius'] for s in spheres])
    locked=set(cfg.get('_manual_bridge_frames', []))
    def worker(k):
        if k in locked:
            return k,poses[k].copy(),dict(accepted=False,reason='manual_bridge_locked')
        with np.load(_prepared_path(files[k],cfg),allow_pickle=False) as data:P,N=data['hp_p'],data['hp_n']
        T,check=refine_sphere_pose(P,N,poses[k],centers,radii)
        return k,T,check
    with ThreadPoolExecutor(max_workers=min(8,_auto_jobs(cfg))) as pool:results=list(pool.map(worker,sorted(poses)))
    return {k:T for k,T,_ in results},{k:check for k,_,check in results}


def _geometry_two_marker_stage(files,poses,features,spheres,cfg,frames,guide_target=None):
    model=guide_target if guide_target is not None else _geometry_model(files,poses,cfg,frames)
    mid=model.voxel_down_sample(cfg['icp_voxel']);mid.normalize_normals()
    gpu=_gpu_backend();gpu.cloud(model,normals=True,tree=True);gpu.cloud(mid,normals=True,tree=True)
    def worker(i):
        cp=gpu.cp;cp.cuda.Device(0).use();checks=[]
        with cp.cuda.Stream(non_blocking=True):
            candidates=two_sphere_transforms(features[i]['spheres'],spheres)
            if not candidates:return i,None,checks
            with np.load(_prepared_path(files[i],cfg),allow_pickle=False) as data:coarse=data['feat_p']
            ranked=rank_two_sphere_candidates(gpu,coarse,model,candidates)
            sm,sh=frames.get(i,'mid'),frames.get(i,'hp');score=frames.get(i,'score')
            for candidate in ranked:
                T=icp_pp(sm,mid,8.,.9,40,T_init=candidate['T']);T=icp_pp(sm,mid,4.,.85,60,T_init=T)
                for threshold in cfg['hp_icp_thr']:
                    T,_=icp_custom(sh,model,threshold,cfg['hp_keep_ratio'],cfg['hp_iter'],5e-5,T)
                median,overlap=gpu.evaluate(score,model,T,1.,_pair_score_bound(cfg))
                A=np.array([features[i]['spheres'][k]['center'] for k in candidate['source_indices']])
                B=np.array([spheres[k]['center'] for k in candidate['target_indices']])
                marker_error=float(np.max(np.linalg.norm(A@T[:3,:3].T+T[:3,3]-B,axis=1)))
                accepted=median<.4 and overlap>.6 and marker_error<1.
                checks.append(dict(frame=i,median=median,overlap=overlap,marker_error=marker_error,accepted=bool(accepted)))
                if accepted:return i,T,checks
        return i,None,checks
    todo=[k for k in range(len(files)) if k not in poses and len(features[k]['spheres'])>=2]
    checks=[];result=dict(poses)
    with ThreadPoolExecutor(max_workers=int(cfg.get('gpu_threads',4))) as pool:
        for i,T,records in pool.map(worker,todo):
            checks.extend(records)
            if T is not None:result[i]=T
    gpu.clear()
    return result,checks


def _geometry_propagate(poses,edges):
    valid=sorted([e for e in edges if e.valid],key=lambda e:(e.median,-e.overlap,e.i,e.j))
    while True:
        before=len(poses)
        for edge in valid:
            if edge.i in poses and edge.j not in poses:poses[edge.j]=poses[edge.i]@np.linalg.inv(edge.T_ij)
            elif edge.j in poses and edge.i not in poses:poses[edge.i]=poses[edge.j]@edge.T_ij
        if len(poses)==before:return


def _geometry_load_prior(files,cfg):
    folder=Path(cfg['_refine_existing']);manifest=json.loads((folder/'registration_run_manifest.json').read_text(encoding='utf8'))
    current=[(Path(p).name,os.stat(p).st_size,os.stat(p).st_mtime_ns) for p in files]
    previous=[(v['name'],v['size_bytes'],v['mtime_ns']) for v in manifest['inputs']]
    if current!=previous:raise ValueError('Existing transforms do not match these input files')
    with np.load(folder/'transforms.npz',allow_pickle=False) as data:
        if 'depth_parameters' in data:raise ValueError('Calibrated output requires its depth model; use a new full registration instead of --refine-existing')
        use=[int(k) for k in data['use_frames']];reference=int(data['ref_frame'])
        poses={k:data['M_%02d'%k].copy() for k in use}
    for T in poses.values():
        if not np.isfinite(T).all() or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(T[:3,:3])<0:
            raise ValueError('Existing poses must be finite rigid transforms')
    return poses,reference


def _geometry_warm_kernels():
    # Run while independent CPU preprocessing is in flight. Compile the same
    # Float64 operations once before concurrent CUDA workers need them.
    begin=time.monotonic();rng=np.random.default_rng(960)
    P=rng.uniform(-5.,5.,(6000,3));P[:,2]=350.+.05*P[:,0]**2+.03*P[:,1]**2
    N=np.column_stack((-.1*P[:,0],-.06*P[:,1],np.ones(len(P))))
    N/=np.linalg.norm(N,axis=1)[:,None]
    source=_cloud_from_arrays(P,N);target=_cloud_from_arrays(P+[.025,-.012,.032],N)
    gpu=_gpu_backend()
    gpu.icp(source,target,.3,.7,max_iter=2)
    icp_pp(source,target,.5,.8,max_iter=2)
    gpu.evaluate(P,target,np.eye(4),.5,4.000001)
    candidates=[]
    for angle in np.arange(0.,360.,3.):
        T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([0.,0.,np.deg2rad(angle)]).as_matrix()
        candidates.append(dict(T=T))
    rank_two_sphere_candidates(gpu,P[:1024],target,candidates)
    gpu.sync();gpu.clear()
    return time.monotonic()-begin


_GEOMETRY_BASE_RUN=run


def run(files,cfg):
    global _DEPTH_MODEL,_DEPTH_INPUTS
    path=cfg.get('sphere_guide_path')
    if not path and (cfg.get('calibrate_depth') or cfg.get('surface_guide')):raise ValueError('Calibration and surface guide require --sphere-guide')
    if not path:return _GEOMETRY_BASE_RUN(files,cfg)
    if cfg.get('compute_backend')!='cuda':raise ValueError('Sphere-guide mode requires CUDA; omit --sphere-guide for CPU baseline mode')
    if cfg.get('scan_model_path'):raise ValueError('Use sphere guide or scan model, one at a time')
    import multiprocessing as mp
    out=Path(cfg['_outdir']);cfg['_files']=files
    _DEPTH_MODEL=None;_DEPTH_INPUTS=set()
    if cfg.get('calibrate_depth'):
        if cfg.get('_refine_existing'):raise ValueError('Depth calibration requires full registration; omit --refine-existing')
        if os.path.normcase(os.path.abspath(path)) in {os.path.normcase(os.path.abspath(p)) for p in files}:
            raise ValueError('Depth calibration guide must be outside the registered input list')
    (out/'prepared_frames').mkdir(exist_ok=True);(out/'sphere_features').mkdir(exist_ok=True)
    cfg['_benchmark_kind']='sphere-guide-refine-existing' if cfg.get('_refine_existing') else 'sphere-guide-full-registration'
    reference=safe_read_point_cloud(path);guide=reference.voxel_down_sample(cfg['hp_voxel'])
    guide.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg['hp_norm_radius'],max_nn=50))
    P,N=np.asarray(guide.points),np.asarray(guide.normals);step=max(1,len(P)//25000)
    spheres=detect_spheres(P[::step],N[::step],np.asarray(reference.points),radius=cfg['sphere_radius'])
    spheres=four_marker_subset(spheres)
    if spheres is None:raise ValueError('Guide must contain four reliable separated spheres of the selected approximate radius')
    cfg['_guide_spheres']=[dict(s,center=s['center'].tolist()) for s in spheres]
    distances=np.sort(pdist(np.array([s['center'] for s in spheres])))
    if np.min(np.diff(distances))<1.5:raise ValueError('Sphere layout is too ambiguous for unlabelled geometry matching')
    Path(out,'sphere_guide.json').write_text(json.dumps(dict(path=os.path.abspath(path),
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),spheres=cfg['_guide_spheres'],
        source='Measured anchor cloud only; calibration measurements are not read by this pipeline.'),indent=2),encoding='utf8')
    surface_target=guide if cfg.get('surface_guide') else None
    surface_path=cfg.get('surface_reference_path')
    if surface_path:
        if os.path.normcase(os.path.abspath(surface_path)) in {os.path.normcase(os.path.abspath(p)) for p in files}:
            raise ValueError('Surface reference must be outside the registered input list')
        surface_target=_DEPTH_BASE_READ(surface_path).voxel_down_sample(cfg['hp_voxel'])
        if surface_target.has_normals():surface_target.normalize_normals()
        else:surface_target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg['hp_norm_radius'],max_nn=50))
        if len(surface_target.points)<100:raise ValueError('Surface reference contains too few valid points')
        distances=cKDTree(np.asarray(surface_target.points)).query(P[::step])[0]
        overlap=float((distances<1.).mean())
        if overlap<.2:raise ValueError('Surface reference is not in sphere-guide coordinates or does not cover the same object')
        Path(out,'surface_reference.json').write_text(json.dumps(dict(path=os.path.abspath(surface_path),
            sha256=hashlib.sha256(Path(surface_path).read_bytes()).hexdigest(),guide_overlap=overlap,
            used_for_optimization=True,independent_accuracy_evidence=False,
            coordinates='Same fixed coordinate system as sphere guide'),indent=2),encoding='utf8')
    del reference,guide,P,N
    prior=None;ref=-1
    if cfg.get('_refine_existing'):prior,ref=_geometry_load_prior(files,cfg)
    todo=sorted(prior) if prior is not None else list(range(len(files)))
    cfg['_prepared_cache_files_before_run']=sum(os.path.isfile(_prepared_path(files[k],cfg)) for k in todo)
    jobs=_auto_jobs(cfg);features={};begin=time.monotonic()
    print('[sphere-guide] prepare and detect %d frames, %d CPU workers'%(len(todo),jobs),flush=True)
    engine_files=files
    cpu=ProcessPoolExecutor(max_workers=jobs,mp_context=mp.get_context('spawn'),initializer=_pipe_feature_init,initargs=(engine_files,cfg))
    try:
        futures=[cpu.submit(_geometry_prepare,(k,files[k],cfg)) for k in todo]
        cfg['_overlapped_cuda_warmup_seconds']=_geometry_warm_kernels()
        for future in as_completed(futures):
            k,features[k]=future.result()
            if len(features)%20==0 or len(features)==len(todo):
                print('[sphere-prepare] %d/%d %.1fs'%(len(features),len(todo),time.monotonic()-begin),flush=True)
        cfg['_prepare_seconds']=time.monotonic()-begin
        if cfg.get('calibrate_depth'):
            previous=dict(cfg);model=_geometry_fit_depth(features,spheres);cfg['_depth_model']=model
            Path(out,'depth_calibration.json').write_text(json.dumps(model,indent=2),encoding='utf8')
            print('[calibration] heldout center distances %.4f -> %.4f mm'%(model['validation']['heldout']['before_rmse'],model['validation']['heldout']['after_rmse']),flush=True)
            corrected=[cpu.submit(_geometry_calibrated_prepare,(k,files[k],cfg,previous,features[k])) for k in todo]
            features={}
            for future in as_completed(corrected):
                k,row=future.result();features[k]=row
            _DEPTH_MODEL=model;_DEPTH_INPUTS={os.path.normcase(os.path.abspath(p)) for p in files}
            print('[calibration] corrected points, normals, features and marker fits',flush=True)
        poses={k:np.array(v['match']['T']) for k,v in features.items() if v['match'] is not None}
        Path(out,'sphere_observations.json').write_text(json.dumps(features,indent=2),encoding='utf8')
        if len(poses)<2:raise ValueError('Too few frames have reliable marker correspondences; this guide does not support the data')
        print('[sphere-guide] %d direct globally constrained poses'%len(poses),flush=True)
        frames=_SharedFrames(files,cfg);raws=LazyFrameLoader(files,capacity=2);reg=PairwiseRegistrator(cfg);edges=[]
        if prior is not None:
            src=[];dst=[]
            for k in sorted(poses):
                match=features[k]['match'];A=np.array([features[k]['spheres'][j]['center'] for j in match['source_indices']])
                src.append(A@prior[k][:3,:3].T+prior[k][:3,3]);dst.append(np.array([spheres[j]['center'] for j in match['target_indices']]))
            G=kabsch(np.vstack(src),np.vstack(dst));poses={k:G@T for k,T in prior.items()}
        else:
            poses,first_checks=_geometry_refine_many(files,poses,cfg,spheres)
            Path(out,'sphere_initial_refinement.json').write_text(json.dumps(first_checks,indent=2),encoding='utf8')
            poses,two_checks=_geometry_two_marker_stage(files,poses,features,spheres,cfg,frames)
            del frames
            Path(out,'sphere_two_marker_checks.json').write_text(json.dumps(two_checks,indent=2),encoding='utf8')
            print('[sphere-guide] %d poses after constrained angular search'%len(poses),flush=True)
            pairs=_GeometryPairPool(cpu,engine_files,cfg)
            unknown=set(range(len(files)))-set(poses);tasks=set()
            for i in unknown:
                for offset in range(-8,9):
                    j=i+offset
                    if i!=j and 0<=j<len(files):tasks.add(tuple(sorted((i,j))))
            def result(i,j,value):
                if value is not None:edges.append(reg._make_edge(i,j,*value))
                return False
            try:
                # Prioritize nearby anchored neighbours; unresolved frames still
                # receive every remaining original window candidate and RANSAC.
                first=set()
                for i in sorted(unknown):
                    anchored=sorted((j for j in poses if 0<abs(i-j)<=8),key=lambda j:(abs(i-j),j))
                    for j in anchored[:4]:first.add(tuple(sorted((i,j))))
                    for j in unknown:
                        if 0<abs(i-j)<=1:first.add(tuple(sorted((i,j))))
                pairs.run(first,'fgr',result);_geometry_propagate(poses,edges)
                remaining=set(range(len(files)))-set(poses)
                fallback=[(i,j) for i,j in tasks-first if i in remaining or j in remaining]
                pairs.run(fallback,'fgr',result);_geometry_propagate(poses,edges)
                remaining=set(range(len(files)))-set(poses);repairs=set()
                for i in sorted(remaining):
                    candidates=[]
                    for j in sorted(poses):
                        if abs(i-j)>8:continue
                        a=pairs.engine.probes.get((i,j),(np.inf,0.));b=pairs.engine.probes.get((j,i),(np.inf,0.))
                        candidates.append((-min(a[1],b[1]),max(a[0],b[0]),abs(i-j),j))
                    for *_,j in sorted(candidates)[:2]:repairs.add(tuple(sorted((i,j))))
                pairs.run(repairs,'ransac',result);_geometry_propagate(poses,edges)
            finally:pairs.close()
            print('[sphere-guide] %d poses after local registration'%len(poses),flush=True)
            rescue_cfg=cfg
            poses,_,_=rescue_failed_frames(raws,poses,sorted(poses),rescue_cfg,edges=edges,
                weak_edges=[e for e in edges if not e.valid and e.consistent])
        poses,checks=_geometry_refine_many(files,poses,cfg,spheres)
        Path(out,'sphere_pose_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
        if surface_target is not None:poses=_geometry_surface_refine(files,poses,cfg,surface_target)
    finally:cpu.shutdown(wait=True,cancel_futures=True)
    use=sorted(poses);requested=cfg.get('sphere_output_frame');G=np.eye(4)
    if requested is not None:
        if requested not in poses:raise ValueError('Requested output frame was not registered')
        G=np.linalg.inv(poses[requested]);poses={k:G@T for k,T in poses.items()};ref=requested
    elif prior is not None and ref in poses:
        G=prior[ref]@np.linalg.inv(poses[ref]);poses={k:G@T for k,T in poses.items()}
    cfg['_sphere_output_from_guide']=G.tolist()
    Path(out,'coordinate_frame.json').write_text(json.dumps(dict(output_from_guide=G.tolist(),
        guide_from_output=np.linalg.inv(G).tolist(),output_input_frame=ref,guide_path=os.path.abspath(path),
        input_geometry='depth_corrected' if cfg.get('_depth_model') else 'raw',
        transform_order='depth_correction_then_rigid_matrix' if cfg.get('_depth_model') else 'rigid_matrix_only'),indent=2),encoding='utf8')
    reg._export_audit(edges,files)
    merged,clean,per_frame=Merger(cfg).merge_and_clean(raws,poses,use)
    scores=leave_one_out_scores(raws,poses,use,cfg);print_scores(scores,cfg)
    return raws,poses,use,scores,ref,merged,clean,per_frame


DEFAULT_CFG.update(sphere_radius=5.8)


def _geometry_input_files(folder):
    import re
    generated={'merged_cloud.ply','merged_cloud_clean.ply','merged_by_frame.ply',
               'source_style_main_before_rescue.ply','manual_target_before_rescue.ply'}
    files=sorted(glob.glob(os.path.join(folder,'*.ply')))
    excluded=[p for p in files if Path(p).name.lower() in generated or re.fullmatch(r'merged_step\d+\.ply',Path(p).name.lower())]
    if excluded:print('[input] excluded generated output files: '+', '.join(Path(p).name for p in excluded),flush=True)
    excluded=set(excluded)
    return [p for p in files if p not in excluded]

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
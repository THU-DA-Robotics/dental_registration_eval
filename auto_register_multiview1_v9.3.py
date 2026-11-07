# -*- coding: utf-8 -*-
"""
多视角点云融合 一体化脚本 (v9.3 Source-Main Metrics版)
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
--freeze-main 为可选，冻结最初主分量位姿以降低救援后全体抛光漂移风险。

"""
import os, sys, glob, json, time, argparse, csv, hashlib
from pathlib import Path
from collections import OrderedDict
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

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
    manual_rescue  = True,
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
    source_style_main = False,      # 用源分量构建法构建主图：BFS局部位姿+合并，跳过PoseGraph/multiway
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
    freeze_seed_main = False,     # 可选：只在source-main时固定初始主分量位姿
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
        'hp_enter_overlap', 'score_voxel', 'overlap_dist',
    )

    def __init__(self, files, cfg):
        outdir = cfg.get('_outdir', '.')
        self.path = cfg.get('pair_cache_path') or os.path.join(outdir, 'pair_results_baseline.jsonl')
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        spec = {
            'version': 'reference_v915_algo_2_stage_cache',
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
                        maxtasksperchild=max(1, int(cfg.get('worker_max_tasks', 24))))
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
        print('  [%s] 最终跳过 %d 对：%s' % (stage, len(tasks), tasks[:20]))
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
        dist, idx = tree.query(P, k=1)
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
        self._feat = {}
        self._icp = {}
        self._hp = {}
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
            T0 = ransac_init(sd, sf, td, tf, cfg['ransac_dist'], cfg['ransac_iter'])
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
        s_hp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=nr, max_nn=50))
        t_hp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=nr, max_nn=50))
        for thr in cfg['hp_icp_thr']:
            T, _ = icp_custom(s_hp, t_hp, thr, cfg['hp_keep_ratio'],
                              max_iter=cfg['hp_iter'], conv=5e-5, T_init=T)
        med, ov = evaluate_pair(s_hp, t_hp, T, 0.2, 0.3)
        if ov < cfg['hp_min_overlap'] or med < cfg['hp_min_resid']:
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
                        p = voxel_down(raws[k], cfg['score_voxel']); p.transform(Ms[k])
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
import pyvista as pv

class ManualAligner:
    SRC_COLOR = 'red'
    TGT_COLOR = 'darkgreen'

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _o3d_to_pv(pcd):
        pts = np.asarray(pcd.points)
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
        tgt_v = voxel_down(tgt_pcd, cfg['score_voxel'])
        tgt_v.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        tgt_show = o3d.geometry.PointCloud(tgt_v)
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
            n = min(len(Ps), len(Pt))
            print('    本次点选: 源 %d 点, 目标 %d 点 -> 有效 %d 对' % (len(Ps), len(Pt), n))
            if n < 4:
                print('    不足4对, 请重新点选')
                continue
            T = kabsch(Ps[:n], Pt[:n])
            src_icp = voxel_down(src_raw, cfg['icp_voxel'])
            for thr in [20.0, 10.0, 5.0]:
                T = icp_pp(src_icp, tgt_v, thr, keep_ratio=0.8, max_iter=80, T_init=T)
            for thr in cfg['pair_icp_thr']:
                T, _ = icp_custom(src_icp, tgt_v, thr, cfg['keep_ratio'], T_init=T)
            src_fine = voxel_down(src_raw, cfg['score_voxel'])
            T, _ = icp_custom(src_fine, tgt_v, 0.8, keep_ratio=0.75, max_iter=150, T_init=T)
            med, ov = evaluate_pair(src_raw, tgt_pcd, T, cfg['score_voxel'], cfg['overlap_dist'])
            print('  人工配准%s: 中位 %.3f mm, 重叠 %.0f%%' % (tag, med, 100 * ov))
            try:
                s = voxel_down(src_raw, 1.0); s.transform(T)
                s.paint_uniform_color([1.0, 0.2, 0.2])
                t = voxel_down(tgt_pcd, 1.0)
                t.paint_uniform_color([0.0, 0.55, 0.2])
                o3d.visualization.draw_geometries([s, t],
                    window_name='叠加检查%s: 红=源 深绿=主云' % tag)
            except Exception as e:
                print('  (无法弹叠加图: %s)' % e)
            print('  结构对上(无重影)选 y; 明显错位选 r 重选; 放弃选 n')
            try:
                cmd = input('  接受该配准? [y/r/n] ').strip().lower()
            except EOFError:
                cmd = 'y' if ov > 0.4 else 'n'
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
            p = voxel_down(raws[k], cfg['score_voxel']); p.transform(Ms[k])
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
            p = voxel_down(raws[k], cfg['score_voxel']); p.transform(Ms[k])
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
def rescue_failed_frames(raws, Ms, use, cfg, edges=None, weak_edges=None):
    aligner = ManualAligner(cfg)
    merger = Merger(cfg)
    use = list(use)

    outdir = cfg.get('_outdir', '.')
    snapshots = []
    step_i = [0]

    def build_merged(frames):
        pts = []
        for k in frames:
            p = voxel_down(raws[k], cfg['score_voxel']); p.transform(Ms[k])
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
        for k in frames:
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
            p = voxel_down(raws[k], cfg['score_voxel']); p.transform(poses[k])
            pts.append(np.asarray(p.points))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.vstack(pts))
        return pc.voxel_down_sample(cfg['score_voxel'])

    def internal_poses(comp):
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
        while dq:
            u = dq.popleft()
            for v, T_uv in adj[u]:
                if v not in T_loc:
                    T_loc[v] = T_loc[u] @ np.linalg.inv(T_uv)
                    dq.append(v)
        return T_loc

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
            _, _, V_sub = np.linalg.svd(np.asarray(sub_pcd.points) - c_sub)
            c_main = np.asarray(mg.points).mean(axis=0)
            _, _, V_main = np.linalg.svd(np.asarray(mg.points) - c_main)
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
        tgt_icp = voxel_down(tgt, cfg['icp_voxel'])
        if anchor and anchor.T_main_to_scan is not None and T_init is not None:
            T_init = anchor.T_main_to_scan @ T_init
        cands = []
        if T_init is not None:
            cands.append(T_init)
        if not cands and src_frame_idx is not None:
            T_weak = weak_init_for_frame(src_frame_idx)
            if T_weak is not None:
                if anchor and anchor.T_main_to_scan is not None:
                    T_weak = anchor.T_main_to_scan @ T_weak
                cands.append(T_weak)
        if not cands and len(src_pcd.points) > 5000:
            T_pca = pca_init_for_group(src_pcd, tgt)
            if T_pca is not None:
                cands.append(T_pca)
        if not cands:
            sd, sf = fpfh_feats(src_pcd, cfg['voxel_fpfh'], cfg['fpfh_radius'])
            mg_d, mg_f = fpfh_feats(tgt, cfg['voxel_fpfh'], cfg['fpfh_radius'])
            try:
                cands.append(ransac_init(sd, sf, mg_d, mg_f,
                                         cfg['ransac_dist'], cfg['ransac_iter']))
            except Exception:
                return None
        T = icp_pp(voxel_down(src_pcd, cfg['icp_voxel']), tgt_icp,
                   8.0, keep_ratio=0.9, max_iter=40, T_init=cands[0])
        T = icp_pp(voxel_down(src_pcd, cfg['icp_voxel']), tgt_icp,
                   4.0, keep_ratio=0.85, max_iter=60, T_init=T)
        for thr in cfg['pair_icp_thr']:
            T, _ = icp_custom(voxel_down(src_pcd, cfg['icp_voxel']),
                              tgt_icp, thr, cfg['keep_ratio'], T_init=T)
        med, ov = evaluate_pair(src_pcd, tgt, T, cfg['score_voxel'], cfg['overlap_dist'])
        return T, med, ov

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
                if nbr in seen or nbr in use or nbr in declined:
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

    for round_i in range(cfg.get('rescue_rounds', 99)):
        progress = False
        for comp in list(groups):
            T_loc = internal_poses(comp)
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
                T_map = {k: T_sub @ T_loc[k] for k in comp}
                refined = refine_frames_to_main(comp, T_map, mg)
                for k in comp:
                    Ms[k] = refined[k]
                use.extend(comp); use.sort()
                for k in comp:
                    if k in failed:
                        failed.remove(k)
                groups.remove(comp)
                update_main()
                progress = True
                print('    -> 分量 %s 救回(已逐帧精修), 主图扩至 %d 帧' % (comp, len(use)))
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
        try:
            _, mg_show, _ = merger.merge_and_clean(raws, Ms, use)
        except Exception:
            mg_show = mg
        if target[0] == 'g':
            comp = target[1]
            T_loc = internal_poses(comp)
            sub_pcd = merge_with_poses(comp, T_loc)
            try:
                Tm = aligner.manual_align(sub_pcd, mg_show, tag=' 分量%s' % comp)
            except Exception as e:
                Tm = None
                print('    (无显示环境, 跳过人工: %s)' % e)
                break
            if Tm is not None:
                T_map = {k: Tm @ T_loc[k] for k in comp}
                refined = refine_frames_to_main(comp, T_map, mg)
                for k in comp:
                    Ms[k] = refined[k]
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
                Tm = aligner.manual_align(raws[k], mg_show, tag=' 帧%d' % k)
            except Exception as e:
                Tm = None
                print('    (无显示环境, 跳过人工: %s)' % e)
                break
            if Tm is not None:
                refined = refine_frames_to_main([k], {k: Tm}, mg)
                Ms[k] = refined[k]
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
            print('[rescue] 主图冻结：跳过救援后的全体multiway，只保留新分量的刚体对齐与逐帧精修')
        else:
            print('[rescue] 全体 %d 帧统一 multiway 抛光...' % len(use))
            Ms = GlobalOptimizer(cfg)._multiway_icp(raws, use, Ms)
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
            p = voxel_down(raws[k], cfg['score_voxel']); p.transform(Ms[k])
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
            p = voxel_down(raws[k], cfg['score_voxel']); p.transform(Ms[k])
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
        p = voxel_down(raws[k], cfg['score_voxel']); p.transform(Ms[k])
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
        p = voxel_down(raws[k], cfg['score_voxel']); p.transform(Ms[k])
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
#  Source-style main: 与 rescue_failed_frames.internal_poses/merge_with_poses
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
    cs = set(comp)
    print('  [source-main] 选择分量 %d，帧数=%d，成员=%s' % (chosen, len(comp), comp))
    if len(comp) < 2:
        print('  [source-main] 所选分量只有1帧，无法形成多帧主图')
        return None, comp, comp[0]

    # 以下采用 rescue_failed_frames.internal_poses() 相同的方程和 BFS 规则。
    deg = {k: 0 for k in comp}
    adj = {k: [] for k in comp}
    for e in edges:
        if e.valid and e.i in cs and e.j in cs:
            adj[e.i].append((e.j, e.T_ij))
            adj[e.j].append((e.i, np.linalg.inv(e.T_ij)))
            deg[e.i] += 1
            deg[e.j] += 1

    anchor = max(comp, key=lambda k: deg[k])
    T_loc = {anchor: np.eye(4)}
    dq = deque([anchor])
    tree_edges = set()
    while dq:
        u = dq.popleft()
        for v, T_uv in adj[u]:
            if v not in T_loc:
                T_loc[v] = T_loc[u] @ np.linalg.inv(T_uv)
                tree_edges.add((min(u, v), max(u, v)))
                dq.append(v)

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
    print('  [source-main] 用源分量的BFS局部位姿构建主云；不执行PoseGraph/multiway抛光')
    return T_loc, sorted(T_loc), anchor

def export_source_style_main_cloud(raws, Ms, use, cfg):
    """按 rescue_failed_frames.merge_with_poses/build_merged 原式导出主图。
    不进行 cover_min 清洗，便于观察纯粹的配准几何结果。
    """
    if Ms is None or not use:
        return None
    pts = []
    for k in use:
        p = voxel_down(raws[k], cfg['score_voxel'])
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
        p = voxel_down(raws[k], voxel)
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
        target = np.vstack([world[j] for j in candidates]) if candidates else np.empty((0, 3))
        tree = cKDTree(target) if len(target) else None
        matched, overlap, nsrc = _metric_directed(P, tree, threshold)
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
        'script_version': 'source_main_v9_3_metrics',
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
    print('输出目录: %s' % outdir)
    return mf, mc

def main():
    ap = argparse.ArgumentParser(description='多视角点云融合 (v9.3 source-main + metrics audit)')
    ap.add_argument('folder', help='存放 ply 的文件夹')
    ap.add_argument('--strategy', choices=['multiway', 'star'], default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    ap.add_argument('--no-rescue', action='store_true')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--exclude', nargs='*', type=int, default=[])
    ap.add_argument('--remerge', action='store_true')
    ap.add_argument('--scan-model', default=None,
                    help='完整口扫模型 ply 路径: 主图先锚定到模型, 失败帧对口扫配准')
    ap.add_argument('--reuse-pairs', action='store_true',
                    help='从匹配文件/参数的缓存读取已完成的pair，避免重算RANSAC')
    ap.add_argument('--no-precheck', action='store_true',
                    help='跳过启动前PLY完整性检查（不建议）')
    ap.add_argument('--precheck-only', action='store_true',
                    help='只检查PLY文件完整性后退出')
    ap.add_argument('--source-main', action='store_true',
                    help='以rescue中源分量的局部BFS方法构建主图，不运行PoseGraph/multiway')
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
    ap.add_argument('--freeze-main', action='store_true',
                    help='源分量模式中冻结初始主图；救援结束不再全体multiway抛光')
    ap.add_argument('--gate-enforce', action='store_true',
                    help='启用修复自身匹配错误后的自动守门（可能改变最终结果）')
    ap.add_argument('--gate-legacy', action='store_true',
                    help='按旧版原样运行守门，以便A/B对照；含自身匹配缺陷')
    ap.add_argument('--gate-off', action='store_true', help='彻底关闭局部守门')
    ap.add_argument('--jobs', type=int, default=None,
                    help='并行进程数(默认 CPU核数-1; 1=回退顺序模式)')
    args = ap.parse_args()

    print('[fusion v9.3] 已验证源分量配准 + 定量误差报告 + 可选冻结/守门')
    cfg = dict(DEFAULT_CFG)
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding='utf-8') as f:
            cfg.update(json.load(f))
    if args.strategy:
        cfg['strategy'] = args.strategy
    if args.scan_model:
        cfg['scan_model_path'] = args.scan_model
    if args.no_rescue:
        cfg['manual_rescue'] = False
    if args.jobs is not None:
        cfg['n_jobs'] = args.jobs
    if args.reuse_pairs:
        cfg['reuse_pairs'] = True
    if args.no_metrics:
        cfg['metrics_enable'] = False
    if args.reference_ply:
        cfg['metrics_reference_ply'] = args.reference_ply
    if args.freeze_main:
        cfg['freeze_seed_main'] = True
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
    if args.source_main:
        cfg['source_style_main'] = True
        # 可按 --initial-only 先检查主图；未设 initial-only 时继续原版救援逻辑。
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

    files = sorted(glob.glob(os.path.join(args.folder, '*.ply')))
    assert files, '文件夹内没有 ply'
    outdir = os.path.join(args.folder, 'multiview_output')
    os.makedirs(outdir, exist_ok=True)
    cfg['_outdir'] = outdir
    if cfg.get('freeze_seed_main', False) and not cfg.get('source_style_main', False) and not args.remerge:
        ap.error('--freeze-main 必须配合 --source-main')
    if cfg.get('freeze_seed_main', False) and cfg.get('gate_mode') == 'enforce':
        ap.error('冻结初始主图不能和 --gate-enforce 同时使用')

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

    if cfg.get('precheck_ply', True) and not args.remerge:
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

    t_main_begin = time.monotonic()
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
    print('输出目录: %s' % outdir)

    if args.show:
        try:
            o3d.visualization.draw_geometries([mc], window_name='合并结果(干净版)')
        except Exception as e:
            print('(无显示环境, 跳过可视化: %s)' % e)

if __name__ == '__main__':
    main()
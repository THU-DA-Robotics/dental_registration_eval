# -*- coding: utf-8 -*-
"""
多视角点云融合 一体化脚本 (v2.9.3 模型锚定修正版)
=====================================================================
v2.9.3 修正:
  ★ 口扫仅用于定位, 不进入最终合并结果;
  ★ 帧对口扫验收: 重残差轻重叠(中位<0.5mm即接受), 不再用重叠率误杀;
  ★ 锚定只是初始化: 主图锚定后, 失败帧/分量对口扫配准, 配不上仍走
    完整抢救流程(架桥/人工兜底), 只是目标从主图换成口扫;
  ★ 对口扫精修阈值放宽到 0.5mm。
"""
import os, sys, glob, json, time, argparse
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import open3d as o3d
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
    ransac_iter    = 200000,
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
    edge_overlap_min = 0.25,
    edge_med_max     = 1.2,
    mw_icp_thr       = [2.0, 1.0, 0.5],
    mw_keep_ratio    = 0.7,
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
    scan_ref_frame  = 0,
    scan_accept_median = 0.5,   # 帧对口扫接受标准: 中位残差<0.5mm(牙齿部分对准即可)
)

# ================================================================
#  通用几何工具
# ================================================================
def voxel_down(pcd, vx):
    return pcd.voxel_down_sample(vx)

def fpfh_feats(pcd, voxel, radius):
    d = voxel_down(pcd, voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
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
#  模块1: 两两配准
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

class PairwiseRegistrator:
    def __init__(self, cfg):
        self.cfg = cfg
        self._feat = {}
        self._icp = {}
        self._hp = {}

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

    def _align_pair(self, raws, i, j):
        cfg = self.cfg
        sd, sf = self._feats(raws, i)
        td, tf = self._feats(raws, j)
        try:
            T0 = ransac_init(sd, sf, td, tf, cfg['ransac_dist'], cfg['ransac_iter'])
        except Exception:
            return None
        med, ov = evaluate_pair(raws[i], raws[j], T0, cfg['score_voxel'], cfg['overlap_dist'])
        if not cfg.get('pair_high_precision', True):
            T = T0
            for thr in cfg['pair_icp_thr']:
                T, _ = icp_custom(self._icp_cloud(raws, i), self._icp_cloud(raws, j),
                                  thr, cfg['keep_ratio'], T_init=T)
            med, ov = evaluate_pair(raws[i], raws[j], T, cfg['score_voxel'], cfg['overlap_dist'])
            med_r, ov_r = evaluate_pair(raws[j], raws[i], np.linalg.inv(T),
                                        cfg['score_voxel'], cfg['overlap_dist'])
            consistent = (min(ov, ov_r) > 0.35) or (max(ov, ov_r) > 0.6 and min(ov, ov_r) > 0.25)
            return T, med, ov, consistent
        if ov < 0.25 or med > 2.0:
            return T0, med, ov, True
        s_mid = self._icp_cloud(raws, i); t_mid = self._icp_cloud(raws, j)
        T, _ = icp_custom(s_mid, t_mid, 2.0, 0.8, max_iter=80, T_init=T0)
        s_hp = self._icp_cloud_hp(raws, i); t_hp = self._icp_cloud_hp(raws, j)
        nr = cfg['hp_norm_radius']
        s_hp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=nr, max_nn=50))
        t_hp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=nr, max_nn=50))
        for thr in cfg['hp_icp_thr']:
            T, _ = icp_custom(s_hp, t_hp, thr, cfg['hp_keep_ratio'],
                              max_iter=cfg['hp_iter'], conv=5e-5, T_init=T)
        med, ov = evaluate_pair(s_hp, t_hp, T, 0.2, 0.3)
        if ov < cfg['hp_min_overlap'] or med < cfg['hp_min_resid']:
            med_lo, ov_lo = evaluate_pair(raws[i], raws[j], T, cfg['score_voxel'], cfg['overlap_dist'])
            return T, med_lo, ov_lo, False
        med_r, ov_r = evaluate_pair(raws[j], raws[i], np.linalg.inv(T),
                                    cfg['score_voxel'], cfg['overlap_dist'])
        consistent = (min(ov, ov_r) > 0.35) or (max(ov, ov_r) > 0.6 and min(ov, ov_r) > 0.25)
        return T, med, ov, consistent
    
    def register_all_pairs(self, raws):
        # ... 原有逻辑 ...
        for i in range(n):
            # ... 原有候选逻辑 ...
            for j in cands:
                # ... 原有早停/尝试逻辑 ...
                res = self._align_pair(raws, i, j)
                if res is None:
                    continue
                e = self._make_edge(i, j, *res)
                edges.append(e)
                # ★ 弱边也缓存(残差<1.0mm 且 重叠>30% 的边, 即使没过阈值)
                if not e.valid and e.median < 1.0 and e.overlap > 0.3:
                    weak_edges.append(e)   # 新增: 弱边缓存列表
                    print('    对(%2d,%2d): 中位 %.3f, 重叠 %.0f%% [弱边缓存]'
                          % (i, j, e.median, 100 * e.overlap))
                # ... 原有打印/统计逻辑 ...
        return self._bridge_components(raws, edges, centers, radii), weak_edges

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

    def register_all_pairs(self, raws):
        cfg = self.cfg
        n = len(raws)
        centers, radii = self._aabb_stats(raws)
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

        edges, have = [], set()
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
                tag = 'OK' if e.valid else ('假阳性' if not e.consistent else 'weak')
                print('    对(%2d,%2d): 中位 %.3f, 重叠 %.0f%% %s'
                      % (i, j, e.median, 100 * e.overlap, tag))
                if e.valid:
                    edge_cnt[i] += 1; edge_cnt[j] += 1
                    best_ov[i] = max(best_ov[i], e.overlap)
                    best_ov[j] = max(best_ov[j], e.overlap)
        print('  [pairwise] 尝试 %d 对(全量 %d), 有效边 %d, 耗时 %.0fs'
              % (trials, n * (n - 1) // 2, sum(e.valid for e in edges), time.time() - t0))
        return self._bridge_components(raws, edges, centers, radii)

    def _bridge_components(self, raws, edges, centers, radii):
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
        print('  [pairwise] 图被切成 %d 个分量(大小 %s), 尝试架桥...'
              % (len(comps), [len(c) for c in comps]))
        main = list(comps[0])
        for other in comps[1:]:
            pairs = [(i, j) for i in main for j in other
                     if np.linalg.norm(centers[i] - centers[j]) <= 2.0 * (radii[i] + radii[j])]
            pairs.sort(key=lambda ij: (min(abs(ij[0] - ij[1]), n - abs(ij[0] - ij[1])),
                                       np.linalg.norm(centers[ij[0]] - centers[ij[1]])))
            bridged = False
            for i, j in pairs[:cfg['pair_max_trials'] * 2]:
                res = self._align_pair(raws, i, j)
                if res is None:
                    continue
                e = self._make_edge(i, j, *res)
                edges.append(e)
                if e.valid:
                    print('    架桥成功: (%d,%d) 中位 %.3f, 重叠 %.0f%%'
                          % (i, j, e.median, 100 * e.overlap))
                    bridged = True
                    break
            if bridged:
                main = main + list(other)
            else:
                print('    分量 %s 架桥失败, 稍后走分量级抢救' % other)
        return edges

# ================================================================
#  模块2: 全局优化
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
        for thr in cfg['mw_icp_thr']:
            for _ in range(3):
                moved = 0.0
                world = {}
                for k in use:
                    p = voxel_down(raws[k], cfg['score_voxel']); p.transform(Ms[k])
                    world[k] = np.asarray(p.points)
                for k in use:
                    others = np.vstack([world[j] for j in use if j != k])
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
#  模块2.5: 模型锚定配准(主图先行, 失败帧对口扫)
# ================================================================
class ScanAnchorRegistrator:
    """口扫模型作为全局基准: 主图先配准到模型, 之后失败帧/分量直接对模型配准,
    落进模型坐标系, 和主图天然对齐。无环闭合, 误差不累积。
    口扫仅用于定位, 不进入最终合并结果。"""

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
        """把任意点云对齐到口扫模型。返回 (T_src_to_scan, med, ov)。
        验收由调用方决定: 重残差轻重叠(牙齿部分对准即可)。"""
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
        """主图(干净合并云)配准到口扫模型。自动先试, 失败则手动。"""
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
        """主图先由两两配准+全局优化得到, 锚定到口扫;
        失败帧/分量对口扫配准(牙齿部分对准即可), 返回 (Ms, use, edges) 供抢救。"""
        cfg = self.cfg
        n = len(raws)
        edges = PairwiseRegistrator(cfg).register_all_pairs(raws)
        go = GlobalOptimizer(cfg)
        Ms, use = go.optimize(n, edges, raws)
        if Ms is None or len(use) < 2:
            return None, [], edges
        merger = Merger(cfg)
        _, mg_clean, _ = merger.merge_and_clean(raws, Ms, use)
        if not self.anchor_main_cloud(mg_clean, cfg):
            return None, [], edges
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
        return Ms, use, edges

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
#  失败帧抢救(模型锚定: 主图先锚定到口扫, 失败帧对口扫配准)
# ================================================================
def rescue_failed_frames(raws, Ms, use, cfg, edges=None):
    aligner = ManualAligner(cfg)
    merger = Merger(cfg)
    use = list(use)

    outdir = cfg.get('_outdir', '.')
    snapshots = []
    step_i = [0]
    weak_map = {}   # k -> [弱边列表]
    if weak_edges:
        for e in weak_edges:
            weak_map.setdefault(e.i, []).append(e)
            weak_map.setdefault(e.j, []).append(e)

    # ★ 新增: 手动架桥成功后, 自动重试该分量/帧的弱边邻居

    def retry_weak_neighbors(just_added_frames):
        """手动救回后, 用新位姿做初值, 自动重试其弱边邻居"""
        if not weak_map:
            return 0
        retried = 0
        for k in just_added_frames:
            if k not in weak_map:
                continue
            for e in weak_map[k]:
                # 找邻居: e.i 或 e.j 是 k, 另一个是目标
                nbr = e.j if e.i == k else e.i
                if nbr in use or nbr in declined:
                    continue
                # 用 k 的新位姿 + 弱边初值 作为 nbr 的初值
                # W_nbr ≈ W_k @ T_knbr  (T_knbr 是弱边)
                T_k_nbr = e.T_ij if e.i == k else np.linalg.inv(e.T_ij)
                T_init = Ms[k] @ T_k_nbr
                # 对口扫/主图自动配准
                res = align_cloud_auto(raws[nbr], mg, T_init=T_init)
                if res is not None:
                    T, med, ov = res
                    if ov > cfg['rescue_accept_overlap'] and med < cfg['rescue_accept_median']:
                        refined = refine_frames_to_main([nbr], {nbr: T}, mg)
                        Ms[nbr] = refined[nbr]
                        use.append(nbr); use.sort()
                        if nbr in failed:
                            failed.remove(nbr)
                        update_main()
                        retried += 1
                        print('      [弱边借势] 帧%d -> 帧%d: 中位 %.3f, 重叠 %.0f%%, 救回'
                              % (k, nbr, med, 100 * ov))
        return retried

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

    # ---- 若启用模型锚定: 主图先锚定到口扫, 后续失败帧对口扫配准 ----
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
    if not failed or len(use) < 2:
        return Ms, sorted(use), 0
    print('[rescue] %d 帧未自动接入: %s' % (len(failed), failed))

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
        anchor = max(comp, key=lambda k: deg[k])
        T_loc = {anchor: np.eye(4)}
        dq = deque([anchor])
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

    def align_cloud_auto(src_pcd, mg, T_init=None):
        # 目标: 有锚定则对口扫, 否则对主图
        tgt = anchor.scan_pcd if (anchor and anchor.T_main_to_scan is not None) else mg
        tgt_icp = voxel_down(tgt, cfg['icp_voxel'])
        # T_init 坐标系换算: 边传播初值是"帧->主图", 锚定后要换成"帧->口扫"
        if anchor and anchor.T_main_to_scan is not None and T_init is not None:
            T_init = anchor.T_main_to_scan @ T_init
        cands = []
        if T_init is not None:
            cands.append(T_init)
        else:
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
        # 目标: 有锚定则对口扫, 否则对主图
        tgt = anchor.scan_pcd if (anchor and anchor.T_main_to_scan is not None) else mg
        tgt_icp = voxel_down(tgt, cfg['icp_voxel'])
        out = {}
        for k in frames:
            T = T_map[k]
            s_icp = voxel_down(raws[k], cfg['icp_voxel'])
            # 对口扫时精修到 0.5mm(牙齿部分对准); 对主图时 2.0/1.0
            thr_seq = [1.0, 0.5] if (anchor and anchor.T_main_to_scan is not None) else [2.0, 1.0]
            for thr in thr_seq:
                T, _ = icp_custom(s_icp, tgt_icp, thr, 0.7, max_iter=60, T_init=T)
            med, ov = evaluate_pair(raws[k], tgt, T, cfg['score_voxel'], cfg['overlap_dist'])
            out[k] = T
            print('      帧%2d 精修后: 中位 %.3f, 重叠 %.0f%%' % (k, med, 100 * ov))
        return out

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
            res = align_cloud_auto(raws[k], mg, T_init=T_init)
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
                # ★ 手动架桥成功, 立刻自动重试该分量的弱边邻居
                n_retried = retry_weak_neighbors(comp)
                if n_retried > 0:
                    print('      [弱边借势] 手动架桥后自动救回 %d 个邻居帧' % n_retried)
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
                # ★ 手动架桥成功, 立刻自动重试该帧的弱边邻居
                n_retried = retry_weak_neighbors([k])
                if n_retried > 0:
                    print('      [弱边借势] 手动架桥后自动救回 %d 个邻居帧' % n_retried)
                print('    -> 手动架桥成功: 帧%d接入, 自动重试剩余' % k)
            else:
                declined.add(k)

    left = [k for k in range(len(raws)) if k not in set(use)]
    if left:
        print('  [rescue] 最终仍失败: %s' % left)

    if len(use) > n0 and len(use) >= 2:
        print('[rescue] 全体 %d 帧统一 multiway 抛光...' % len(use))
        Ms = GlobalOptimizer(cfg)._multiway_icp(raws, use, Ms)
    return Ms, sorted(use), len(use) - n0

# ================================================================
#  局部区域守门
# ================================================================
def consistency_gate_local(raws, Ms, use, cfg):
    if not cfg.get('gate_enable', True) or len(use) < 3:
        return Ms, use
    use = list(use)
    block_size = cfg.get('gate_block_size', 15.0)
    min_block_pts = cfg.get('gate_block_min_pts', 200)
    mad_k = cfg.get('gate_mad_k', 3.0)

    for rd in range(cfg.get('gate_max_rounds', 3)):
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
#  主流程
# ================================================================
def run(files, cfg):
    n = len(files)
    print('[1] 读取 %d 帧...' % n)
    raws = [o3d.io.read_point_cloud(f) for f in files]

    if cfg.get('scan_model_path'):
        print('[2] 配准策略 = 模型锚定(用完整口扫做基准)')
        anchor = ScanAnchorRegistrator(cfg)
        anchor.load_scan(cfg['scan_model_path'])
        Ms, use, edges = anchor.register(raws, files)
        if Ms is None or len(use) < 2:
            print('  锚定模式失败, 回退 multiway')
            cfg['scan_model_path'] = ''
            edges = None
        else:
            # 锚定只负责初始化, 后续抢救/架桥/人工兜底照常
            Ms, use, n_rescued = rescue_failed_frames(raws, Ms, use, cfg, edges=edges)
            print('  配准成功 %d/%d 帧' % (len(use), n))
            if len(use) < 2:
                return raws, Ms, use, {}, 0, None, None, None
            mf, mc, per_frame_t = Merger(cfg).merge_and_clean(raws, Ms, use)
            scores = leave_one_out_scores(raws, Ms, use, cfg)
            print_scores(scores, cfg)
            return raws, Ms, use, scores, 0, mf, mc, per_frame_t

    edges = None
    ref = -1
    if cfg['strategy'] == 'multiway':
        print('[2] 配准策略 = pairwise + global multiway ICP')
        edges, weak_edges = PairwiseRegistrator(cfg).register_all_pairs(raws)   # 接住弱边        
        go = GlobalOptimizer(cfg)        
        Ms, use = go.optimize(n, edges, raws)
        ref = go.pick_anchor(n, edges) if use else -1
        if Ms is None or len(use) < 2:
            print('  multiway 失败, 回退 star')
            cfg['strategy'] = 'star'
    if cfg['strategy'] == 'star':
        print('[2] 配准策略 = star(旧版星型拓扑)')
        Ms, use, n_rescued = rescue_failed_frames(raws, Ms, use, cfg,
                                              edges=edges, weak_edges=weak_edges)   # 传下去

    Ms, use, n_rescued = rescue_failed_frames(raws, Ms, use, cfg, edges=edges)
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
             score_median=np.array([scores[k][0] for k in use]) if scores else np.array([]),
             score_overlap=np.array([scores[k][1] for k in use]) if scores else np.array([]))

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
    print('输出目录: %s' % outdir)
    return mf, mc

def main():
    ap = argparse.ArgumentParser(description='多视角点云融合 (v2.9.3)')
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
    args = ap.parse_args()

    print('[fusion v2.9.3] 高精两两 + 跨帧边 + 局部守门 + 模型锚定(口扫仅定位不进结果)')
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

    if args.remerge:
        mf, mc = remerge(files, cfg, outdir, args.exclude)
        if args.show:
            try:
                o3d.visualization.draw_geometries([mc], window_name='合并结果(干净版)')
            except Exception as e:
                print('(无显示环境: %s)' % e)
        return

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
    print('输出目录: %s' % outdir)

    if args.show:
        try:
            o3d.visualization.draw_geometries([mc], window_name='合并结果(干净版)')
        except Exception as e:
            print('(无显示环境, 跳过可视化: %s)' % e)

if __name__ == '__main__':
    main()
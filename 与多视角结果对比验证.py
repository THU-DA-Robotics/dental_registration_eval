# -*- coding: utf-8 -*-
"""
与多视角结果对比验证
====================
用途：把你之前单帧代码（auto_register2.py 等）的结果，与 47 帧多视角配准/拟合的
     "参考解" 做定量对比，验证旧代码的精度。

坐标系约定：多视角结果（合并点云 / 变换矩阵 / 车针拟合）全部在
    帧6 (point_cloud_20260910_120802_225.ply) 的原始坐标系 下。
    配准变换矩阵.npz 里的 M_XX 把 第XX帧原始坐标 -> 帧6坐标系。

用法示例：
  1) 验证你的单帧车针拟合（假设你处理的是第 12 帧，结果在该帧自身坐标系）：
       python 与多视角结果对比验证.py --frame 12 \
           --tip  12.3 -45.6 290.1  --dir  -0.58 0.49 -0.65
  2) 验证你的某帧配准矩阵（4x4 .npy，映射 帧k -> 你的参考系）：
       python 与多视角结果对比验证.py --frame 12 --M 你的矩阵.npy
     注意：若你的参考系不是帧6，脚本会先通过"你参考系对应帧"的矩阵做链式换算，
           用 --your-ref 指定你的参考帧号（默认与 --frame 相同，即假设恒等）。
"""
import argparse
import os
import numpy as np
import json

def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 1)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='配准变换矩阵.npz')
    ap.add_argument('--json', default='车针拟合结果.json')
    ap.add_argument('--frame', type=int, required=True, help='你的结果对应的帧号')
    ap.add_argument('--tip', nargs=3, type=float, help='你的车针尖端 (该帧坐标系, mm)')
    ap.add_argument('--dir', nargs=3, type=float, help='你的车针轴向 (指向尖端)')
    ap.add_argument('--M', help='你的该帧配准矩阵 .npy (帧k -> 你的参考系)')
    ap.add_argument('--your-ref', type=int, default=None,
                    help='你的参考帧号（默认=--frame，即认为你的结果就是帧k自身坐标）')
    args = ap.parse_args()

    for f in (args.npz, args.json):
        if not os.path.exists(f):
            print('错误：找不到文件 %s' % f)
            print('请把【配准变换矩阵.npz】和【车针拟合结果.json】下载后放到本脚本同目录，')
            print('或用 --npz / --json 指定完整路径，例如：')
            print('  python 与多视角结果对比验证.py --frame 12 --npz D:\\data\\配准变换矩阵.npz --json D:\\data\\车针拟合结果.json')
            return
    if args.tip is None and args.M is None:
        print('提示：你没有给任何待验证的结果，只会打印参考解。')
        print('  验证车针拟合: 加 --tip x y z --dir dx dy dz（你旧代码的输出，该帧自身坐标系）')
        print('  验证配准矩阵: 加 --M 你的矩阵.npy')
        print()

    z = np.load(args.npz)
    use = [int(k) for k in z['use_frames']]
    Ms = {k: z['M_%02d' % k] for k in use}
    if 'dropped_frames' in z.files:
        print('多视角流程舍弃的帧:', [int(k) for k in z['dropped_frames']])
    if args.frame not in Ms:
        print('错误：第%d帧不在成功配准的帧列表里 %s' % (args.frame, use))
        return
    ref = json.load(open(args.json, encoding='utf-8'))
    tip_ref = np.array(ref['尖端位置_mm'])
    dir_ref = np.array(ref['轴线方向_指向尖端'])
    cen_ref = np.array(ref['轴心参考点_mm'])
    print('多视角参考解: tip=%s, dir=%s' % (np.round(tip_ref, 3), np.round(dir_ref, 4)))

    # 你的帧k -> 帧6 的链式变换：M_chain = Ms[your_ref] @ 你的M（若你的M参考系=帧k自身则 M=I）
    your_ref = args.your_ref if args.your_ref is not None else args.frame
    if your_ref not in Ms:
        print('错误：你的参考帧%d不在成功列表里' % your_ref)
        return

    if args.M:
        M_yours = np.load(args.M)
        M_chain = Ms[your_ref] @ M_yours
        M_ours = Ms[args.frame]
        dR = M_chain[:3, :3] @ M_ours[:3, :3].T
        dang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        dt = np.linalg.norm(M_chain[:3, 3] - M_ours[:3, 3])
        print('\n[配准对比] 帧%d' % args.frame)
        print('  旋转差 %.4f deg, 平移差 %.4f mm' % (dang, dt))
        print('  (多视角整体配准精度: 中位残差 %.3f mm 量级，差异在此量级内即一致)'
              % float(np.mean(z['score_median'])))
        T_use = M_chain
    else:
        T_use = Ms[args.frame]

    if args.tip is not None and args.dir is not None:
        tip = np.array(args.tip, float)
        d = np.array(args.dir, float); d /= np.linalg.norm(d)
        tip_w = T_use[:3, :3] @ tip + T_use[:3, 3]
        dir_w = T_use[:3, :3] @ d
        dir_w /= np.linalg.norm(dir_w)
        # 统一朝向后再比
        s = np.sign(dir_w @ dir_ref) or 1.0
        dir_al = dir_w * s
        # 两轴线的间距（比单比tip更稳）
        n = np.cross(dir_al, dir_ref)
        if np.linalg.norm(n) > 1e-9:
            n /= np.linalg.norm(n)
            line_gap = abs((tip_w - cen_ref) @ n)
        else:
            line_gap = np.linalg.norm(np.cross(tip_w - cen_ref, dir_ref))
        print('\n[车针拟合对比] 帧%d -> 帧6坐标系' % args.frame)
        print('  你的尖端(变换后) = %s' % np.round(tip_w, 3))
        print('  尖端偏差   = %.3f mm' % np.linalg.norm(tip_w - tip_ref))
        print('  轴向夹角   = %.3f deg' % angle_deg(dir_w, dir_ref))
        print('  轴线间距   = %.3f mm' % line_gap)
        print('  参考: 多视角拟合壁面残差中位 %.3f mm，尖端偏差在此量级内即吻合'
              % ref['壁面残差中位_mm'])

if __name__ == '__main__':
    main()
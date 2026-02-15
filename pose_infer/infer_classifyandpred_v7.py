#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTMDet-Pose V7 推理脚本: 检测 + 姿态 + 跌倒分类 + ByteTrack 跟踪

支持两种模式:
  1. 图片文件夹 (单帧推理, 无跟踪)
  2. 视频文件 (逐帧推理 + ByteTrack 跟踪 + GRU 时序分类)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import numpy as np
import glob
import collections
import torch
import argparse

from mmdet.apis import init_detector
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.dataset import Compose


def parse_args():
    parser = argparse.ArgumentParser(
        description='RTMDet-Pose V7: Detection + Pose + Falling + Tracking')
    parser.add_argument('--config',
                        default='/home/tbai/mmdetection/mmdetection/rtmdet_pose_v7_withclass_track.py',
                        help='配置文件路径')
    parser.add_argument('--checkpoint',
                        default=None,
                        help='权重文件路径')
    parser.add_argument('--input',
                        default='/home/tbai/Desktop/SenIRDatasetProcessed/1A-7 Falling while walking/AAG/stitched_png',
                        help='输入图片文件夹或视频文件路径')
    parser.add_argument('--output',
                        default='/home/tbai/Desktop/inference_results_v7',
                        help='输出文件夹路径')
    parser.add_argument('--det-thr', type=float, default=0.3,
                        help='检测框置信度阈值')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
                        help='关键点置信度阈值')
    parser.add_argument('--fall-thr', type=float, default=0.5,
                        help='跌倒分类阈值')
    parser.add_argument('--motion-thr', type=float, default=0.03,
                        help='关键点运动幅度阈值，低于此值视为静止')
    parser.add_argument('--stable-frames', type=int, default=3,
                        help='连续静止帧数超过此值触发抑制检查')
    parser.add_argument('--spike-window', type=int, default=15,
                        help='回看N帧检查是否有运动高峰(摔倒动作)')
    parser.add_argument('--spike-thr', type=float, default=0.06,
                        help='运动高峰阈值，超过此值说明最近有摔倒动作')
    parser.add_argument('--use-tracker', action='store_true',
                        help='启用 ByteTrack 跟踪 + 时序 GRU (视频/连续帧)')
    parser.add_argument('--max-seq-len', type=int, default=30,
                        help='时序 GRU 最大帧缓冲长度')
    parser.add_argument('--show-kpt-idx', action='store_true',
                        help='是否显示关键点编号')
    parser.add_argument('--skeleton', type=str, nargs='*', default=None,
                        help='骨架连接，格式: 0-1 1-2 1-3 ...')
    parser.add_argument('--thickness', type=int, default=2,
                        help='线条粗细')
    parser.add_argument('--radius', type=int, default=3,
                        help='关键点显示半径')
    parser.add_argument('--save-video', action='store_true',
                        help='将结果保存为视频 (仅 tracker 模式)')
    parser.add_argument('--fps', type=int, default=10,
                        help='输出视频帧率')
    args = parser.parse_args()

    if args.skeleton is not None:
        parsed = []
        for s in args.skeleton:
            a, b = s.split('-')
            parsed.append((int(a), int(b)))
        args.skeleton = parsed

    return args


# ---- 推理 pipeline (不需要标注) ----
INFER_PIPELINE = [
    dict(type='LoadImageFromFile', _scope_='mmdet'),
    dict(type='Resize', scale=(384, 384), keep_ratio=False, _scope_='mmdet'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
         _scope_='mmdet'),
]

# ---- 颜色定义 ----
KPT_COLORS = [
    (0, 0, 255),    # 0: 红
    (0, 128, 255),  # 1: 橙
    (0, 255, 255),  # 2: 黄
    (0, 255, 0),    # 3: 绿
    (255, 255, 0),  # 4: 青
    (255, 0, 0),    # 5: 蓝
    (255, 0, 255),  # 6: 紫
]
LIMB_COLOR = (0, 255, 128)
COLOR_NORMAL = (0, 255, 0)     #  - normal
COLOR_FALLING = (0, 0, 255)    #  - fall
COLOR_TRACK = [
    (255, 128, 0), (0, 255, 128), (128, 0, 255),
    (255, 255, 0), (0, 128, 255), (255, 0, 128),
    (128, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 128, 255),
]


def inference_single(model, img_path, pipeline):
    """单张图片推理"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    data_info = {
        'img_path': img_path,
        'img_id': 0,
        'ori_shape': (h, w),
        'height': h,
        'width': w,
    }
    data = pipeline(data_info)
    device = next(model.parameters()).device
    data['inputs'] = [data['inputs'].to(device)]
    data['data_samples'] = [data['data_samples'].to(device)]
    with torch.no_grad():
        results = model.test_step(data)
    return results, img


COLOR_LYING = (255, 165, 0)    # 橙色 - 躺着(被抑制的跌倒)

def visualize(img, pred, args, track_id=None, temporal_score=None,
              raw_score=None):
    """可视化检测框 + 关键点 + 跌倒状态

    Args:
        raw_score: GRU原始分数(未抑制), 用于区分 lying vs normal
    """
    img_vis = img.copy()

    bboxes = pred.bboxes.cpu().numpy()
    bbox_scores = pred.scores.cpu().numpy()
    keypoints = pred.keypoints.cpu().numpy() if hasattr(pred, 'keypoints') else np.zeros((len(bboxes), 0, 2))
    kpt_scores = pred.keypoint_scores.cpu().numpy() if hasattr(pred, 'keypoint_scores') else np.zeros((len(bboxes), 0))
    action_scores = pred.action_scores.cpu().numpy() if hasattr(pred, 'action_scores') else np.zeros(len(bboxes))

    for idx in range(len(bboxes)):
        det_score = bbox_scores[idx]
        act_score = temporal_score if temporal_score is not None else action_scores[idx]
        is_falling = act_score >= args.fall_thr

        # 判断是否被运动抑制 (raw_score高但act_score被衰减 → lying)
        is_suppressed = (raw_score is not None
                         and raw_score >= args.fall_thr
                         and not is_falling)

        # 框颜色: 跌倒=红, 躺着=橙, 正常=绿
        if is_falling:
            box_color = COLOR_FALLING
        elif is_suppressed:
            box_color = COLOR_LYING
        else:
            box_color = COLOR_NORMAL
        if track_id is not None and not is_falling and not is_suppressed:
            box_color = COLOR_TRACK[track_id % len(COLOR_TRACK)]

        x1, y1, x2, y2 = map(int, bboxes[idx][:4])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), box_color, args.thickness)

        # 标签文字
        label_parts = [f'{det_score:.2f}']
        if is_falling:
            label_parts.append(f'FALLING({act_score:.2f})')
        elif is_suppressed:
            label_parts.append(f'LYING({raw_score:.2f})')
        elif act_score > 0:
            label_parts.append(f'normal({act_score:.2f})')
        if track_id is not None:
            label_parts.insert(0, f'ID:{track_id}')
        label = ' '.join(label_parts)

        # 文字背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(img_vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 跌倒警告
        if is_falling:
            cv2.putText(img_vis, 'FALL DETECTED', (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FALLING, 2)

        if idx >= len(keypoints):
            continue
        kpts = keypoints[idx]
        scrs = kpt_scores[idx]

        # 骨架
        if args.skeleton:
            for (a, b) in args.skeleton:
                if a < len(kpts) and b < len(kpts):
                    if scrs[a] > args.kpt_thr and scrs[b] > args.kpt_thr:
                        pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                        pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                        cv2.line(img_vis, pt1, pt2, LIMB_COLOR, args.thickness)

        # 关键点
        for i, (kpt, score) in enumerate(zip(kpts, scrs)):
            if score > args.kpt_thr:
                x, y = int(kpt[0]), int(kpt[1])
                color = KPT_COLORS[i % len(KPT_COLORS)]
                cv2.circle(img_vis, (x, y), args.radius + 1, (255, 255, 255), -1)
                cv2.circle(img_vis, (x, y), args.radius, color, -1)
                if args.show_kpt_idx:
                    cv2.putText(img_vis, f'{i}', (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return img_vis


# ================================================================
# Tracker wrapper (simplified inline version)
# ================================================================
class SimpleTracker:
    """Lightweight tracker wrapper for inference."""

    def __init__(self, model, tracker_cfg, max_seq_len=30,
                 motion_thr=0.03, stable_frames=3, spike_window=15,
                 spike_thr=0.06):
        self.model = model
        self.tracker = MODELS.build(tracker_cfg)
        self.max_seq_len = max_seq_len
        self.kpt_buffers = {}    # track_id -> deque of (K*3,) features
        self.prev_kpts = {}      # track_id -> previous (K,2) normalized kpts
        self.stable_count = {}   # track_id -> consecutive stable frame count
        self.motion_hist = {}    # track_id -> deque of per-frame motion values
        self.motion_thr = motion_thr      # 静止判定阈值
        self.stable_frames = stable_frames  # 连续静止帧数
        self.spike_window = spike_window  # 回看窗口(帧)
        self.spike_thr = spike_thr        # 运动高峰阈值(摔倒动作)

    def reset(self):
        self.tracker.reset()
        self.kpt_buffers.clear()
        self.prev_kpts.clear()
        self.stable_count.clear()
        self.motion_hist.clear()

    @torch.no_grad()
    def process(self, img_path, pipeline, frame_id):
        """Process one frame, return (pred_instances, track_info_dict)."""
        result_tuple = inference_single(self.model, img_path, pipeline)
        if result_tuple is None:
            return None, None, None

        results, img = result_tuple
        if len(results) == 0:
            return img, None, None

        result = results[0]
        # Set frame_id for tracker
        result.set_metainfo({'frame_id': frame_id})

        # Run ByteTracker
        track_inst = self.tracker.track(result)
        device = track_inst.bboxes.device
        num_tracks = len(track_inst)

        if num_tracks == 0:
            return img, track_inst, {}

        # Match tracked bboxes to predicted instances for keypoints
        pred = result.pred_instances
        from mmdet.structures.bbox import bbox_overlaps
        if len(pred) > 0 and num_tracks > 0:
            ious = bbox_overlaps(track_inst.bboxes, pred.bboxes)
            # Copy keypoints/action_scores from best-matching predictions
            track_kpts = torch.zeros(num_tracks, 7, 2, device=device)
            track_kpt_scores = torch.zeros(num_tracks, 7, device=device)

            for t in range(num_tracks):
                best = ious[t].argmax()
                if ious[t, best] > 0.3:
                    if hasattr(pred, 'keypoints'):
                        track_kpts[t] = pred.keypoints[best]
                    if hasattr(pred, 'keypoint_scores'):
                        track_kpt_scores[t] = pred.keypoint_scores[best]

            track_inst.keypoints = track_kpts
            track_inst.keypoint_scores = track_kpt_scores

        # Buffer keypoints per track_id & run temporal GRU
        temporal_scores = {}
        if self.model.action_head is not None:
            for i in range(num_tracks):
                tid = int(track_inst.instances_id[i])
                if tid not in self.kpt_buffers:
                    self.kpt_buffers[tid] = collections.deque(maxlen=self.max_seq_len)

                # Normalize keypoints relative to bbox
                bbox = track_inst.bboxes[i]
                kpt = track_inst.keypoints[i] if hasattr(track_inst, 'keypoints') else torch.zeros(7, 2, device=device)
                vis = track_inst.keypoint_scores[i] if hasattr(track_inst, 'keypoint_scores') else torch.zeros(7, device=device)

                x1, y1, x2, y2 = bbox
                w = max(float(x2 - x1), 1.0)
                h = max(float(y2 - y1), 1.0)
                norm_x = ((kpt[:, 0] - x1) / w).clamp(0, 1)
                norm_y = ((kpt[:, 1] - y1) / h).clamp(0, 1)

                # Out-of-bounds → invisible
                oob = (norm_x <= 0) | (norm_x >= 1) | (norm_y <= 0) | (norm_y >= 1)
                vis_clean = vis.clone()
                vis_clean[oob] = 0.0
                norm_x[oob] = 0.0
                norm_y[oob] = 0.0

                feat = torch.stack([norm_x, norm_y, vis_clean], dim=-1).reshape(-1)
                self.kpt_buffers[tid].append(feat)

                # --- Motion check: lying still vs fell-then-still ---
                cur_xy = torch.stack([norm_x, norm_y], dim=-1)  # (K, 2)
                if tid not in self.motion_hist:
                    self.motion_hist[tid] = collections.deque(
                        maxlen=self.spike_window)

                if tid in self.prev_kpts:
                    motion = (cur_xy - self.prev_kpts[tid]).abs().mean().item()
                    if motion < self.motion_thr:
                        self.stable_count[tid] = self.stable_count.get(tid, 0) + 1
                    else:
                        self.stable_count[tid] = 0
                else:
                    motion = 0.0
                    self.stable_count[tid] = 0
                self.prev_kpts[tid] = cur_xy.clone()
                self.motion_hist[tid].append(motion)

                seq = torch.stack(list(self.kpt_buffers[tid])).unsqueeze(0).to(device)
                prob = self.model.action_head.predict(seq)
                score = float(prob.squeeze())
                raw_score = score

                # 抑制逻辑:
                #   当前静止 + GRU说跌倒 → 检查最近有没有运动高峰
                #   有高峰 = 真摔倒(倒下后躺着) → 保留分数
                #   无高峰 = 一直躺着/坐着 → 抑制分数
                if self.stable_count.get(tid, 0) >= self.stable_frames:
                    peak = max(self.motion_hist[tid])
                    if peak < self.spike_thr:
                        # 最近N帧从未有大运动 → 一直躺/坐 → 抑制
                        score = score * 0.1
                    # else: 有运动高峰 → 真摔倒后静止 → 不抑制

                temporal_scores[tid] = (score, raw_score)

        # Clean stale tracks
        active = set(track_inst.instances_id.tolist()) if num_tracks > 0 else set()
        for tid in list(self.kpt_buffers.keys()):
            if tid not in active:
                del self.kpt_buffers[tid]
                self.prev_kpts.pop(tid, None)
                self.stable_count.pop(tid, None)
                self.motion_hist.pop(tid, None)

        return img, track_inst, temporal_scores


# ================================================================
# Mode 1: 图片文件夹 (单帧, 无跟踪)
# ================================================================
def run_images(model, args):
    """处理图片文件夹"""
    pipeline = Compose(INFER_PIPELINE)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(args.input, f'*{ext.upper()}')))
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"No images found in {args.input}")
        return

    print(f"Found {len(image_files)} images")
    os.makedirs(args.output, exist_ok=True)

    # Optional: use tracker for sequential images
    tracker = None
    if args.use_tracker:
        cfg = Config.fromfile(args.config)
        tracker_cfg = cfg.get('tracker', None)
        if tracker_cfg is not None:
            tracker = SimpleTracker(model, tracker_cfg, args.max_seq_len,
                              motion_thr=args.motion_thr,
                              stable_frames=args.stable_frames,
                              spike_window=args.spike_window,
                              spike_thr=args.spike_thr)
            print("ByteTrack + temporal GRU enabled")

    success = 0
    fall_count = 0

    for i, img_path in enumerate(image_files):
        fname = os.path.basename(img_path)
        print(f"[{i+1}/{len(image_files)}] {fname}", end="  ")

        if tracker is not None:
            # Tracker mode
            img, track_inst, temporal_scores = tracker.process(
                img_path, pipeline, frame_id=i)
            if img is None or track_inst is None or len(track_inst) == 0:
                print("no detection")
                continue

            img_vis = img.copy()
            for t_idx in range(len(track_inst)):
                tid = int(track_inst.instances_id[t_idx])
                score_pair = temporal_scores.get(tid, (0.0, 0.0))
                t_score, t_raw = (score_pair if isinstance(score_pair, tuple)
                                  else (score_pair, score_pair))
                is_fall = t_score >= args.fall_thr

                # Create a single-instance pred for visualization
                from mmengine.structures import InstanceData
                single = InstanceData()
                single.bboxes = track_inst.bboxes[t_idx:t_idx+1]
                single.scores = track_inst.scores[t_idx:t_idx+1]
                if hasattr(track_inst, 'keypoints'):
                    single.keypoints = track_inst.keypoints[t_idx:t_idx+1]
                    single.keypoint_scores = track_inst.keypoint_scores[t_idx:t_idx+1]
                single.action_scores = torch.tensor([t_score])

                img_vis = visualize(img_vis, single, args,
                                    track_id=tid, temporal_score=t_score,
                                    raw_score=t_raw)
                if is_fall:
                    fall_count += 1

            n_tracks = len(track_inst)
            print(f"{n_tracks} tracks, "
                  f"scores={[f'{temporal_scores.get(int(track_inst.instances_id[j]),(0,0))[0]:.2f}' for j in range(n_tracks)]}")

        else:
            # Single-frame mode (no tracker)
            result_tuple = inference_single(model, img_path, pipeline)
            if result_tuple is None:
                print("read failed")
                continue
            results, img = result_tuple
            if not results:
                print("no detection")
                continue

            pred = results[0].pred_instances
            keep = pred.scores > args.det_thr
            pred = pred[keep]
            if len(pred) == 0:
                print("no valid detection")
                continue

            action_scores = pred.action_scores.cpu().numpy() if hasattr(pred, 'action_scores') else np.zeros(len(pred))
            n_fall = int((action_scores >= args.fall_thr).sum())
            fall_count += n_fall

            img_vis = visualize(img, pred, args)
            print(f"{len(pred)} dets, action={action_scores.round(2).tolist()}"
                  f"{' FALL!' if n_fall > 0 else ''}")

        # Save
        output_path = os.path.join(args.output, fname)
        cv2.imwrite(output_path, img_vis)
        success += 1

    print(f"\nDone! {success}/{len(image_files)} images processed")
    print(f"Falling detected in {fall_count} instances")
    print(f"Results saved to: {args.output}")


# ================================================================
# Mode 2: 视频文件
# ================================================================
def run_video(model, args):
    """处理视频文件"""
    pipeline = Compose(INFER_PIPELINE)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Cannot open video: {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

    os.makedirs(args.output, exist_ok=True)

    # Setup tracker
    cfg = Config.fromfile(args.config)
    tracker_cfg = cfg.get('tracker', None)
    tracker = None
    if tracker_cfg is not None:
        tracker = SimpleTracker(model, tracker_cfg, args.max_seq_len,
                              motion_thr=args.motion_thr,
                              stable_frames=args.stable_frames,
                              spike_window=args.spike_window,
                              spike_thr=args.spike_thr)
        print("ByteTrack + temporal GRU enabled")

    # Video writer
    writer = None
    if args.save_video:
        out_path = os.path.join(args.output, 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"Saving video to: {out_path}")

    frame_id = 0
    fall_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame temporarily for pipeline
        tmp_path = os.path.join(args.output, '_tmp_frame.png')
        cv2.imwrite(tmp_path, frame)

        if tracker is not None:
            img, track_inst, temporal_scores = tracker.process(
                tmp_path, pipeline, frame_id=frame_id)

            if img is not None and track_inst is not None and len(track_inst) > 0:
                img_vis = frame.copy()
                has_fall = False
                for t_idx in range(len(track_inst)):
                    tid = int(track_inst.instances_id[t_idx])
                    score_pair = temporal_scores.get(tid, (0.0, 0.0))
                    t_score, t_raw = (score_pair if isinstance(score_pair, tuple)
                                      else (score_pair, score_pair))
                    if t_score >= args.fall_thr:
                        has_fall = True

                    from mmengine.structures import InstanceData
                    single = InstanceData()
                    single.bboxes = track_inst.bboxes[t_idx:t_idx+1]
                    single.scores = track_inst.scores[t_idx:t_idx+1]
                    if hasattr(track_inst, 'keypoints'):
                        single.keypoints = track_inst.keypoints[t_idx:t_idx+1]
                        single.keypoint_scores = track_inst.keypoint_scores[t_idx:t_idx+1]
                    single.action_scores = torch.tensor([t_score])
                    img_vis = visualize(img_vis, single, args,
                                        track_id=tid, temporal_score=t_score,
                                        raw_score=t_raw)

                if has_fall:
                    fall_frames += 1
            else:
                img_vis = frame.copy()
        else:
            result_tuple = inference_single(model, tmp_path, pipeline)
            if result_tuple is not None and len(result_tuple[0]) > 0:
                _, img = result_tuple
                pred = result_tuple[0][0].pred_instances
                keep = pred.scores > args.det_thr
                pred = pred[keep]
                img_vis = visualize(frame, pred, args) if len(pred) > 0 else frame.copy()
            else:
                img_vis = frame.copy()

        # Frame info overlay
        info = f'Frame {frame_id}'
        if tracker:
            n = len(track_inst) if track_inst is not None else 0
            info += f' | Tracks: {n}'
        cv2.putText(img_vis, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer:
            writer.write(img_vis)

        # Save key frames with falling
        if frame_id % 30 == 0:
            cv2.imwrite(os.path.join(args.output, f'frame_{frame_id:06d}.jpg'), img_vis)

        if frame_id % 50 == 0:
            print(f"  Frame {frame_id}/{total_frames}")

        frame_id += 1

    cap.release()
    if writer:
        writer.release()

    # Cleanup
    tmp_path = os.path.join(args.output, '_tmp_frame.png')
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"\nDone! {frame_id} frames processed")
    print(f"Falling detected in {fall_frames}/{frame_id} frames")


def main():
    args = parse_args()

    print("=" * 70)
    print("RTMDet-Pose V7: Detection + Pose + Falling + Tracking")
    print("=" * 70)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Det thr:    {args.det_thr}")
    print(f"Kpt thr:    {args.kpt_thr}")
    print(f"Fall thr:   {args.fall_thr}")
    print(f"Tracker:    {'ON' if args.use_tracker else 'OFF'}")

    if torch.cuda.is_available():
        print(f"Device:     GPU ({torch.cuda.get_device_name(0)})")
        device = 'cuda:0'
    else:
        print("Device:     CPU")
        device = 'cpu'
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = init_detector(args.config, args.checkpoint, device=device)
    model.eval()
    has_action = hasattr(model, 'action_head') and model.action_head is not None
    print(f"Model loaded! action_head={'YES' if has_action else 'NO'}")

    # Decide mode: video or images
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    is_video = os.path.isfile(args.input) and \
               os.path.splitext(args.input)[1].lower() in video_exts

    if is_video:
        print("\nMode: VIDEO")
        run_video(model, args)
    else:
        print("\nMode: IMAGES")
        run_images(model, args)


if __name__ == '__main__':
    main()

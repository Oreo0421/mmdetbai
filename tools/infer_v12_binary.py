#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTMDet-Pose V12 Binary 推理脚本: 二分类跌倒检测 + 规则后处理

使用关键点特征(kpt_dim=5: x, y, vis, dx, dy) + GRU时序分类
action_head.predict返回 (N,) 概率值 (binary模式)

规则后处理 (仅推理时):
  规则1: 头部关键点y坐标快速下降 → 确认跌倒 (提高概率)
  规则2: 头部持续低位+无运动 → 躺着/已倒地 (抑制概率，减少误检)

遍历 infer/{action}/{camera}/undistorted_stitched_png/*.png
输出到 output/{action}/{camera}/*.png

Usage:
  python tools/infer_v12_binary.py \
      --input /home/tbai/Desktop/infer \
      --output /home/tbai/Desktop/fall_infer/v12
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import collections
import torch
import argparse

from mmdet.apis import init_detector
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.structures import InstanceData


# Skeleton connections for visualization
SKELETON = [
    (0, 1),  # head -> shoulder
    (1, 2),  # shoulder -> hand_right
    (1, 3),  # shoulder -> hand_left
    (1, 4),  # shoulder -> hips
    (4, 5),  # hips -> foot_right
    (4, 6),  # hips -> foot_left
]

# ---- 推理 pipeline ----
INFER_PIPELINE = [
    dict(type='LoadImageFromFile', _scope_='mmdet'),
    dict(type='Resize', scale=(384, 384), keep_ratio=False, _scope_='mmdet'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
         _scope_='mmdet'),
]

# ---- 颜色 ----
KPT_COLORS = [
    (0, 0, 255),    # 0: head
    (0, 128, 255),  # 1: shoulder
    (0, 255, 255),  # 2: hand_R
    (0, 255, 0),    # 3: hand_L
    (255, 255, 0),  # 4: hips
    (255, 0, 0),    # 5: foot_R
    (255, 0, 255),  # 6: foot_L
]
LIMB_COLOR = (0, 255, 128)
COLOR_NORMAL = (0, 255, 0)
COLOR_FALLING = (0, 0, 255)
COLOR_TRACK = [
    (255, 128, 0), (0, 255, 128), (128, 0, 255),
    (255, 255, 0), (0, 128, 255), (255, 0, 128),
    (128, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 128, 255),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='RTMDet-Pose V12: Binary Fall Detection + Rules')
    parser.add_argument('--config',
                        default='/home/tbai/mmdetection/mmdetection/configs/rtmdet_bai/rtmdet_pose_v12_binary.py',
                        help='config path')
    parser.add_argument('--checkpoint',
                        default='work_dirs/rtmdet_pose_v12_binary/epoch_50.pth',
                        help='checkpoint path')
    parser.add_argument('--input',
                        default='/home/tbai/Desktop/infer',
                        help='input root dir')
    parser.add_argument('--output',
                        default='/home/tbai/Desktop/fall_infer/v12',
                        help='output root dir')
    parser.add_argument('--det-thr', type=float, default=0.3)
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument('--fall-thr', type=float, default=0.5)
    parser.add_argument('--max-seq-len', type=int, default=30)
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--thickness', type=int, default=1)
    # Rule post-processing thresholds
    parser.add_argument('--rule-head-drop-thr', type=float, default=0.15,
                        help='Head y-drop over 3 frames to confirm fall')
    parser.add_argument('--rule-head-low-thr', type=float, default=0.7,
                        help='Head y threshold for "low position" (relative to bbox)')
    parser.add_argument('--rule-static-frames', type=int, default=5,
                        help='Frames of no motion to consider "lying still"')
    args = parser.parse_args()
    return args


# ================================================================
# Single image inference
# ================================================================
def inference_single(model, img_path, pipeline):
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


# ================================================================
# Keypoint feature computation (kpt_dim=5: x, y, vis, dx, dy)
# ================================================================
def build_kpt_feature(norm_x, norm_y, vis, prev_x=None, prev_y=None):
    """Build per-keypoint feature: [x, y, vis, dx, dy] x 7 = 35 dim."""
    K = len(norm_x)
    feat = np.zeros(K * 5, dtype=np.float32)
    for i in range(K):
        dx = 0.0
        dy = 0.0
        if prev_x is not None and prev_y is not None:
            dx = float(norm_x[i] - prev_x[i])
            dy = float(norm_y[i] - prev_y[i])
        feat[i * 5 + 0] = float(norm_x[i])
        feat[i * 5 + 1] = float(norm_y[i])
        feat[i * 5 + 2] = float(vis[i])
        feat[i * 5 + 3] = dx
        feat[i * 5 + 4] = dy
    return feat


# ================================================================
# Rule-based post-processing
# ================================================================
class RulePostProcessor:
    """Rule-based post-processing for fall detection.

    Rule 1: Head y-coordinate rapid drop → confirm fall (boost probability)
    Rule 2: Head sustained low position + no motion → lying/fallen (suppress probability)
    """

    def __init__(self, head_drop_thr=0.15, head_low_thr=0.7, static_frames=5):
        self.head_drop_thr = head_drop_thr
        self.head_low_thr = head_low_thr
        self.static_frames = static_frames
        # Per-track history: list of (head_norm_y, motion_magnitude)
        self.history = {}

    def reset(self):
        self.history.clear()

    def update(self, tid, head_norm_y, motion_mag):
        if tid not in self.history:
            self.history[tid] = collections.deque(maxlen=30)
        self.history[tid].append((head_norm_y, motion_mag))

    def adjust_prob(self, tid, raw_prob):
        """Adjust fall probability based on rules.

        Returns adjusted probability.
        """
        if tid not in self.history or len(self.history[tid]) < 2:
            return raw_prob

        hist = list(self.history[tid])

        # Rule 1: Head rapid drop → boost probability
        # Check last 3 frames for rapid head y increase (falling down = y increases)
        if len(hist) >= 3:
            y_now = hist[-1][0]
            y_3ago = hist[-3][0]
            drop = y_now - y_3ago  # positive = head moved down
            if drop > self.head_drop_thr:
                # Significant head drop → boost fall probability
                boost = min(drop / self.head_drop_thr * 0.3, 0.4)
                raw_prob = min(raw_prob + boost, 1.0)

        # Rule 2: Head sustained low + no motion → suppress probability
        # (person already lying down, not actively falling)
        if len(hist) >= self.static_frames:
            recent = hist[-self.static_frames:]
            all_low = all(h[0] > self.head_low_thr for h in recent)
            all_static = all(h[1] < 0.02 for h in recent)
            if all_low and all_static:
                # Person is lying still → suppress fall alarm
                raw_prob = raw_prob * 0.5

        return raw_prob

    def cleanup(self, active_ids):
        for tid in list(self.history.keys()):
            if tid not in active_ids:
                del self.history[tid]


# ================================================================
# Visualization
# ================================================================
def visualize(img, pred, args, track_id=None, fall_prob=None, is_falling=False):
    img_vis = img.copy()

    bboxes = pred.bboxes.cpu().numpy()
    bbox_scores = pred.scores.cpu().numpy()
    keypoints = pred.keypoints.cpu().numpy() if hasattr(pred, 'keypoints') else np.zeros((len(bboxes), 0, 2))
    kpt_scores = pred.keypoint_scores.cpu().numpy() if hasattr(pred, 'keypoint_scores') else np.zeros((len(bboxes), 0))

    for idx in range(len(bboxes)):
        det_score = bbox_scores[idx]

        if is_falling:
            box_color = COLOR_FALLING
        elif track_id is not None:
            box_color = COLOR_TRACK[track_id % len(COLOR_TRACK)]
        else:
            box_color = COLOR_NORMAL

        x1, y1, x2, y2 = map(int, bboxes[idx][:4])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), box_color, args.thickness)

        parts = []
        if track_id is not None:
            parts.append(f'ID:{track_id}')
        parts.append(f'{det_score:.2f}')
        if fall_prob is not None:
            parts.append(f'fall:{fall_prob:.2f}')
        if is_falling:
            parts.append('FALL!')
        label = ' '.join(parts)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img_vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), box_color, -1)
        cv2.putText(img_vis, label, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if idx >= len(keypoints):
            continue
        kpts = keypoints[idx]
        scrs = kpt_scores[idx]

        for (a, b) in SKELETON:
            if a < len(kpts) and b < len(kpts):
                if scrs[a] > args.kpt_thr and scrs[b] > args.kpt_thr:
                    pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                    pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                    cv2.line(img_vis, pt1, pt2, LIMB_COLOR, args.thickness)

        for i, (kpt, score) in enumerate(zip(kpts, scrs)):
            if score > args.kpt_thr:
                x, y = int(kpt[0]), int(kpt[1])
                color = KPT_COLORS[i % len(KPT_COLORS)]
                cv2.circle(img_vis, (x, y), args.radius, color, -1)

    return img_vis


# ================================================================
# Keypoint Tracker with Rule Post-Processing
# ================================================================
class KptTracker:
    """V12 tracker: keypoint features (kpt_dim=5) + binary fall + rule post-processing."""

    def __init__(self, model, tracker_cfg, max_seq_len=30, rule_processor=None):
        self.model = model
        self.tracker = MODELS.build(tracker_cfg)
        self.max_seq_len = max_seq_len
        self.rule_processor = rule_processor
        # Per-track buffers
        self.feat_buffers = {}
        self.prev_norm = {}  # {tid: (norm_x, norm_y)}

    def reset(self):
        self.tracker.reset()
        self.feat_buffers.clear()
        self.prev_norm.clear()
        if self.rule_processor:
            self.rule_processor.reset()

    @torch.no_grad()
    def process(self, img_path, pipeline, frame_id, device):
        result_tuple = inference_single(self.model, img_path, pipeline)
        if result_tuple is None:
            return None, None, {}
        results, img = result_tuple
        if len(results) == 0:
            return img, None, {}

        result = results[0]
        result.set_metainfo({'frame_id': frame_id})

        # ByteTracker
        track_inst = self.tracker.track(result)
        num_tracks = len(track_inst)
        if num_tracks == 0:
            return img, track_inst, {}

        # Match tracked → predicted for keypoints
        pred = result.pred_instances
        from mmdet.structures.bbox import bbox_overlaps

        track_kpts = torch.zeros(num_tracks, 7, 2, device=device)
        track_kpt_scores = torch.zeros(num_tracks, 7, device=device)

        if len(pred) > 0:
            ious = bbox_overlaps(track_inst.bboxes, pred.bboxes)
            for t in range(num_tracks):
                best = ious[t].argmax()
                if ious[t, best] > 0.3:
                    if hasattr(pred, 'keypoints'):
                        track_kpts[t] = pred.keypoints[best]
                    if hasattr(pred, 'keypoint_scores'):
                        track_kpt_scores[t] = pred.keypoint_scores[best]

        track_inst.keypoints = track_kpts
        track_inst.keypoint_scores = track_kpt_scores

        # Action classification with keypoint temporal features
        action_info = {}
        if self.model.action_head is not None:
            for i in range(num_tracks):
                tid = int(track_inst.instances_id[i])
                bbox = track_inst.bboxes[i]
                kpt = track_kpts[i]
                vis = track_kpt_scores[i]

                x1, y1, x2, y2 = bbox
                w = max(float(x2 - x1), 1.0)
                h = max(float(y2 - y1), 1.0)
                norm_x = ((kpt[:, 0] - x1) / w).clamp(0, 1).cpu().numpy()
                norm_y = ((kpt[:, 1] - y1) / h).clamp(0, 1).cpu().numpy()
                vis_np = vis.cpu().numpy()

                # Build kpt feature (kpt_dim=5)
                prev = self.prev_norm.get(tid, None)
                prev_x = prev[0] if prev is not None else None
                prev_y = prev[1] if prev is not None else None
                kpt_feat = build_kpt_feature(
                    norm_x, norm_y, vis_np, prev_x, prev_y)  # (35,)
                self.prev_norm[tid] = (norm_x.copy(), norm_y.copy())

                # Accumulate in buffer
                if tid not in self.feat_buffers:
                    self.feat_buffers[tid] = collections.deque(
                        maxlen=self.max_seq_len)
                self.feat_buffers[tid].append(kpt_feat)

                # GRU temporal prediction (binary: returns (N,) probability)
                seq = np.stack(list(self.feat_buffers[tid]))  # (T, 35)
                seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
                raw_prob = float(
                    self.model.action_head.predict(seq_tensor).cpu().item())

                # Rule-based post-processing
                if self.rule_processor:
                    # head_norm_y: head keypoint[0] normalized y position
                    head_y = float(norm_y[0])
                    # motion_magnitude: average keypoint movement
                    if prev_x is not None:
                        motion = float(np.mean(
                            np.abs(norm_x - prev_x) + np.abs(norm_y - prev_y)))
                    else:
                        motion = 0.0
                    self.rule_processor.update(tid, head_y, motion)
                    fall_prob = self.rule_processor.adjust_prob(tid, raw_prob)
                else:
                    fall_prob = raw_prob

                action_info[tid] = {
                    'fall_prob': fall_prob,
                    'raw_prob': raw_prob,
                }

        # Cleanup stale tracks
        active = set(track_inst.instances_id.tolist()) if num_tracks > 0 else set()
        for tid in list(self.feat_buffers.keys()):
            if tid not in active:
                del self.feat_buffers[tid]
                self.prev_norm.pop(tid, None)
        if self.rule_processor:
            self.rule_processor.cleanup(active)

        return img, track_inst, action_info


# ================================================================
# Process one sequence (one camera folder)
# ================================================================
def process_sequence(model, tracker, pipeline, image_files, output_dir,
                     args, device):
    os.makedirs(output_dir, exist_ok=True)
    tracker.reset()

    fall_count = 0

    for i, img_path in enumerate(image_files):
        fname = os.path.basename(img_path)

        img, track_inst, action_info = tracker.process(
            img_path, pipeline, frame_id=i, device=device)

        if img is None:
            continue

        if track_inst is None or len(track_inst) == 0:
            cv2.imwrite(os.path.join(output_dir, fname), img)
            continue

        img_vis = img.copy()
        has_fall = False

        for t_idx in range(len(track_inst)):
            tid = int(track_inst.instances_id[t_idx])
            info = action_info.get(tid, {})
            fall_prob = info.get('fall_prob', 0.0)
            is_falling = fall_prob >= args.fall_thr

            single = InstanceData()
            single.bboxes = track_inst.bboxes[t_idx:t_idx+1]
            single.scores = track_inst.scores[t_idx:t_idx+1]
            if hasattr(track_inst, 'keypoints'):
                single.keypoints = track_inst.keypoints[t_idx:t_idx+1]
                single.keypoint_scores = track_inst.keypoint_scores[t_idx:t_idx+1]

            img_vis = visualize(img_vis, single, args,
                                track_id=tid,
                                fall_prob=fall_prob,
                                is_falling=is_falling)
            if is_falling:
                has_fall = True

        if has_fall:
            fall_count += 1

        cv2.imwrite(os.path.join(output_dir, fname), img_vis)

    return len(image_files), fall_count


# ================================================================
# Main
# ================================================================
def main():
    args = parse_args()

    print("=" * 70)
    print("RTMDet-Pose V12: Binary Fall Detection + Rule Post-Processing")
    print("=" * 70)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Radius:     {args.radius}")
    print(f"Fall thr:   {args.fall_thr}")
    print(f"Rules: head_drop={args.rule_head_drop_thr}, "
          f"head_low={args.rule_head_low_thr}, "
          f"static_frames={args.rule_static_frames}")
    print("=" * 70)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nLoading model... (device: {device})")
    model = init_detector(args.config, args.checkpoint, device=device)
    model.eval()
    has_action = hasattr(model, 'action_head') and model.action_head is not None
    num_cls = model.action_head.num_classes if has_action else 'N/A'
    print(f"Model loaded! action_head={'YES' if has_action else 'NO'}, "
          f"num_classes={num_cls}")

    cfg = Config.fromfile(args.config)
    tracker_cfg = cfg.get('tracker', None)
    if tracker_cfg is None:
        raise RuntimeError('No tracker config found in config file')

    rule_processor = RulePostProcessor(
        head_drop_thr=args.rule_head_drop_thr,
        head_low_thr=args.rule_head_low_thr,
        static_frames=args.rule_static_frames,
    )

    tracker = KptTracker(
        model, tracker_cfg,
        max_seq_len=args.max_seq_len,
        rule_processor=rule_processor)
    pipeline = Compose(INFER_PIPELINE)

    # Walk the nested directory
    input_root = args.input
    output_root = args.output
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    total_images = 0
    total_falls = 0
    total_sequences = 0

    action_dirs = sorted([
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])

    for action_dir in action_dirs:
        action_path = os.path.join(input_root, action_dir)
        camera_dirs = sorted([
            d for d in os.listdir(action_path)
            if os.path.isdir(os.path.join(action_path, d))
        ])

        for camera_dir in camera_dirs:
            img_dir = os.path.join(action_path, camera_dir, 'undistorted_stitched_png')
            if not os.path.isdir(img_dir):
                continue

            image_files = sorted([
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower() in image_exts
            ])
            if not image_files:
                continue

            out_dir = os.path.join(output_root, action_dir, camera_dir)
            print(f"\n[{action_dir}/{camera_dir}] {len(image_files)} frames -> {out_dir}")

            n_imgs, n_falls = process_sequence(
                model, tracker, pipeline, image_files, out_dir, args, device)

            total_images += n_imgs
            total_falls += n_falls
            total_sequences += 1

            print(f"  Done: {n_imgs} frames, {n_falls} falling frames")

    print("\n" + "=" * 70)
    print(f"All done! {total_sequences} sequences, {total_images} images")
    print(f"Falling detected in {total_falls} frames")
    print(f"Output: {output_root}")


if __name__ == '__main__':
    main()

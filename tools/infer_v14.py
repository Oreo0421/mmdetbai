#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTMDet-Pose V14 推理脚本: 4-class action + binary fall detection + rule post-processing

4-class action: standing(0), walking(1), sitting(2), lying(3)
Binary fall: from dedicated fall head (BCE) + rule post-processing (same as V12)

Uses keypoint features (kpt_dim=5: x, y, vis, dx, dy) + GRU temporal classification.
action_head.predict() returns dict: action_probs(N,4), action_class(N,), fall_prob(N,), is_falling(N,)

Visualization shows: ID:X det_score [standing/walking/sitting/lying] fall:0.XX FALL!

Usage:
  python tools/infer_v14.py \
      --input /home/tbai/Desktop/infer \
      --output /home/tbai/Desktop/fall_infer/v14
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import json
import numpy as np
import collections
import torch
import argparse

from mmdet.apis import init_detector
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.structures import InstanceData


# 4-class action names
ACTION4_NAMES = ['standing', 'walking', 'sitting', 'lying']

# Skeleton connections for visualization
SKELETON = [
    (0, 1),  # head -> shoulder
    (1, 2),  # shoulder -> hand_right
    (1, 3),  # shoulder -> hand_left
    (1, 4),  # shoulder -> hips
    (4, 5),  # hips -> foot_right
    (4, 6),  # hips -> foot_left
]

# ---- Inference pipeline ----
INFER_PIPELINE = [
    dict(type='LoadImageFromFile', _scope_='mmdet'),
    dict(type='Resize', scale=(384, 384), keep_ratio=False, _scope_='mmdet'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
         _scope_='mmdet'),
]

# ---- Colors ----
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
        description='RTMDet-Pose V14: 4-Class Action + Binary Fall Detection')
    parser.add_argument('--config',
                        default='/home/tbai/mmdetection/mmdetection/configs/rtmdet_bai/rtmdet_pose_v14_action4.py',
                        help='config path')
    parser.add_argument('--checkpoint',
                        default='work_dirs/rtmdet_pose_v14b_tuned/best_action_fall_f1_epoch_11.pth',
                        help='checkpoint path')
    parser.add_argument('--input',
                        default='/home/tbai/Desktop/infer',
                        help='input root dir')
    parser.add_argument('--output',
                        default='/home/tbai/Desktop/fall_infer/v14',
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
    parser.add_argument('--rule-sit-frames', type=int, default=5,
                        help='Frames to check for sit-down pattern')
    parser.add_argument('--rule-sit-max-head-speed', type=float, default=0.06,
                        help='Max per-frame head y change for "slow descent"')
    parser.add_argument('--rule-sit-max-foot-motion', type=float, default=0.03,
                        help='Max foot motion for "feet stable"')
    # V14 action-based rule params
    parser.add_argument('--rule-lying-suppress-frames', type=int, default=5,
                        help='Consecutive lying frames to trigger suppression')
    parser.add_argument('--rule-getup-frames', type=int, default=3,
                        help='Frames to detect head rising (getting up)')
    parser.add_argument('--rule-transition-window', type=int, default=8,
                        help='Window to check for standing/walking->lying transition')
    # V14c: consecutive frame confirmation + velocity gate
    parser.add_argument('--min-fall-frames', type=int, default=1,
                        help='Consecutive frames above fall-thr to confirm fall (1=off)')
    parser.add_argument('--rule-velocity-suppress-thr', type=float, default=0.02,
                        help='If avg body motion < this for recent frames, suppress fall_prob')
    # V15: Direction-based rules
    parser.add_argument('--rule-head-vel-suppress', type=float, default=0.04,
                        help='Head downward velocity below this = slow descent, suppress')
    parser.add_argument('--rule-head-vel-boost', type=float, default=0.08,
                        help='Head downward velocity above this = fast fall, boost')
    parser.add_argument('--rule-head-rise-thr', type=float, default=0.03,
                        help='Head upward velocity above this = getting up, suppress')
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
# V15: Rule-based action state estimation from keypoint features
# ================================================================
# 7 states: standing, walking, sitting, lying, falling, getting_up, sitting_down
RULE_ACTION_NAMES = ['standing', 'walking', 'sitting', 'lying', 'falling', 'getting_up', 'sitting_down']


def classify_action_by_rules(hist, fall_prob, fall_thr=0.5):
    """Classify action state from keypoint history + fall_prob.

    Uses head position, head velocity, body motion, foot motion to infer
    the current action state. This replaces the V14 4-class action head
    with pure rule-based classification for V15/V12 binary models.

    Args:
        hist: list of (head_norm_y, motion_mag, foot_avg_y, foot_motion)
            from RulePostProcessor history buffer.
        fall_prob: post-processed fall probability.
        fall_thr: threshold for falling state.

    Returns:
        action_idx: int index into RULE_ACTION_NAMES
        action_name: str name
    """
    if len(hist) < 2:
        return 0, 'standing'

    head_y = hist[-1][0]       # current head y (0=top, 1=bottom of bbox)
    motion = hist[-1][1]       # current body motion magnitude
    foot_y = hist[-1][2]       # current foot avg y

    # Head velocity: positive = moving DOWN, negative = moving UP
    if len(hist) >= 3:
        head_vel = (hist[-1][0] - hist[-3][0]) / 2.0
    else:
        head_vel = hist[-1][0] - hist[-2][0]

    # Average body motion over recent frames
    recent_n = min(len(hist), 5)
    avg_motion = np.mean([h[1] for h in hist[-recent_n:]])
    avg_foot_motion = np.mean([h[3] for h in hist[-recent_n:]])

    # ---- Classification rules (priority order) ----

    # 1. Falling: model says falling
    if fall_prob >= fall_thr:
        return 4, 'falling'

    # 2. Getting up: head moving UP significantly
    if head_vel < -0.03 and avg_motion > 0.02:
        return 5, 'getting_up'

    # 3. Sitting down: head slowly descending + feet stable
    if 0.01 < head_vel < 0.06 and avg_foot_motion < 0.03:
        return 6, 'sitting_down'

    # 4. Lying: head very low + barely moving
    if head_y > 0.7 and avg_motion < 0.03:
        return 3, 'lying'

    # 5. Sitting: head in middle zone + low motion + feet stable
    if 0.4 < head_y < 0.75 and avg_motion < 0.04 and avg_foot_motion < 0.02:
        return 2, 'sitting'

    # 6. Walking: significant foot motion + moderate body motion
    if avg_foot_motion > 0.02 and avg_motion > 0.02:
        return 1, 'walking'

    # 7. Default: standing
    return 0, 'standing'


# ================================================================
# Rule-based post-processing (same as V12)
# ================================================================
class RulePostProcessor:
    """Rule-based post-processing for fall detection (V14 enhanced).

    Rule 1: Head y-coordinate rapid drop -> confirm fall (boost probability)
    Rule 2: Head sustained low position + no motion -> lying still (suppress)
    Rule 3: Sit-down suppression -> head drops slowly + feet stable (suppress)
    Rule 5 (V14): Action-gated: consecutive "lying" without transition from standing/walking -> suppress
    Rule 6 (V14): Getting-up detection: head rising over recent frames -> suppress
    Rule 7 (V14): Transition-based: only allow high fall_prob near a standing/walking->lying transition
    """

    def __init__(self, head_drop_thr=0.15, head_low_thr=0.7, static_frames=5,
                 sit_frames=5, sit_max_head_speed=0.06, sit_max_foot_motion=0.03,
                 lying_suppress_frames=5, getup_frames=3, transition_window=8,
                 min_fall_frames=1, velocity_suppress_thr=0.02,
                 head_vel_suppress=0.04, head_vel_boost=0.08,
                 head_rise_thr=0.03):
        self.head_drop_thr = head_drop_thr
        self.head_low_thr = head_low_thr
        self.static_frames = static_frames
        self.sit_frames = sit_frames
        self.sit_max_head_speed = sit_max_head_speed
        self.sit_max_foot_motion = sit_max_foot_motion
        # V14 new params
        self.lying_suppress_frames = lying_suppress_frames
        self.getup_frames = getup_frames
        self.transition_window = transition_window
        # V14c: consecutive frame confirmation + velocity gate
        self.min_fall_frames = min_fall_frames
        self.velocity_suppress_thr = velocity_suppress_thr
        # V15: direction-based rule thresholds
        self.head_vel_suppress = head_vel_suppress
        self.head_vel_boost = head_vel_boost
        self.head_rise_thr = head_rise_thr
        self.history = {}
        self.action_history = {}  # tid -> deque of action_class
        self.consec_fall = {}     # tid -> consecutive fall frame count

    def reset(self):
        self.history.clear()
        self.action_history.clear()
        self.consec_fall.clear()

    def update(self, tid, head_norm_y, motion_mag, foot_avg_y=None,
               foot_motion=None, action_class=None):
        if tid not in self.history:
            self.history[tid] = collections.deque(maxlen=30)
        self.history[tid].append((head_norm_y, motion_mag,
                                  foot_avg_y if foot_avg_y is not None else 0.0,
                                  foot_motion if foot_motion is not None else 0.0))
        if action_class is not None:
            if tid not in self.action_history:
                self.action_history[tid] = collections.deque(maxlen=30)
            self.action_history[tid].append(action_class)

    def adjust_prob(self, tid, raw_prob):
        """Adjust fall probability based on rules."""
        if tid not in self.history or len(self.history[tid]) < 2:
            return raw_prob

        hist = list(self.history[tid])
        act_hist = list(self.action_history.get(tid, []))

        # Rule 1: Head rapid drop -> boost probability
        if len(hist) >= 3:
            y_now = hist[-1][0]
            y_3ago = hist[-3][0]
            drop = y_now - y_3ago
            if drop > self.head_drop_thr:
                foot_motion_avg = np.mean([h[3] for h in hist[-3:]])
                if foot_motion_avg > self.sit_max_foot_motion:
                    boost = min(drop / self.head_drop_thr * 0.3, 0.4)
                    raw_prob = min(raw_prob + boost, 1.0)

        # Rule 2: Head sustained low + no motion -> lying still (suppress)
        if len(hist) >= self.static_frames:
            recent = hist[-self.static_frames:]
            all_low = all(h[0] > self.head_low_thr for h in recent)
            all_static = all(h[1] < 0.02 for h in recent)
            if all_low and all_static:
                raw_prob = raw_prob * 0.3

        # Rule 3: Sit-down suppression
        if len(hist) >= self.sit_frames:
            recent = hist[-self.sit_frames:]
            head_ys = [h[0] for h in recent]
            total_head_drop = head_ys[-1] - head_ys[0]
            if total_head_drop > 0.05:
                per_frame_drops = [head_ys[j+1] - head_ys[j]
                                   for j in range(len(head_ys)-1)]
                max_per_frame = max(per_frame_drops)
                foot_motions = [h[3] for h in recent]
                avg_foot_motion = np.mean(foot_motions)

                if (max_per_frame < self.sit_max_head_speed and
                        avg_foot_motion < self.sit_max_foot_motion):
                    raw_prob = raw_prob * 0.3

        # ---- V14 action-based rules ----

        # Rule 5: Action-gated lying suppression
        # Only suppress if lying for many frames AND no recent upright history
        # (i.e. person was already lying before tracking started, not a fall)
        if len(act_hist) >= self.lying_suppress_frames:
            recent_acts = act_hist[-self.lying_suppress_frames:]
            all_lying = all(a == 3 for a in recent_acts)
            if all_lying:
                window = act_hist[-min(len(act_hist), self.transition_window):]
                had_upright = any(a in (0, 1) for a in window)
                if not had_upright:
                    # Pure lying from start, no transition -> suppress
                    raw_prob = raw_prob * 0.3

        # Rule 6: Getting-up detection -> head is rising (y decreasing)
        if len(hist) >= self.getup_frames:
            recent_ys = [h[0] for h in hist[-self.getup_frames:]]
            head_rise = recent_ys[0] - recent_ys[-1]  # positive = rising
            if head_rise > 0.1:
                raw_prob = raw_prob * 0.3

        # Rule 7: Sitting steady state suppression
        if len(act_hist) >= self.lying_suppress_frames:
            recent_acts = act_hist[-self.lying_suppress_frames:]
            all_sitting = all(a == 2 for a in recent_acts)
            if all_sitting:
                raw_prob = raw_prob * 0.4

        # Rule 8 (V14c): Velocity gate — very low body motion -> suppress
        # If person is barely moving, it's likely lying still or sitting, not falling
        if len(hist) >= 3:
            recent_motions = [h[1] for h in hist[-3:]]
            avg_motion = np.mean(recent_motions)
            if avg_motion < self.velocity_suppress_thr:
                # No head drop either -> strong suppress
                y_now = hist[-1][0]
                y_prev = hist[-3][0] if len(hist) >= 3 else y_now
                head_drop = y_now - y_prev
                if head_drop < 0.05:  # head not dropping fast
                    raw_prob = raw_prob * 0.2

        # Rule 10 (V15): Direction + speed gate
        # Head vertical velocity: positive = moving DOWN, negative = moving UP
        if len(hist) >= 3:
            head_vel = (hist[-1][0] - hist[-3][0]) / 2.0  # smoothed over 3 frames

            # Head acceleration (change in velocity)
            if len(hist) >= 5:
                vel_old = (hist[-3][0] - hist[-5][0]) / 2.0
                head_accel = head_vel - vel_old
            else:
                head_accel = 0.0

            # (a) Head moving UP -> not falling, suppress
            if head_vel < -self.head_rise_thr:
                raw_prob *= 0.15

            # (b) Head moving down SLOWLY -> lying/sitting, suppress
            elif 0 < head_vel < self.head_vel_suppress:
                raw_prob *= 0.4

            # (c) Head moving down FAST + acceleration -> likely fall, boost
            elif head_vel > self.head_vel_boost and head_accel > 0.02:
                raw_prob = min(raw_prob + 0.3, 1.0)

        # Rule 9 (V14c): Consecutive frame confirmation
        # Only declare fall if fall_prob >= threshold for min_fall_frames in a row
        if self.min_fall_frames > 1:
            if raw_prob >= 0.5:  # use a fixed internal threshold for counting
                self.consec_fall[tid] = self.consec_fall.get(tid, 0) + 1
            else:
                self.consec_fall[tid] = 0
            if self.consec_fall.get(tid, 0) < self.min_fall_frames:
                # Not enough consecutive frames yet — suppress to below threshold
                raw_prob = min(raw_prob, 0.3)

        return raw_prob

    def cleanup(self, active_ids):
        for tid in list(self.history.keys()):
            if tid not in active_ids:
                del self.history[tid]
        for tid in list(self.action_history.keys()):
            if tid not in active_ids:
                del self.action_history[tid]
        for tid in list(self.consec_fall.keys()):
            if tid not in active_ids:
                del self.consec_fall[tid]


# ================================================================
# Visualization
# ================================================================
def visualize(img, pred, args, track_id=None, fall_prob=None,
              is_falling=False, action_name=None):
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
        if action_name is not None:
            parts.append(f'[{action_name}]')
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
# Keypoint Tracker with V14 Action + Fall + Rule Post-Processing
# ================================================================
class KptTracker:
    """V14 tracker: kpt features (kpt_dim=5) + 4-class action + binary fall + rules."""

    def __init__(self, model, tracker_cfg, max_seq_len=30, rule_processor=None):
        self.model = model
        self.tracker = MODELS.build(tracker_cfg)
        self.max_seq_len = max_seq_len
        self.rule_processor = rule_processor
        self.feat_buffers = {}
        self.prev_norm = {}

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

        # Match tracked -> predicted for keypoints
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
                    norm_x, norm_y, vis_np, prev_x, prev_y)
                self.prev_norm[tid] = (norm_x.copy(), norm_y.copy())

                # Accumulate in buffer
                if tid not in self.feat_buffers:
                    self.feat_buffers[tid] = collections.deque(
                        maxlen=self.max_seq_len)
                self.feat_buffers[tid].append(kpt_feat)

                # GRU temporal prediction (V14/V15: returns dict)
                seq = np.stack(list(self.feat_buffers[tid]))  # (T, 35)
                seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
                pred_out = self.model.action_head.predict(seq_tensor)

                if isinstance(pred_out, dict):
                    # V14/V15 dict output
                    action_class = int(pred_out['action_class'][0].cpu().item())
                    raw_fall_prob = float(pred_out['fall_prob'][0].cpu().item())
                    if pred_out.get('action_probs') is not None:
                        action_probs = pred_out['action_probs'][0].cpu().numpy()
                        action_name = ACTION4_NAMES[min(action_class, len(ACTION4_NAMES) - 1)]
                    else:
                        # V15 binary: no action classes
                        action_probs = None
                        action_name = None
                else:
                    # V12 legacy: returns tensor
                    action_class = 0
                    action_probs = None
                    action_name = None
                    raw_fall_prob = float(pred_out[0].cpu().item())

                # Rule-based post-processing on fall_prob
                if self.rule_processor:
                    head_y = float(norm_y[0])
                    foot_avg_y = float((norm_y[5] + norm_y[6]) / 2.0)
                    if prev_x is not None:
                        motion = float(np.mean(
                            np.abs(norm_x - prev_x) + np.abs(norm_y - prev_y)))
                        foot_motion = float(
                            (np.abs(norm_x[5] - prev_x[5]) + np.abs(norm_y[5] - prev_y[5]) +
                             np.abs(norm_x[6] - prev_x[6]) + np.abs(norm_y[6] - prev_y[6])) / 2.0)
                    else:
                        motion = 0.0
                        foot_motion = 0.0
                    self.rule_processor.update(tid, head_y, motion,
                                               foot_avg_y, foot_motion,
                                               action_class=action_class)
                    fall_prob = self.rule_processor.adjust_prob(tid, raw_fall_prob)
                else:
                    fall_prob = raw_fall_prob

                # V15: rule-based action state when model has no action head
                if action_name is None and self.rule_processor:
                    rule_hist = list(
                        self.rule_processor.history.get(tid, []))
                    rule_act_idx, rule_act_name = classify_action_by_rules(
                        rule_hist, fall_prob, fall_thr=0.5)
                    action_class = rule_act_idx
                    action_name = rule_act_name
                    # Update action history with rule-based class
                    if tid in self.rule_processor.action_history:
                        self.rule_processor.action_history[tid][-1] = \
                            rule_act_idx

                action_info[tid] = {
                    'action_class': action_class,
                    'action_name': action_name,
                    'action_probs': action_probs,
                    'fall_prob': fall_prob,
                    'raw_fall_prob': raw_fall_prob,
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
    frame_fall_flags = []  # per-frame: 1 if any person detected falling, else 0

    for i, img_path in enumerate(image_files):
        fname = os.path.basename(img_path)

        img, track_inst, action_info = tracker.process(
            img_path, pipeline, frame_id=i, device=device)

        if img is None:
            frame_fall_flags.append(0)
            continue

        if track_inst is None or len(track_inst) == 0:
            cv2.imwrite(os.path.join(output_dir, fname), img)
            frame_fall_flags.append(0)
            continue

        img_vis = img.copy()
        has_fall = False

        for t_idx in range(len(track_inst)):
            tid = int(track_inst.instances_id[t_idx])
            info = action_info.get(tid, {})
            fall_prob = info.get('fall_prob', 0.0)
            action_name = info.get('action_name', None)
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
                                is_falling=is_falling,
                                action_name=action_name)
            if is_falling:
                has_fall = True

        if has_fall:
            fall_count += 1
        frame_fall_flags.append(1 if has_fall else 0)

        cv2.imwrite(os.path.join(output_dir, fname), img_vis)

    return len(image_files), fall_count, frame_fall_flags


# ================================================================
# Main
# ================================================================
def main():
    args = parse_args()

    print("=" * 70)
    print("RTMDet-Pose V14/V15: Action + Binary Fall Detection + Rules")
    print("=" * 70)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Radius:     {args.radius}")
    print(f"Fall thr:   {args.fall_thr}")
    print(f"Actions:    {ACTION4_NAMES}")
    print(f"Rules: head_drop={args.rule_head_drop_thr}, "
          f"head_low={args.rule_head_low_thr}, "
          f"static_frames={args.rule_static_frames}, "
          f"min_fall_frames={args.min_fall_frames}, "
          f"velocity_suppress={args.rule_velocity_suppress_thr}")
    print(f"V15 rules: head_vel_suppress={args.rule_head_vel_suppress}, "
          f"head_vel_boost={args.rule_head_vel_boost}, "
          f"head_rise_thr={args.rule_head_rise_thr}")
    print("=" * 70)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nLoading model... (device: {device})")
    model = init_detector(args.config, args.checkpoint, device=device)
    model.eval()
    has_action = hasattr(model, 'action_head') and model.action_head is not None
    if has_action:
        num_cls = getattr(model.action_head, 'num_classes', '?')
        num_act = getattr(model.action_head, 'num_action_classes', num_cls)
        focal = getattr(model.action_head, 'focal_gamma', 0.0)
        print(f"Model loaded! action_head=YES, num_classes={num_cls}, "
              f"num_action_classes={num_act}, focal_gamma={focal}")
    else:
        print("Model loaded! action_head=NO")

    cfg = Config.fromfile(args.config)
    tracker_cfg = cfg.get('tracker', None)
    if tracker_cfg is None:
        raise RuntimeError('No tracker config found in config file')

    rule_processor = RulePostProcessor(
        head_drop_thr=args.rule_head_drop_thr,
        head_low_thr=args.rule_head_low_thr,
        static_frames=args.rule_static_frames,
        sit_frames=args.rule_sit_frames,
        sit_max_head_speed=args.rule_sit_max_head_speed,
        sit_max_foot_motion=args.rule_sit_max_foot_motion,
        lying_suppress_frames=args.rule_lying_suppress_frames,
        getup_frames=args.rule_getup_frames,
        transition_window=args.rule_transition_window,
        min_fall_frames=args.min_fall_frames,
        velocity_suppress_thr=args.rule_velocity_suppress_thr,
        head_vel_suppress=args.rule_head_vel_suppress,
        head_vel_boost=args.rule_head_vel_boost,
        head_rise_thr=args.rule_head_rise_thr,
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

    # GT-based metrics tracking
    # Sequence-level: one entry per sequence
    seq_gt_list = []      # 1=has any falling frame in annotation, 0=none
    seq_pred_list = []     # 1=any frame detected fall, 0=none
    # Frame-level: per-frame GT from annotation falling field
    frame_gt_list = []     # 1=annotation says falling, 0=not
    frame_pred_list = []   # 1=fall detected in frame, 0=not
    # Per-sequence detail for printing
    seq_details = []

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

            # Load per-frame GT from annotation
            ann_file = os.path.join(action_path, camera_dir,
                                    'annotations', 'merged_person_bbox_keypoints.json')
            frame_gt_map = {}  # filename -> 1/0
            has_ann = False
            if os.path.exists(ann_file):
                has_ann = True
                with open(ann_file) as f:
                    ann_data = json.load(f)
                # Build image_id -> filename mapping
                id2name = {img['id']: img['file_name'] for img in ann_data['images']}
                # Per-frame: falling=True if any annotation on that frame has falling=True
                for ann in ann_data['annotations']:
                    fname = id2name.get(ann['image_id'], '')
                    fall_flag = 1 if ann['attributes'].get('falling', False) else 0
                    # Take max (any person falling -> frame is falling)
                    frame_gt_map[fname] = max(frame_gt_map.get(fname, 0), fall_flag)

            out_dir = os.path.join(output_root, action_dir, camera_dir)
            gt_fall_frames = sum(frame_gt_map.values()) if has_ann else 0
            ann_str = f'GT:{gt_fall_frames}fall' if has_ann else 'no-ann'
            print(f"\n[{action_dir}/{camera_dir}] {len(image_files)} frames ({ann_str}) -> {out_dir}")

            n_imgs, n_falls, frame_flags = process_sequence(
                model, tracker, pipeline, image_files, out_dir, args, device)

            total_images += n_imgs
            total_falls += n_falls
            total_sequences += 1

            # Sequence-level GT: has any falling frame in annotation
            seq_gt = 1 if gt_fall_frames > 0 else 0
            seq_pred = 1 if n_falls > 0 else 0
            seq_gt_list.append(seq_gt)
            seq_pred_list.append(seq_pred)

            # Frame-level GT from annotation
            if has_ann:
                for img_path, pred_flag in zip(image_files, frame_flags):
                    fname = os.path.basename(img_path)
                    gt_flag = frame_gt_map.get(fname, 0)
                    frame_gt_list.append(gt_flag)
                    frame_pred_list.append(pred_flag)

            # Detail
            status = ''
            if seq_gt == 1 and seq_pred == 1:
                status = 'TP'
            elif seq_gt == 1 and seq_pred == 0:
                status = 'FN (MISSED!)'
            elif seq_gt == 0 and seq_pred == 1:
                status = 'FP (FALSE ALARM)'
            else:
                status = 'TN'

            seq_details.append({
                'name': f'{action_dir}/{camera_dir}',
                'gt': seq_gt,
                'gt_fall_frames': gt_fall_frames,
                'pred': seq_pred,
                'n_frames': n_imgs,
                'n_fall_frames': n_falls,
                'status': status,
            })

            gt_label = 'FALL' if seq_gt else 'normal'
            print(f"  Done: {n_imgs} frames, {n_falls} pred fall | GT={gt_label}({gt_fall_frames}f) -> {status}")

    # ================================================================
    # Print metrics summary
    # ================================================================
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)
    print(f"Total: {total_sequences} sequences, {total_images} frames")
    print(f"Fall threshold: {args.fall_thr}")
    print()

    # Per-sequence detail table
    print("-" * 80)
    print(f"{'Sequence':<45} {'GT':>4} {'GTf':>4} {'Pred':>4} {'Pf':>5} {'Status':>14}")
    print("-" * 80)
    for d in seq_details:
        gt_str = 'FALL' if d['gt'] else 'norm'
        pred_str = 'FALL' if d['pred'] else 'norm'
        print(f"{d['name']:<45} {gt_str:>4} {d['gt_fall_frames']:>4} "
              f"{pred_str:>4} {d['n_fall_frames']:>4}f {d['status']:>14}")
    print("-" * 80)

    # Compute and print metrics
    def compute_metrics(gt, pred, level_name):
        gt = np.array(gt)
        pred = np.array(pred)
        tp = int(((gt == 1) & (pred == 1)).sum())
        fn = int(((gt == 1) & (pred == 0)).sum())
        fp = int(((gt == 0) & (pred == 1)).sum())
        tn = int(((gt == 0) & (pred == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / len(gt) if len(gt) > 0 else 0.0

        print(f"\n[{level_name} Metrics]")
        print(f"  Positives (fall): {tp + fn}  |  Negatives (normal): {fp + tn}")
        print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
        print(f"  Recall    = {recall:.4f}  ({tp}/{tp+fn})")
        print(f"  Precision = {precision:.4f}  ({tp}/{tp+fp})")
        print(f"  F1        = {f1:.4f}")
        print(f"  FPR       = {fpr:.4f}  ({fp}/{fp+tn})")
        print(f"  Accuracy  = {accuracy:.4f}")

        # Target comparison
        print(f"  --- Target: Recall>=1.0, FPR<=0.05 ---")
        recall_gap = max(0, 1.0 - recall)
        fpr_gap = max(0, fpr - 0.05)
        if recall_gap == 0 and fpr_gap == 0:
            print(f"  >>> TARGET MET! <<<")
        else:
            if recall_gap > 0:
                print(f"  Recall gap:  -{recall_gap:.4f} (need +{fn} more TP)")
            if fpr_gap > 0:
                print(f"  FPR gap:     +{fpr_gap:.4f} (need -{fp - int(0.05 * (fp+tn))} fewer FP)")

    if len(seq_gt_list) > 0:
        compute_metrics(seq_gt_list, seq_pred_list, "Sequence-Level")

    print("\n" + "=" * 70)
    print(f"Output: {output_root}")


if __name__ == '__main__':
    main()

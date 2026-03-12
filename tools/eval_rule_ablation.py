#!/usr/bin/env python3
"""
Rule-based post-processing ablation for Table 8.

Evaluates cumulative rule contributions on the Proposed method (V14c)
on the validation set.

Approach:
  1. Load model, run inference on val set, collect raw fall_prob per instance
  2. For each instance, extract temporal features from GT kpt_sequence
  3. Simulate RulePostProcessor with different rule combinations
  4. Compute binary fall metrics for each configuration

Usage:
    python tools/eval_rule_ablation.py \
        configs/rtmdet_bai/rtmdet_pose_v14c_action4.py \
        work_dirs/rtmdet_pose_v14c/best_action_fall_f1_epoch_6.pth
"""

import argparse
import collections
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from mmengine.config import Config
from mmengine.fileio import load
from mmengine.runner import Runner
from mmengine.logging import MMLogger

# Fall class IDs (10-class GT)
FALLING_CLASS_IDS = {6, 7, 8, 9}
NUM_KEYPOINTS = 7
KPT_DIM = 5  # x, y, vis, dx, dy


def extract_features_from_kpt_seq(kpt_seq):
    """Extract per-frame rule features from kpt_sequence.

    kpt_seq: (T, 35) tensor or array — [x, y, vis, dx, dy] × 7 keypoints
    Keypoints: 0=Head, 1=ShoulderC, 2=HandR, 3=HandL, 4=HipsC, 5=FootR, 6=FootL

    Returns list of (head_y, motion_mag, foot_avg_y, foot_motion) per frame.
    """
    if isinstance(kpt_seq, torch.Tensor):
        kpt_seq = kpt_seq.cpu().numpy()

    T = kpt_seq.shape[0]
    features = []
    for t in range(T):
        frame = kpt_seq[t]  # (35,)

        # Head y-coordinate (keypoint 0, index 1)
        head_y = float(frame[0 * KPT_DIM + 1])

        # Motion magnitude: mean of sqrt(dx^2 + dy^2) over all keypoints
        motions = []
        for k in range(NUM_KEYPOINTS):
            dx = float(frame[k * KPT_DIM + 3])
            dy = float(frame[k * KPT_DIM + 4])
            motions.append(np.sqrt(dx ** 2 + dy ** 2))
        motion_mag = float(np.mean(motions))

        # Foot average y-coordinate (keypoints 5, 6)
        foot_r_y = float(frame[5 * KPT_DIM + 1])
        foot_l_y = float(frame[6 * KPT_DIM + 1])
        foot_avg_y = (foot_r_y + foot_l_y) / 2.0

        # Foot motion magnitude
        foot_r_dx = float(frame[5 * KPT_DIM + 3])
        foot_r_dy = float(frame[5 * KPT_DIM + 4])
        foot_l_dx = float(frame[6 * KPT_DIM + 3])
        foot_l_dy = float(frame[6 * KPT_DIM + 4])
        foot_r_motion = np.sqrt(foot_r_dx ** 2 + foot_r_dy ** 2)
        foot_l_motion = np.sqrt(foot_l_dx ** 2 + foot_l_dy ** 2)
        foot_motion = float((foot_r_motion + foot_l_motion) / 2.0)

        features.append((head_y, motion_mag, foot_avg_y, foot_motion))

    return features


class RulePostProcessorAblation:
    """Simplified RulePostProcessor for ablation.

    Applies rules cumulatively based on enabled_rules set.
    Same logic as infer_v14.py's RulePostProcessor.
    """

    def __init__(self, enabled_rules=None):
        """
        enabled_rules: set of rule names to enable. Options:
            'head_drop'       — Rule 1: Head rapid drop boost
            'lying_still'     — Rule 2: Lying still suppression
            'sit_down'        — Rule 3: Sit-down suppression
            'action_lying'    — Rule 5: Action-gated lying suppression
            'getup'           — Rule 6: Getting-up detection
            'sit_steady'      — Rule 7: Sitting steady state suppression
            'velocity_gate'   — Rule 8: Velocity gate
            'direction_speed' — Rule 10: Direction + speed gate
            'consec_confirm'  — Rule 9: Consecutive frame confirmation
        """
        self.enabled = enabled_rules or set()

        # Default parameters (same as infer_v14.py defaults)
        self.head_drop_thr = 0.15
        self.head_low_thr = 0.7
        self.static_frames = 5
        self.sit_frames = 5
        self.sit_max_head_speed = 0.06
        self.sit_max_foot_motion = 0.03
        self.lying_suppress_frames = 5
        self.getup_frames = 3
        self.transition_window = 8
        self.min_fall_frames = 1
        self.velocity_suppress_thr = 0.02
        self.head_vel_suppress = 0.04
        self.head_vel_boost = 0.08
        self.head_rise_thr = 0.03

    def adjust_prob(self, hist, act_hist, raw_prob):
        """Apply enabled rules to adjust fall probability.

        Args:
            hist: list of (head_y, motion_mag, foot_avg_y, foot_motion) tuples
            act_hist: list of action_class predictions (0-3)
            raw_prob: raw fall probability from model

        Returns:
            adjusted fall probability
        """
        if len(hist) < 2:
            return raw_prob

        # Rule 1: Head rapid drop -> boost
        if 'head_drop' in self.enabled and len(hist) >= 3:
            y_now = hist[-1][0]
            y_3ago = hist[-3][0]
            drop = y_now - y_3ago
            if drop > self.head_drop_thr:
                foot_motion_avg = np.mean([h[3] for h in hist[-3:]])
                if foot_motion_avg > self.sit_max_foot_motion:
                    boost = min(drop / self.head_drop_thr * 0.3, 0.4)
                    raw_prob = min(raw_prob + boost, 1.0)

        # Rule 2: Lying still suppression
        if 'lying_still' in self.enabled and len(hist) >= self.static_frames:
            recent = hist[-self.static_frames:]
            all_low = all(h[0] > self.head_low_thr for h in recent)
            all_static = all(h[1] < 0.02 for h in recent)
            if all_low and all_static:
                raw_prob = raw_prob * 0.3

        # Rule 3: Sit-down suppression
        if 'sit_down' in self.enabled and len(hist) >= self.sit_frames:
            recent = hist[-self.sit_frames:]
            head_ys = [h[0] for h in recent]
            total_head_drop = head_ys[-1] - head_ys[0]
            if total_head_drop > 0.05:
                per_frame_drops = [head_ys[j + 1] - head_ys[j]
                                   for j in range(len(head_ys) - 1)]
                max_per_frame = max(per_frame_drops)
                foot_motions = [h[3] for h in recent]
                avg_foot_motion = np.mean(foot_motions)
                if (max_per_frame < self.sit_max_head_speed and
                        avg_foot_motion < self.sit_max_foot_motion):
                    raw_prob = raw_prob * 0.3

        # Rule 5: Action-gated lying suppression
        if 'action_lying' in self.enabled and len(act_hist) >= self.lying_suppress_frames:
            recent_acts = act_hist[-self.lying_suppress_frames:]
            all_lying = all(a == 3 for a in recent_acts)
            if all_lying:
                window = act_hist[-min(len(act_hist), self.transition_window):]
                had_upright = any(a in (0, 1) for a in window)
                if not had_upright:
                    raw_prob = raw_prob * 0.3

        # Rule 6: Getting-up detection
        if 'getup' in self.enabled and len(hist) >= self.getup_frames:
            recent_ys = [h[0] for h in hist[-self.getup_frames:]]
            head_rise = recent_ys[0] - recent_ys[-1]
            if head_rise > 0.1:
                raw_prob = raw_prob * 0.3

        # Rule 7: Sitting steady state suppression
        if 'sit_steady' in self.enabled and len(act_hist) >= self.lying_suppress_frames:
            recent_acts = act_hist[-self.lying_suppress_frames:]
            all_sitting = all(a == 2 for a in recent_acts)
            if all_sitting:
                raw_prob = raw_prob * 0.4

        # Rule 8: Velocity gate
        if 'velocity_gate' in self.enabled and len(hist) >= 3:
            recent_motions = [h[1] for h in hist[-3:]]
            avg_motion = np.mean(recent_motions)
            if avg_motion < self.velocity_suppress_thr:
                y_now = hist[-1][0]
                y_prev = hist[-3][0]
                head_drop = y_now - y_prev
                if head_drop < 0.05:
                    raw_prob = raw_prob * 0.2

        # Rule 10: Direction + speed gate
        if 'direction_speed' in self.enabled and len(hist) >= 3:
            head_vel = (hist[-1][0] - hist[-3][0]) / 2.0
            if len(hist) >= 5:
                vel_old = (hist[-3][0] - hist[-5][0]) / 2.0
                head_accel = head_vel - vel_old
            else:
                head_accel = 0.0

            if head_vel < -self.head_rise_thr:
                raw_prob *= 0.15
            elif 0 < head_vel < self.head_vel_suppress:
                raw_prob *= 0.4
            elif head_vel > self.head_vel_boost and head_accel > 0.02:
                raw_prob = min(raw_prob + 0.3, 1.0)

        # Rule 9: Consecutive frame confirmation
        # For per-instance evaluation with 8-frame sequences,
        # this rule is less meaningful (no cross-sequence state).
        # Skip for ablation as it requires true sequential tracking.

        return raw_prob


def compute_binary_metrics(pred_falling, gt_falling, fall_probs=None):
    """Compute binary fall detection metrics."""
    tp = int(((pred_falling == 1) & (gt_falling == 1)).sum())
    fp = int(((pred_falling == 1) & (gt_falling == 0)).sum())
    fn = int(((pred_falling == 0) & (gt_falling == 1)).sum())
    tn = int(((pred_falling == 0) & (gt_falling == 0)).sum())

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': prec, 'recall': rec, 'f1': f1,
    }


def run_inference_collect(config_path, checkpoint_path):
    """Run model inference on val set, collect per-instance data.

    Returns list of dicts:
        raw_fall_prob, pred_action_class, gt_action_class, kpt_sequence
    """
    cfg = Config.fromfile(config_path)

    # Get annotation file path
    ann_file = cfg.test_dataloader.dataset.ann_file
    if not os.path.isabs(ann_file):
        ann_file = os.path.join(cfg.data_root, ann_file)

    # Load GT data from annotations
    print(f"[INFO] Loading annotations from {ann_file}")
    ann_data = load(ann_file)

    # Build GT lookup: img_id -> list of {bbox, action_class, kpt_sequence}
    gt_data = {}
    for ann in ann_data.get('annotations', []):
        img_id = ann['image_id']
        attrs = ann.get('attributes', {})
        if not isinstance(attrs, dict):
            attrs = {}

        action_class = int(attrs.get('action_class', 0))

        kpt_seq = None
        if 'kpt_sequence' in attrs:
            kpt_seq = np.array(attrs['kpt_sequence'], dtype=np.float32)

        if img_id not in gt_data:
            gt_data[img_id] = []
        gt_data[img_id].append({
            'bbox': ann['bbox'],  # [x, y, w, h] COCO format
            'action_class': action_class,
            'kpt_sequence': kpt_seq,
        })

    # Build and run model
    print(f"[INFO] Loading model from {checkpoint_path}")
    cfg.load_from = checkpoint_path

    # Ensure work_dir doesn't conflict
    cfg.work_dir = 'work_dirs/_rule_ablation_tmp'

    # Build runner for testing
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model
    model.eval()

    dataloader = runner.test_dataloader

    # Collect per-instance data
    results = []
    iou_thr = 0.5

    print(f"[INFO] Running inference on {len(dataloader)} batches...")
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            # Forward pass
            outputs = model.test_step(data_batch)

            for data_sample in outputs:
                pred = data_sample.pred_instances
                img_id = data_sample.img_id

                if not hasattr(pred, 'fall_prob') or pred.fall_prob is None:
                    continue

                pred_bboxes = pred.bboxes.cpu().numpy()
                pred_scores = pred.scores.cpu().numpy()
                fall_probs = pred.fall_prob.cpu().numpy()

                # Get action_class predictions
                if hasattr(pred, 'action_class') and pred.action_class is not None:
                    pred_action = pred.action_class.cpu().numpy()
                else:
                    pred_action = np.zeros(len(pred_bboxes), dtype=int)

                gt_list = gt_data.get(img_id, [])
                if len(gt_list) == 0 or len(pred_bboxes) == 0:
                    continue

                gt_bboxes = np.array([
                    [g['bbox'][0], g['bbox'][1],
                     g['bbox'][0] + g['bbox'][2], g['bbox'][1] + g['bbox'][3]]
                    for g in gt_list
                ])
                gt_classes = np.array([g['action_class'] for g in gt_list])
                gt_kpt_seqs = [g['kpt_sequence'] for g in gt_list]

                # Compute IoU
                ious = _compute_iou(pred_bboxes, gt_bboxes)

                # Match predictions to GT
                gt_matched = set()
                for pred_idx in np.argsort(-pred_scores):
                    if ious.shape[1] == 0:
                        break
                    best_gt = ious[pred_idx].argmax()
                    if (ious[pred_idx, best_gt] >= iou_thr
                            and best_gt not in gt_matched):
                        gt_matched.add(best_gt)
                        results.append({
                            'raw_fall_prob': float(fall_probs[pred_idx]),
                            'pred_action_class': int(pred_action[pred_idx]),
                            'gt_action_class': int(gt_classes[best_gt]),
                            'kpt_sequence': gt_kpt_seqs[best_gt],
                        })

            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches, "
                      f"collected {len(results)} instances")

    print(f"[INFO] Collected {len(results)} matched instances")
    return results


def _compute_iou(bboxes1, bboxes2):
    """Compute IoU between two sets of bboxes (xyxy format)."""
    x1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0:1].T)
    y1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1:2].T)
    x2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2:3].T)
    y2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3:4].T)
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-8)


def run_rule_ablation(results):
    """Apply different rule configurations and compute metrics.

    Cumulative ablation (each row adds one more rule):
    1. Model only (no rules)
    2. + Head rapid drop boost (Rule 1)
    3. + Lying still suppression (Rule 2)
    4. + Velocity gate (Rule 8)
    5. + Direction/speed gate (Rule 10)
    6. All rules combined (adds Rules 3, 5, 6, 7)
    """
    gt_action_classes = np.array([r['gt_action_class'] for r in results])
    gt_falling = np.array([1 if c in FALLING_CLASS_IDS else 0
                           for c in gt_action_classes])
    raw_fall_probs = np.array([r['raw_fall_prob'] for r in results])

    # Extract per-instance temporal features
    print("[INFO] Extracting temporal features from kpt_sequences...")
    instance_features = []
    instance_act_hist = []
    for r in results:
        kpt_seq = r['kpt_sequence']
        if kpt_seq is not None and len(kpt_seq) > 0:
            feats = extract_features_from_kpt_seq(kpt_seq)
            # Action history: repeat predicted action for all T frames
            # (since we only have one prediction per instance)
            act_hist = [r['pred_action_class']] * len(feats)
        else:
            feats = []
            act_hist = []
        instance_features.append(feats)
        instance_act_hist.append(act_hist)

    # Define cumulative rule configurations
    configs = [
        ('Model only (no rules)', set()),
        ('+ Head rapid drop boost', {'head_drop'}),
        ('+ Lying still suppression', {'head_drop', 'lying_still'}),
        ('+ Velocity gate', {'head_drop', 'lying_still', 'velocity_gate'}),
        ('+ Direction/speed gate', {'head_drop', 'lying_still', 'velocity_gate',
                                     'direction_speed'}),
        ('All rules combined', {'head_drop', 'lying_still', 'velocity_gate',
                                 'direction_speed', 'sit_down', 'action_lying',
                                 'getup', 'sit_steady'}),
    ]

    print("\n" + "=" * 85)
    print(f"{'Configuration':<30s} {'Fall F1':>8s} {'Prec':>8s} {'Recall':>8s} "
          f"{'FP':>5s} {'TP':>5s} {'FN':>5s} {'TN':>5s}")
    print("=" * 85)

    for config_name, enabled_rules in configs:
        processor = RulePostProcessorAblation(enabled_rules=enabled_rules)

        adjusted_probs = np.zeros(len(results))
        for i in range(len(results)):
            if len(enabled_rules) == 0:
                adjusted_probs[i] = raw_fall_probs[i]
            else:
                adjusted_probs[i] = processor.adjust_prob(
                    instance_features[i],
                    instance_act_hist[i],
                    raw_fall_probs[i]
                )

        pred_falling = (adjusted_probs >= 0.5).astype(int)
        metrics = compute_binary_metrics(pred_falling, gt_falling, adjusted_probs)

        print(f"{config_name:<30s} {metrics['f1']:8.3f} {metrics['precision']:8.3f} "
              f"{metrics['recall']:8.3f} {metrics['fp']:5d} {metrics['tp']:5d} "
              f"{metrics['fn']:5d} {metrics['tn']:5d}")

    print("=" * 85)

    # Also print LaTeX table format
    print("\n% LaTeX table rows (copy-paste into experiments.tex):")
    for config_name, enabled_rules in configs:
        processor = RulePostProcessorAblation(enabled_rules=enabled_rules)

        adjusted_probs = np.zeros(len(results))
        for i in range(len(results)):
            if len(enabled_rules) == 0:
                adjusted_probs[i] = raw_fall_probs[i]
            else:
                adjusted_probs[i] = processor.adjust_prob(
                    instance_features[i],
                    instance_act_hist[i],
                    raw_fall_probs[i]
                )

        pred_falling = (adjusted_probs >= 0.5).astype(int)
        m = compute_binary_metrics(pred_falling, gt_falling, adjusted_probs)

        latex_name = config_name.replace('+ ', '+ ').replace('(no rules)', '(no rules)')
        print(f"        {latex_name} & {m['f1']:.3f} & {m['precision']:.3f} "
              f"& {m['recall']:.3f} & {m['fp']} \\\\")


def main():
    parser = argparse.ArgumentParser(description='Rule post-processing ablation')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    args = parser.parse_args()

    # Initialize logger
    MMLogger.get_instance('mmdet', log_level='WARNING')

    # Step 1: Run inference and collect data
    results = run_inference_collect(args.config, args.checkpoint)

    if len(results) == 0:
        print("[ERROR] No matched instances found!")
        return

    # Step 2: Run rule ablation
    run_rule_ablation(results)


if __name__ == '__main__':
    main()

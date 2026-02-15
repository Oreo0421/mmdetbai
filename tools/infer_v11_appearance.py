#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTMDet-Pose V11 Appearance 推理脚本: 骨骼(36) + RoI外观(96) = 132维时序动作分类

与V10不同：不用高层 model.test_step()，手动 extract_feat → bbox_head → ByteTrack
→ 对tracked bbox 提取 roi_feats → GAP → concat(bone_36, appearance_96) = 132维

遍历 infer/{action}/{camera}/undistorted_stitched_png/*.png
输出到 output/{action}/{camera}/*.png

Usage:
  python tools/infer_v11_appearance.py \
      --input /home/tbai/Desktop/infer \
      --output /home/tbai/Desktop/fall_infer/v11
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import math
import collections
import torch
import argparse

from mmdet.apis import init_detector
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_overlaps


# ---- 10 Action Classes ----
ACTION_CLASSES = [
    'Standing still',       # 0
    'Walking',              # 1
    'Sitting down',         # 2
    'Standing up',          # 3
    'Lying down',           # 4
    'Getting up',           # 5
    'Falling walking',      # 6  *
    'Falling standing',     # 7  *
    'Falling sitting',      # 8  *
    'Falling standing up',  # 9  *
]
FALLING_IDS = {6, 7, 8, 9}

# V10 bone connections: head(0), shoulder(1), hand_R(2), hand_L(3), hips(4), foot_R(5), foot_L(6)
BONE_CONNECTIONS = [
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
        description='RTMDet-Pose V11: Bone + Appearance Inference')
    parser.add_argument('--config',
                        default='/home/tbai/mmdetection/mmdetection/configs/rtmdet_bai/rtmdet_pose_v11_appearance.py',
                        help='config path')
    parser.add_argument('--checkpoint',
                        default='work_dirs/rtmdet_pose_v11_appearance/epoch_50.pth',
                        help='checkpoint path')
    parser.add_argument('--input',
                        default='/home/tbai/Desktop/infer',
                        help='input root dir')
    parser.add_argument('--output',
                        default='/home/tbai/Desktop/fall_infer/v11',
                        help='output root dir')
    parser.add_argument('--det-thr', type=float, default=0.3)
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument('--fall-thr', type=float, default=0.5)
    parser.add_argument('--max-seq-len', type=int, default=30)
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--thickness', type=int, default=1)
    args = parser.parse_args()
    return args


# ================================================================
# Bone feature computation
# ================================================================
def kpts_to_bones(norm_x, norm_y):
    """Compute bone features from normalized keypoints."""
    bones = []
    for start, end in BONE_CONNECTIONS:
        dx = float(norm_x[end] - norm_x[start])
        dy = float(norm_y[end] - norm_y[start])
        length = math.sqrt(dx * dx + dy * dy + 1e-8)
        angle = math.atan2(dx, dy)
        bones.append((dx, dy, angle, length))
    return bones


def build_bone_feature(bones_curr, bones_prev=None):
    """Build per-bone feature: [dx, dy, angle, length, d_angle, d_length] x 6 = 36."""
    feat = []
    for i, (dx, dy, angle, length) in enumerate(bones_curr):
        if bones_prev is not None:
            _, _, prev_angle, prev_length = bones_prev[i]
            d_angle = angle - prev_angle
            d_angle = (d_angle + math.pi) % (2 * math.pi) - math.pi
            d_length = length - prev_length
        else:
            d_angle = 0.0
            d_length = 0.0
        feat.extend([dx, dy, angle, length, d_angle, d_length])
    return np.array(feat, dtype=np.float32)


# ================================================================
# Low-level inference: extract_feat → bbox_head → NMS → pose → roi GAP
# ================================================================
@torch.no_grad()
def inference_low_level(model, img_path, pipeline, device):
    """Run model forward manually to get both detections and RoI features.

    Returns:
        img: original image (BGR)
        pred_instances: InstanceData with bboxes, scores, keypoints, keypoint_scores
        roi_gap_feats: (N, C) GAP features from RoI-aligned features, or None
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    data_info = {
        'img_path': img_path,
        'img_id': 0,
        'ori_shape': (h, w),
        'height': h,
        'width': w,
    }
    data = pipeline(data_info)
    inputs = data['inputs'].unsqueeze(0).to(device)  # (1, C, H, W)

    # Preprocess
    data_preprocessor = model.data_preprocessor
    batch_inputs = data_preprocessor({'inputs': [data['inputs'].to(device)],
                                       'data_samples': [data['data_samples'].to(device)]})
    batch_imgs = batch_inputs['inputs']

    # 1. Extract multi-scale features
    x = model.extract_feat(batch_imgs)

    # 2. Detection: bbox_head forward + predict_by_feat
    outs = model.bbox_head(x)
    batch_img_metas = [data['data_samples'].metainfo]

    cfg = model.test_cfg
    det_results = model.bbox_head.predict_by_feat(
        *outs, batch_img_metas=batch_img_metas, cfg=cfg, rescale=True)

    if len(det_results) == 0 or len(det_results[0]) == 0:
        return img, None, None

    pred = det_results[0]

    # 3. Build RoIs for pose (in feature space, no rescale)
    det_results_feat = model.bbox_head.predict_by_feat(
        *outs, batch_img_metas=batch_img_metas, cfg=cfg, rescale=False)
    pred_feat = det_results_feat[0]

    if len(pred_feat) == 0:
        return img, pred, None

    bboxes_feat = pred_feat.bboxes
    # Filter by score
    valid = pred_feat.scores >= 0.3
    if not valid.any():
        return img, pred, None
    bboxes_feat = bboxes_feat[valid]

    # Build RoI tensor: (N, 5) = [batch_idx, x1, y1, x2, y2]
    batch_inds = bboxes_feat.new_zeros(bboxes_feat.size(0), 1)
    rois = torch.cat([batch_inds, bboxes_feat], dim=1)

    # 4. RoI feature extraction
    roi_feats = model.pose_roi_extractor(x, rois)  # (N, C, H, W)

    # 5. Pose prediction from RoI features
    pose_kpts, pose_scores = model.pose_head.predict(
        roi_feats, rois, [data['data_samples']], rescale=True)

    # Attach pose to pred (only valid detections)
    num_kpts = pose_kpts.size(1) if pose_kpts.numel() > 0 else 7
    pred.keypoints = torch.zeros(len(pred), num_kpts, 2, device=device)
    pred.keypoint_scores = torch.zeros(len(pred), num_kpts, device=device)

    # Match valid detections back to pred via IoU
    if len(pred) > 0 and bboxes_feat.numel() > 0:
        # Use rescaled bboxes for matching
        pred_bboxes_rescaled = pred.bboxes
        # pred_feat valid bboxes → rescale for matching
        scale_factor = batch_img_metas[0].get('scale_factor', (1.0, 1.0))
        if isinstance(scale_factor, (tuple, list)):
            sx, sy = float(scale_factor[0]), float(scale_factor[1])
        else:
            sx = sy = float(scale_factor)
        bboxes_rescaled = bboxes_feat.clone()
        bboxes_rescaled[:, 0::2] /= sx
        bboxes_rescaled[:, 1::2] /= sy

        ious = bbox_overlaps(pred_bboxes_rescaled, bboxes_rescaled)
        for j in range(bboxes_rescaled.size(0)):
            best_pred = ious[:, j].argmax()
            if ious[best_pred, j] > 0.3 and j < pose_kpts.size(0):
                pred.keypoints[best_pred] = pose_kpts[j]
                pred.keypoint_scores[best_pred] = pose_scores[j]

    # 6. GAP features from RoI: (N_valid, C)
    roi_gap = roi_feats.mean(dim=[2, 3])  # (N_valid, C)

    # Build full GAP aligned with pred
    gap_full = torch.zeros(len(pred), roi_gap.size(1), device=device)
    if len(pred) > 0 and bboxes_feat.numel() > 0:
        for j in range(bboxes_rescaled.size(0)):
            best_pred = ious[:, j].argmax()
            if ious[best_pred, j] > 0.3 and j < roi_gap.size(0):
                gap_full[best_pred] = roi_gap[j]

    return img, pred, gap_full


# ================================================================
# Visualization
# ================================================================
def visualize(img, pred, args, track_id=None, action_cls=None,
              action_name=None, fall_prob=None, is_falling=False):
    img_vis = img.copy()

    bboxes = pred.bboxes.cpu().numpy()
    bbox_scores = pred.scores.cpu().numpy()
    keypoints = pred.keypoints.cpu().numpy() if hasattr(pred, 'keypoints') else np.zeros((len(bboxes), 0, 2))
    kpt_scores = pred.keypoint_scores.cpu().numpy() if hasattr(pred, 'keypoint_scores') else np.zeros((len(bboxes), 0))

    for idx in range(len(bboxes)):
        det_score = bbox_scores[idx]

        # Box color
        if is_falling:
            box_color = COLOR_FALLING
        elif track_id is not None:
            box_color = COLOR_TRACK[track_id % len(COLOR_TRACK)]
        else:
            box_color = COLOR_NORMAL

        x1, y1, x2, y2 = map(int, bboxes[idx][:4])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), box_color, args.thickness)

        # Label
        parts = []
        if track_id is not None:
            parts.append(f'ID:{track_id}')
        parts.append(f'{det_score:.2f}')
        if action_name is not None:
            parts.append(action_name)
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

        # Skeleton
        for (a, b) in BONE_CONNECTIONS:
            if a < len(kpts) and b < len(kpts):
                if scrs[a] > args.kpt_thr and scrs[b] > args.kpt_thr:
                    pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                    pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                    cv2.line(img_vis, pt1, pt2, LIMB_COLOR, args.thickness)

        # Keypoints
        for i, (kpt, score) in enumerate(zip(kpts, scrs)):
            if score > args.kpt_thr:
                x, y = int(kpt[0]), int(kpt[1])
                color = KPT_COLORS[i % len(KPT_COLORS)]
                cv2.circle(img_vis, (x, y), args.radius, color, -1)

    return img_vis


# ================================================================
# Appearance + Bone Tracker
# ================================================================
class AppearanceBoneTracker:
    """V11 tracker: bone features (36) + GAP appearance features (96) = 132-dim."""

    def __init__(self, model, tracker_cfg, max_seq_len=30, appearance_dim=96):
        self.model = model
        self.tracker = MODELS.build(tracker_cfg)
        self.max_seq_len = max_seq_len
        self.appearance_dim = appearance_dim
        # Per-track buffers: stores 132-dim features per frame
        self.feat_buffers = {}
        self.prev_bones = {}

    def reset(self):
        self.tracker.reset()
        self.feat_buffers.clear()
        self.prev_bones.clear()

    def process(self, img_path, pipeline, frame_id, device, det_thr=0.3):
        result = inference_low_level(self.model, img_path, pipeline, device)
        if result[0] is None:
            return None, None, {}
        img, pred, gap_feats = result

        if pred is None or len(pred) == 0:
            return img, None, {}

        # Build DataSample for ByteTracker
        from mmengine.structures import InstanceData as ID
        from mmdet.structures import DetDataSample
        ds = DetDataSample()
        ds.pred_instances = pred
        ds.set_metainfo({'frame_id': frame_id})

        track_inst = self.tracker.track(ds)
        num_tracks = len(track_inst)
        if num_tracks == 0:
            return img, track_inst, {}

        # Match tracked → predicted for keypoints + GAP
        track_kpts = torch.zeros(num_tracks, 7, 2, device=device)
        track_kpt_scores = torch.zeros(num_tracks, 7, device=device)
        track_gap = torch.zeros(num_tracks, gap_feats.size(1) if gap_feats is not None else 96, device=device)

        if len(pred) > 0:
            ious = bbox_overlaps(track_inst.bboxes, pred.bboxes)
            for t in range(num_tracks):
                best = ious[t].argmax()
                if ious[t, best] > 0.3:
                    if hasattr(pred, 'keypoints'):
                        track_kpts[t] = pred.keypoints[best]
                    if hasattr(pred, 'keypoint_scores'):
                        track_kpt_scores[t] = pred.keypoint_scores[best]
                    if gap_feats is not None:
                        track_gap[t] = gap_feats[best]

        track_inst.keypoints = track_kpts
        track_inst.keypoint_scores = track_kpt_scores

        # Action classification with bone + appearance features
        action_info = {}
        if self.model.action_head is not None:
            for i in range(num_tracks):
                tid = int(track_inst.instances_id[i])
                bbox = track_inst.bboxes[i]
                kpt = track_kpts[i]

                x1, y1, x2, y2 = bbox
                w = max(float(x2 - x1), 1.0)
                h = max(float(y2 - y1), 1.0)
                norm_x = ((kpt[:, 0] - x1) / w).clamp(0, 1).cpu().numpy()
                norm_y = ((kpt[:, 1] - y1) / h).clamp(0, 1).cpu().numpy()

                # Compute bone features (36-dim)
                bones_curr = kpts_to_bones(norm_x, norm_y)
                bones_prev = self.prev_bones.get(tid, None)
                bone_feat = build_bone_feature(bones_curr, bones_prev)  # (36,)
                self.prev_bones[tid] = bones_curr

                # GAP appearance features (96-dim)
                app_feat = track_gap[i, :self.appearance_dim].cpu().numpy()  # (96,)

                # Concat: bone(36) + appearance(96) = 132
                combined_feat = np.concatenate([bone_feat, app_feat])  # (132,)

                # Accumulate in buffer
                if tid not in self.feat_buffers:
                    self.feat_buffers[tid] = collections.deque(
                        maxlen=self.max_seq_len)
                self.feat_buffers[tid].append(combined_feat)

                # GRU temporal prediction
                seq = np.stack(list(self.feat_buffers[tid]))  # (T, 132)
                seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, T, 132)
                out = self.model.action_head.predict(seq_tensor)
                probs = out['action_probs'][0].cpu().numpy()
                cls_id = int(out['action_class'][0].cpu().item())
                fall_prob = float(probs[6:10].sum())

                action_info[tid] = {
                    'cls_id': cls_id,
                    'name': ACTION_CLASSES[cls_id],
                    'fall_prob': fall_prob,
                    'probs': probs,
                }

        # Cleanup stale tracks
        active = set(track_inst.instances_id.tolist()) if num_tracks > 0 else set()
        for tid in list(self.feat_buffers.keys()):
            if tid not in active:
                del self.feat_buffers[tid]
                self.prev_bones.pop(tid, None)

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
            img_path, pipeline, frame_id=i, device=device, det_thr=args.det_thr)

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
            cls_id = info.get('cls_id', 0)
            action_name = info.get('name', '')
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
                                action_cls=cls_id,
                                action_name=action_name,
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
    print("RTMDet-Pose V11: Bone + Appearance Fusion Inference")
    print("=" * 70)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Radius:     {args.radius}")
    print(f"Fall thr:   {args.fall_thr}")
    print("=" * 70)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nLoading model... (device: {device})")
    model = init_detector(args.config, args.checkpoint, device=device)
    model.eval()
    has_action = hasattr(model, 'action_head') and model.action_head is not None
    appearance_dim = getattr(model.action_head, 'appearance_dim', 0) if has_action else 0
    print(f"Model loaded! action_head={'YES' if has_action else 'NO'}, "
          f"appearance_dim={appearance_dim}")

    cfg = Config.fromfile(args.config)
    tracker_cfg = cfg.get('tracker', None)
    if tracker_cfg is None:
        raise RuntimeError('No tracker config found in config file')

    tracker = AppearanceBoneTracker(
        model, tracker_cfg,
        max_seq_len=args.max_seq_len,
        appearance_dim=appearance_dim)
    pipeline = Compose(INFER_PIPELINE)

    # Walk the nested directory: input/{action}/{camera}/undistorted_stitched_png/*.png
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

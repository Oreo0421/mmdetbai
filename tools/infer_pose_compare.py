#!/usr/bin/env python3
"""
Stage 1 Pose Inference: compare V4 (Heatmap), Regression, V6 (SimDR).
No tracker, no action head. Pure detection + keypoint visualization.

Usage:
  python tools/infer_pose_compare.py \
      --config CONFIG --checkpoint CKPT \
      --input DIR --output DIR \
      --tag v4
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import torch
import argparse

from mmdet.apis import init_detector
from mmengine.dataset import Compose


SKELETON = [
    (0, 1),  # head -> shoulder
    (1, 2),  # shoulder -> hand_right
    (1, 3),  # shoulder -> hand_left
    (1, 4),  # shoulder -> hips
    (4, 5),  # hips -> foot_right
    (4, 6),  # hips -> foot_left
]

KPT_COLORS = [
    (0, 0, 255),    # head
    (0, 128, 255),  # shoulder
    (0, 255, 255),  # hand_R
    (0, 255, 0),    # hand_L
    (255, 255, 0),  # hips
    (255, 0, 0),    # foot_R
    (255, 0, 255),  # foot_L
]
LIMB_COLOR = (0, 255, 128)
BOX_COLOR = (0, 255, 0)

INFER_PIPELINE = [
    dict(type='LoadImageFromFile', _scope_='mmdet'),
    dict(type='Resize', scale=(384, 384), keep_ratio=False, _scope_='mmdet'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
         _scope_='mmdet'),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True, help='root dir with action subdirs')
    p.add_argument('--output', required=True, help='output root')
    p.add_argument('--tag', default='', help='method tag for output subdir')
    p.add_argument('--det-thr', type=float, default=0.3)
    p.add_argument('--kpt-thr', type=float, default=0.3)
    p.add_argument('--radius', type=int, default=2)
    p.add_argument('--thickness', type=int, default=1)
    return p.parse_args()


def inference_single(model, img_path, pipeline, device):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    data_info = {
        'img_path': img_path,
        'img_id': 0,
        'ori_shape': (h, w),
        'height': h,
        'width': w,
    }
    data = pipeline(data_info)
    data['inputs'] = [data['inputs'].to(device)]
    data['data_samples'] = [data['data_samples'].to(device)]
    with torch.no_grad():
        results = model.test_step(data)
    return results, img


def visualize(img, pred, args):
    img_vis = img.copy()
    bboxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    kpts = pred.keypoints.cpu().numpy() if hasattr(pred, 'keypoints') else None
    kpt_scores = pred.keypoint_scores.cpu().numpy() if hasattr(pred, 'keypoint_scores') else None

    for i in range(len(bboxes)):
        if scores[i] < args.det_thr:
            continue

        x1, y1, x2, y2 = map(int, bboxes[i][:4])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), BOX_COLOR, args.thickness)
        cv2.putText(img_vis, f'{scores[i]:.2f}', (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1)

        if kpts is None or i >= len(kpts):
            continue

        # Draw keypoints only (no skeleton)
        for j in range(len(kpts[i])):
            s = kpt_scores[i][j] if kpt_scores is not None else 1.0
            if s > args.kpt_thr:
                x, y = int(kpts[i][j][0]), int(kpts[i][j][1])
                cv2.circle(img_vis, (x, y), args.radius, KPT_COLORS[j % len(KPT_COLORS)], -1)

    return img_vis


def process_sequence(model, pipeline, img_dir, out_dir, args, device):
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    ])
    if not files:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    for img_path in files:
        results, img = inference_single(model, img_path, pipeline, device)
        if results is None:
            continue
        pred = results[0].pred_instances
        img_vis = visualize(img, pred, args)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), img_vis)

    return len(files)


def main():
    args = parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model: {args.checkpoint}")
    model = init_detector(args.config, args.checkpoint, device=device)
    model.eval()

    pipeline = Compose(INFER_PIPELINE)

    # Walk input dir: expect action_dir/camera_dir/undistorted_stitched_png/
    input_root = args.input
    total = 0

    if os.path.isdir(input_root):
        # Check if it has subdirs with undistorted_stitched_png
        # Structure: input_root/camera_dir/undistorted_stitched_png/ OR input_root/action_dir/camera_dir/undistorted_stitched_png/
        for entry in sorted(os.listdir(input_root)):
            entry_path = os.path.join(input_root, entry)
            if not os.path.isdir(entry_path):
                continue

            img_dir = os.path.join(entry_path, 'undistorted_stitched_png')
            if os.path.isdir(img_dir):
                # Direct: input_root/camera_dir/undistorted_stitched_png/
                out_dir = os.path.join(args.output, args.tag, entry)
                print(f"  [{entry}] ", end='', flush=True)
                n = process_sequence(model, pipeline, img_dir, out_dir, args, device)
                print(f"{n} frames")
                total += n
            else:
                # Nested: input_root/action_dir/camera_dir/undistorted_stitched_png/
                for cam in sorted(os.listdir(entry_path)):
                    cam_path = os.path.join(entry_path, cam)
                    img_dir2 = os.path.join(cam_path, 'undistorted_stitched_png')
                    if os.path.isdir(img_dir2):
                        out_dir = os.path.join(args.output, args.tag, entry, cam)
                        print(f"  [{entry}/{cam}] ", end='', flush=True)
                        n = process_sequence(model, pipeline, img_dir2, out_dir, args, device)
                        print(f"{n} frames")
                        total += n

    print(f"\nDone! Total: {total} frames -> {os.path.join(args.output, args.tag)}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare 10-class action recognition COCO dataset from SenIR data.

Reads per-subfolder merged_person_bbox_keypoints.json, derives action_class
from `activity_folder` in image metadata, merges all annotations, and
splits into 80/20 train/val.

Usage:
    python tools/prepare_10class_coco.py [--src SRC] [--dst DST] [--seed 42]
"""

import argparse
import copy
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


# ---- Class mapping ----
ACTIVITY_FOLDER_TO_CLASS = {
    '1A-1 Standing still': 0,
    '1A-2 Walking': 1,
    '1A-3 Sitting down': 2,
    '1A-4 Standing up': 3,
    '1A-5 Lying down': 4,
    '1A-6 Getting up from lying down': 5,
    '1A-7 Falling while walking': 6,
    '1A-8 Falling while trying to stand': 7,
    '1A-9 Falling while trying to sit': 8,
    '1A-10 Falling and immediately standing up': 9,
}

ACTION_NAMES = [
    'Standing still',
    'Walking',
    'Sitting down',
    'Standing up',
    'Lying down',
    'Getting up',
    'Falling walking',
    'Falling standing',
    'Falling sitting',
    'Falling standing up',
]

FALLING_CLASSES = {6, 7, 8, 9}

CATEGORIES = [
    {
        'id': 1,
        'name': 'human',
        'supercategory': '',
        'keypoints': [
            'head', 'shoulder', 'hand right', 'hand left',
            'hips', 'foot right', 'foot left',
        ],
        'skeleton': [],
    }
]


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare 10-class COCO dataset')
    parser.add_argument(
        '--src', default='/home/tbai/Desktop/SenIRDatasetProcessed/',
        help='Source data root with 10 activity folders')
    parser.add_argument(
        '--dst', default='/home/tbai/Desktop/sensir_coco/',
        help='Output COCO dataset root')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--train-ratio', type=float, default=0.8,
        help='Train split ratio (default: 0.8)')
    parser.add_argument(
        '--symlink', action='store_true', default=True,
        help='Use symlinks instead of copying images (default: True)')
    parser.add_argument(
        '--copy', dest='symlink', action='store_false',
        help='Copy images instead of symlinking')
    parser.add_argument(
        '--seq-len', type=int, default=8,
        help='Temporal sequence length for kpt_sequence (default: 8, 0=disable)')
    parser.add_argument(
        '--bone', action='store_true', default=False,
        help='Use bone/skeleton features (V10) instead of keypoint features (V9)')
    return parser.parse_args()


def collect_all_data(src_root):
    """Collect all images and annotations from 10 activity folders.

    Returns:
        all_images: list of image dicts (with unique ids and updated file_name)
        all_annotations: list of annotation dicts (with unique ids)
        img_src_paths: dict mapping new_img_id -> absolute source image path
    """
    all_images = []
    all_annotations = []
    img_src_paths = {}

    global_img_id = 0
    global_ann_id = 0

    for activity_folder, action_class in sorted(
            ACTIVITY_FOLDER_TO_CLASS.items(), key=lambda x: x[1]):
        activity_path = os.path.join(src_root, activity_folder)
        if not os.path.isdir(activity_path):
            print(f'  WARNING: folder not found: {activity_path}')
            continue

        subfolders = sorted([
            d for d in os.listdir(activity_path)
            if os.path.isdir(os.path.join(activity_path, d))
        ])

        for sub in subfolders:
            ann_file = os.path.join(
                activity_path, sub, 'annotations',
                'merged_person_bbox_keypoints.json')
            img_dir = os.path.join(activity_path, sub, 'stitched_png')

            if not os.path.exists(ann_file):
                continue

            with open(ann_file) as f:
                data = json.load(f)

            # Build local image_id -> new image_id mapping
            old_to_new_img = {}

            for img_info in data.get('images', []):
                old_img_id = img_info['id']
                global_img_id += 1
                new_img_id = global_img_id

                old_to_new_img[old_img_id] = new_img_id

                # Unique file name: activity_sub_originalname
                orig_name = img_info['file_name']
                # e.g. "1A-7_AAG_frame_0000.png"
                safe_activity = activity_folder.replace(' ', '_')
                unique_name = f'{safe_activity}_{sub}_{orig_name}'

                new_img = {
                    'id': new_img_id,
                    'width': img_info['width'],
                    'height': img_info['height'],
                    'file_name': unique_name,
                    'activity_folder': activity_folder,
                    'person_id': sub,
                    'action_class': action_class,
                }
                all_images.append(new_img)

                # Source image path
                src_img = os.path.join(img_dir, orig_name)
                img_src_paths[new_img_id] = src_img

            for ann in data.get('annotations', []):
                old_img_id = ann['image_id']
                if old_img_id not in old_to_new_img:
                    continue

                # Skip non-human categories (e.g. category_id=9 "bbox")
                if ann.get('category_id', 1) != 1:
                    continue

                global_ann_id += 1
                new_ann = copy.deepcopy(ann)
                new_ann['id'] = global_ann_id
                new_ann['image_id'] = old_to_new_img[old_img_id]

                # Fix attributes
                attrs = new_ann.get('attributes', {})
                if not isinstance(attrs, dict):
                    attrs = {}
                attrs['action_class'] = action_class
                attrs['Activity'] = ACTION_NAMES[action_class]
                attrs['falling'] = action_class in FALLING_CLASSES
                new_ann['attributes'] = attrs

                all_annotations.append(new_ann)

    return all_images, all_annotations, img_src_paths


def split_by_subfolder(all_images, all_annotations, train_ratio, seed):
    """Split data by person_id (subfolder) with stratified sampling.

    Ensures every action class has at least 1 subfolder in val.
    Within each class, shuffles subfolders and picks ~20% for val.

    Returns:
        train_images, train_annotations, val_images, val_annotations
    """
    random.seed(seed)

    # Group images by (activity_folder, person_id)
    groups = defaultdict(list)
    for img in all_images:
        key = (img['activity_folder'], img['person_id'])
        groups[key].append(img['id'])

    # Group subfolders by action_class
    class_to_subs = defaultdict(list)
    for (activity_folder, person_id) in groups.keys():
        action_class = ACTIVITY_FOLDER_TO_CLASS[activity_folder]
        class_to_subs[action_class].append((activity_folder, person_id))

    # Stratified split: per class, at least 1 subfolder goes to val
    train_img_ids = set()
    val_img_ids = set()

    for cls_id in sorted(class_to_subs.keys()):
        subs = class_to_subs[cls_id]
        random.shuffle(subs)

        # At least 1 for val, rest follows train_ratio
        n_val = max(1, int(len(subs) * (1 - train_ratio)))
        val_subs = subs[:n_val]
        train_subs = subs[n_val:]

        for key in train_subs:
            train_img_ids.update(groups[key])
        for key in val_subs:
            val_img_ids.update(groups[key])

    # Build annotation lookup by image_id
    ann_by_img = defaultdict(list)
    for ann in all_annotations:
        ann_by_img[ann['image_id']].append(ann)

    train_images, val_images = [], []
    train_annotations, val_annotations = [], []

    for img in all_images:
        if img['id'] in val_img_ids:
            val_images.append(img)
            val_annotations.extend(ann_by_img.get(img['id'], []))
        else:
            train_images.append(img)
            train_annotations.extend(ann_by_img.get(img['id'], []))

    return train_images, train_annotations, val_images, val_annotations


def make_coco_dict(images, annotations):
    """Create a COCO format dictionary."""
    return {
        'images': images,
        'annotations': annotations,
        'categories': CATEGORIES,
    }


def link_or_copy_images(images, img_src_paths, dst_dir, use_symlink=True):
    """Create symlinks or copy images to destination."""
    os.makedirs(dst_dir, exist_ok=True)
    missing = 0
    for img in images:
        src = img_src_paths.get(img['id'])
        if src is None or not os.path.exists(src):
            missing += 1
            continue
        dst = os.path.join(dst_dir, img['file_name'])
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        if use_symlink:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
    if missing > 0:
        print(f'  WARNING: {missing} source images not found')


def print_stats(name, images, annotations):
    """Print dataset statistics."""
    print(f'\n  {name}:')
    print(f'    Images: {len(images)}, Annotations: {len(annotations)}')

    # Per-class distribution
    class_counts = defaultdict(int)
    for ann in annotations:
        cls = ann.get('attributes', {}).get('action_class', -1)
        class_counts[cls] += 1

    falling_total = 0
    not_falling_total = 0
    for cls_id in sorted(class_counts.keys()):
        name_str = ACTION_NAMES[cls_id] if 0 <= cls_id < len(ACTION_NAMES) else '???'
        is_fall = cls_id in FALLING_CLASSES
        marker = ' [FALL]' if is_fall else ''
        print(f'    Class {cls_id}: {class_counts[cls_id]:5d}  {name_str}{marker}')
        if is_fall:
            falling_total += class_counts[cls_id]
        else:
            not_falling_total += class_counts[cls_id]

    print(f'    --- Not-falling: {not_falling_total}, Falling: {falling_total}')


def add_kpt_sequences(all_images, all_annotations, seq_len=8):
    """Add temporal keypoint sequences with velocity to each annotation.

    For each annotation, looks back seq_len-1 frames in the same subfolder
    to build a T-length normalized keypoint sequence. Keypoints are normalized
    to [0,1] relative to each frame's bbox.

    Per-keypoint features (kpt_dim=5):
        x, y, vis, dx, dy
    where dx/dy is the inter-frame velocity (difference from previous frame).
    Padded frames (copies of the first frame) have dx=dy=0.

    Args:
        all_images: list of image dicts (must have activity_folder, person_id).
        all_annotations: list of annotation dicts (modified in-place).
        seq_len: temporal window size (default: 8).
    """
    K = 7
    K3 = K * 3  # 21 (intermediate: x, y, vis)
    KD = K * 5  # 35 (final: x, y, vis, dx, dy)

    # image_id -> list of annotations
    ann_by_img = defaultdict(list)
    for ann in all_annotations:
        ann_by_img[ann['image_id']].append(ann)

    # Group images by (activity_folder, person_id), sort by file_name
    groups = defaultdict(list)
    for img in all_images:
        key = (img['activity_folder'], img['person_id'])
        groups[key].append(img)

    for key in groups:
        groups[key].sort(key=lambda x: x['file_name'])

    def normalize_kpts(ann):
        """Normalize keypoints relative to bbox -> list of K*3 floats."""
        kpts = ann.get('keypoints', [])
        bbox = ann.get('bbox', [0, 0, 1, 1])  # [x, y, w, h]
        bx, by, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
        if bw <= 0 or bh <= 0:
            return [0.0] * K3

        feat = []
        for k in range(K):
            if k * 3 + 2 < len(kpts):
                x, y, v = kpts[k * 3], kpts[k * 3 + 1], kpts[k * 3 + 2]
                nx = max(0.0, min(1.0, (x - bx) / bw))
                ny = max(0.0, min(1.0, (y - by) / bh))
                vis = 1.0 if v >= 2 else (0.5 if v == 1 else 0.0)
            else:
                nx, ny, vis = 0.0, 0.0, 0.0
            feat.extend([round(nx, 4), round(ny, 4), round(vis, 1)])
        return feat

    def add_velocity(seq_k3):
        """Convert K*3 sequence to K*5 by adding inter-frame velocity (dx, dy).

        Args:
            seq_k3: list of T lists, each K*3 floats (x, y, vis).

        Returns:
            list of T lists, each K*5 floats (x, y, vis, dx, dy).
        """
        seq_k5 = []
        for t in range(len(seq_k3)):
            curr = seq_k3[t]
            prev = seq_k3[t - 1] if t > 0 else curr
            feat5 = []
            for k in range(K):
                x = curr[k * 3]
                y = curr[k * 3 + 1]
                vis = curr[k * 3 + 2]
                dx = round(x - prev[k * 3], 4)
                dy = round(y - prev[k * 3 + 1], 4)
                feat5.extend([x, y, vis, dx, dy])
            seq_k5.append(feat5)
        return seq_k5

    count = 0
    for key, imgs in groups.items():
        # Pre-compute normalized keypoints (K*3) for each frame in this subfolder
        frame_feats = []
        for img in imgs:
            anns = ann_by_img.get(img['id'], [])
            if len(anns) > 0:
                frame_feats.append(normalize_kpts(anns[0]))
            else:
                frame_feats.append([0.0] * K3)

        # For each annotation, build kpt_sequence with velocity
        for frame_idx, img in enumerate(imgs):
            anns = ann_by_img.get(img['id'], [])
            for ann in anns:
                # Sequence: last seq_len frames ending at frame_idx (K*3)
                start = max(0, frame_idx - seq_len + 1)
                seq_k3 = [frame_feats[t] for t in range(start, frame_idx + 1)]

                # Pad from front by repeating earliest frame
                while len(seq_k3) < seq_len:
                    seq_k3.insert(0, seq_k3[0])

                # Add velocity features: K*3 -> K*5
                seq_k5 = add_velocity(seq_k3)

                ann['attributes']['kpt_sequence'] = seq_k5
                count += 1

    print(f'  Added kpt_sequence (T={seq_len}, dim={KD}) to {count} annotations')


# ---- V10: Bone skeleton connections ----
# head(0), shoulder(1), hand_R(2), hand_L(3), hips(4), foot_R(5), foot_L(6)
BONE_CONNECTIONS = [
    (0, 1),  # head → shoulder (neck)
    (1, 2),  # shoulder → hand_right (right arm)
    (1, 3),  # shoulder → hand_left (left arm)
    (1, 4),  # shoulder → hips (torso)
    (4, 5),  # hips → foot_right (right leg)
    (4, 6),  # hips → foot_left (left leg)
]
NUM_BONES = len(BONE_CONNECTIONS)


def add_bone_sequences(all_images, all_annotations, seq_len=8):
    """Add temporal bone/skeleton feature sequences to each annotation (V10).

    For each annotation, looks back seq_len-1 frames in the same subfolder.
    Computes bone vectors from 6 skeleton connections, then adds temporal
    velocity features.

    Per-bone features (6 dim):
        dx, dy, angle, length, d_angle, d_length
    Total per frame: 6 bones × 6 = 36

    Args:
        all_images: list of image dicts (must have activity_folder, person_id).
        all_annotations: list of annotation dicts (modified in-place).
        seq_len: temporal window size (default: 8).
    """
    import math

    K = 7
    K3 = K * 3  # 21
    BD = NUM_BONES * 6  # 36

    # image_id -> list of annotations
    ann_by_img = defaultdict(list)
    for ann in all_annotations:
        ann_by_img[ann['image_id']].append(ann)

    # Group images by (activity_folder, person_id), sort by file_name
    groups = defaultdict(list)
    for img in all_images:
        key = (img['activity_folder'], img['person_id'])
        groups[key].append(img)

    for key in groups:
        groups[key].sort(key=lambda x: x['file_name'])

    def normalize_kpts(ann):
        """Normalize keypoints relative to bbox -> list of K*3 floats."""
        kpts = ann.get('keypoints', [])
        bbox = ann.get('bbox', [0, 0, 1, 1])
        bx, by, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
        if bw <= 0 or bh <= 0:
            return [0.0] * K3

        feat = []
        for k in range(K):
            if k * 3 + 2 < len(kpts):
                x, y, v = kpts[k * 3], kpts[k * 3 + 1], kpts[k * 3 + 2]
                nx = max(0.0, min(1.0, (x - bx) / bw))
                ny = max(0.0, min(1.0, (y - by) / bh))
                vis = 1.0 if v >= 2 else (0.5 if v == 1 else 0.0)
            else:
                nx, ny, vis = 0.0, 0.0, 0.0
            feat.extend([round(nx, 4), round(ny, 4), round(vis, 1)])
        return feat

    def kpt_to_bones(kpt_k3):
        """Compute bone features from K*3 keypoint features.

        Returns list of NUM_BONES * 4 floats: (dx, dy, angle, length) per bone.
        """
        bones = []
        for start, end in BONE_CONNECTIONS:
            sx, sy = kpt_k3[start * 3], kpt_k3[start * 3 + 1]
            ex, ey = kpt_k3[end * 3], kpt_k3[end * 3 + 1]
            dx = ex - sx
            dy = ey - sy
            length = math.sqrt(dx * dx + dy * dy + 1e-8)
            angle = math.atan2(dx, dy)  # angle from vertical (y-axis)
            bones.extend([round(dx, 4), round(dy, 4),
                          round(angle, 4), round(length, 4)])
        return bones  # NUM_BONES * 4 = 24

    def add_bone_velocity(bone_seq):
        """Add inter-frame velocity (d_angle, d_length) to bone features.

        Input:  T × 24 (6 bones × 4: dx, dy, angle, length)
        Output: T × 36 (6 bones × 6: dx, dy, angle, length, d_angle, d_length)
        """
        seq_out = []
        for t in range(len(bone_seq)):
            curr = bone_seq[t]
            prev = bone_seq[t - 1] if t > 0 else curr
            feat = []
            for b in range(NUM_BONES):
                dx = curr[b * 4]
                dy = curr[b * 4 + 1]
                angle = curr[b * 4 + 2]
                length = curr[b * 4 + 3]
                d_angle = round(angle - prev[b * 4 + 2], 4)
                d_length = round(length - prev[b * 4 + 3], 4)
                feat.extend([dx, dy, angle, length, d_angle, d_length])
            seq_out.append(feat)
        return seq_out  # T × 36

    count = 0
    for key, imgs in groups.items():
        # Pre-compute normalized keypoints (K*3) for each frame
        frame_kpts = []
        for img in imgs:
            anns = ann_by_img.get(img['id'], [])
            if len(anns) > 0:
                frame_kpts.append(normalize_kpts(anns[0]))
            else:
                frame_kpts.append([0.0] * K3)

        # Pre-compute bone features (NUM_BONES*4) for each frame
        frame_bones = [kpt_to_bones(kpt) for kpt in frame_kpts]

        # For each annotation, build bone_sequence with velocity
        for frame_idx, img in enumerate(imgs):
            anns = ann_by_img.get(img['id'], [])
            for ann in anns:
                start = max(0, frame_idx - seq_len + 1)
                seq_bones = [frame_bones[t]
                             for t in range(start, frame_idx + 1)]

                # Pad from front by repeating earliest frame
                while len(seq_bones) < seq_len:
                    seq_bones.insert(0, seq_bones[0])

                # Add velocity: 24 -> 36 per frame
                seq_full = add_bone_velocity(seq_bones)

                ann['attributes']['kpt_sequence'] = seq_full
                count += 1

    print(f'  Added bone_sequence (T={seq_len}, dim={BD}) '
          f'to {count} annotations')


def main():
    args = parse_args()
    print('=' * 60)
    print('Preparing 10-class action recognition COCO dataset')
    print('=' * 60)
    print(f'Source: {args.src}')
    print(f'Destination: {args.dst}')
    print(f'Seed: {args.seed}, Train ratio: {args.train_ratio}')
    print(f'Image mode: {"symlink" if args.symlink else "copy"}')

    # 1. Collect all data
    print('\nCollecting data from 10 activity folders...')
    all_images, all_annotations, img_src_paths = collect_all_data(args.src)
    print(f'  Total: {len(all_images)} images, {len(all_annotations)} annotations')

    # 1.5. Add temporal sequences (V9: keypoint features, V10: bone features)
    if args.seq_len > 0:
        if args.bone:
            print(f'\nAdding bone_sequences (T={args.seq_len}, V10)...')
            add_bone_sequences(all_images, all_annotations,
                               seq_len=args.seq_len)
        else:
            print(f'\nAdding kpt_sequences (T={args.seq_len}, V9)...')
            add_kpt_sequences(all_images, all_annotations,
                              seq_len=args.seq_len)

    # 2. Split train/val
    print('\nSplitting by subfolder...')
    train_imgs, train_anns, val_imgs, val_anns = split_by_subfolder(
        all_images, all_annotations, args.train_ratio, args.seed)

    print_stats('Train', train_imgs, train_anns)
    print_stats('Val', val_imgs, val_anns)

    # 3. Write annotation files
    ann_dir = os.path.join(args.dst, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)

    train_file = os.path.join(ann_dir, 'instances_train.json')
    val_file = os.path.join(ann_dir, 'instances_val.json')

    print(f'\nWriting {train_file}...')
    with open(train_file, 'w') as f:
        json.dump(make_coco_dict(train_imgs, train_anns), f)

    print(f'Writing {val_file}...')
    with open(val_file, 'w') as f:
        json.dump(make_coco_dict(val_imgs, val_anns), f)

    # 4. Link/copy images
    # Use flat image directory (train and val share the same images/ folder)
    img_dir = os.path.join(args.dst, 'images')
    print(f'\nLinking images to {img_dir}...')
    link_or_copy_images(
        all_images, img_src_paths, img_dir, use_symlink=args.symlink)

    print('\nDone!')
    print('=' * 60)


if __name__ == '__main__':
    main()

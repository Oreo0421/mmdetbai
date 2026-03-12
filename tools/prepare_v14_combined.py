#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare combined V14 dataset from SenIR + Synthetic data.

Merges both datasets into a single COCO format with:
  - action_class (0-9) derived from activity/action names
  - kpt_sequence (T=8, dim=35) temporal keypoint features
  - Flat images/ directory with unique file names
  - 80/20 train/val split by person_id

Usage:
    python tools/prepare_v14_combined.py
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

# Synthetic data action name -> action_class mapping
# Action names start with a number prefix (1-10)
SYNTH_ACTION_TO_CLASS = {
    '1_standingStill': 0,
    '1_simple_standingStill': 0,
    '2_simple_walk': 1,
    '3_simple_sit': 2,
    '4_simple_sitGetUp': 3,
    '5_LieDown': 4,
    '5_simple_LieDown': 4,
    '6_lieGetUp': 5,
    '6_simple_lieGetUp': 5,
    '7_simple_fallForward': 6,
    '7_fallFORward_b': 6,
    '7_simple_fallFORward_bb': 6,
    '8_simple_fallBackwardSit_b': 7,
    '8_fallBackward': 7,
    '9_simple_fallWhileTryingSit_b': 8,
    '9_fallWhileSit': 8,
    '10_simple_fallFORward_standup': 9,
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
    parser = argparse.ArgumentParser(
        description='Prepare combined V14 dataset')
    parser.add_argument(
        '--senir-src',
        default='/home/tbai/Desktop/SenIRDatasetProcessed/',
        help='SenIR dataset root')
    parser.add_argument(
        '--synth-src',
        default='/home/tbai/Desktop/systhetic_data/',
        help='Synthetic dataset root')
    parser.add_argument(
        '--dst',
        default='/home/tbai/Desktop/V14dataset/',
        help='Output dataset root')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--seq-len', type=int, default=8)
    return parser.parse_args()


# ===================== SenIR data collection =====================

def collect_senir_data(src_root, id_offset_img=0, id_offset_ann=0):
    """Collect SenIR data from 10 activity folders.

    Returns:
        all_images, all_annotations, img_src_paths
    """
    all_images = []
    all_annotations = []
    img_src_paths = {}

    global_img_id = id_offset_img
    global_ann_id = id_offset_ann

    for activity_folder, action_class in sorted(
            ACTIVITY_FOLDER_TO_CLASS.items(), key=lambda x: x[1]):
        activity_path = os.path.join(src_root, activity_folder)
        if not os.path.isdir(activity_path):
            print(f'  WARNING: SenIR folder not found: {activity_path}')
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

            old_to_new_img = {}

            for img_info in data.get('images', []):
                old_img_id = img_info['id']
                global_img_id += 1
                new_img_id = global_img_id
                old_to_new_img[old_img_id] = new_img_id

                orig_name = img_info['file_name']
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
                    'source': 'senir',
                }
                all_images.append(new_img)

                src_img = os.path.join(img_dir, orig_name)
                img_src_paths[new_img_id] = src_img

            for ann in data.get('annotations', []):
                old_img_id = ann['image_id']
                if old_img_id not in old_to_new_img:
                    continue
                if ann.get('category_id', 1) != 1:
                    continue

                global_ann_id += 1
                new_ann = copy.deepcopy(ann)
                new_ann['id'] = global_ann_id
                new_ann['image_id'] = old_to_new_img[old_img_id]

                attrs = new_ann.get('attributes', {})
                if not isinstance(attrs, dict):
                    attrs = {}
                attrs['action_class'] = action_class
                attrs['Activity'] = ACTION_NAMES[action_class]
                attrs['falling'] = action_class in FALLING_CLASSES
                new_ann['attributes'] = attrs

                all_annotations.append(new_ann)

    print(f'  SenIR: {len(all_images)} images, '
          f'{len(all_annotations)} annotations')
    return all_images, all_annotations, img_src_paths, global_img_id, global_ann_id


# ===================== Synthetic data collection =====================

def collect_synth_data(src_root, id_offset_img=0, id_offset_ann=0):
    """Collect synthetic data from annotations_all.json.

    Fixes file_name mismatch by adding scene_avatar_ prefix.
    Derives action_class from action string.

    Returns:
        all_images, all_annotations, img_src_paths
    """
    all_images = []
    all_annotations = []
    img_src_paths = {}

    ann_file = os.path.join(src_root, 'annotations_all.json')
    if not os.path.exists(ann_file):
        print(f'  WARNING: {ann_file} not found')
        return all_images, all_annotations, img_src_paths, id_offset_img, id_offset_ann

    with open(ann_file) as f:
        data = json.load(f)

    img_dir = os.path.join(src_root, 'images')

    # Build old_id -> annotation mapping
    ann_by_img = defaultdict(list)
    for ann in data['annotations']:
        ann_by_img[ann['image_id']].append(ann)

    global_img_id = id_offset_img
    global_ann_id = id_offset_ann

    # Build actual filename lookup from images dir
    actual_files = set(os.listdir(img_dir)) if os.path.isdir(img_dir) else set()

    for img_info in data['images']:
        old_img_id = img_info['id']
        orig_name = img_info['file_name']

        # Get scene and avatar from the annotation attributes
        anns = ann_by_img.get(old_img_id, [])
        if not anns:
            continue

        ann = anns[0]
        attrs = ann.get('attributes', {})
        scene = attrs.get('scene', '')
        avatar = attrs.get('avatar', '')
        action = attrs.get('action', '')

        # Fix file_name: actual files have {scene}_{avatar}_{orig_name} prefix
        fixed_name = f'{scene}_{avatar}_{orig_name}'

        if fixed_name not in actual_files:
            # Try to find a match
            candidates = [f for f in actual_files if orig_name in f]
            if candidates:
                fixed_name = candidates[0]
            else:
                continue

        # Derive action_class
        action_class = SYNTH_ACTION_TO_CLASS.get(action, -1)
        if action_class == -1:
            # Fallback: try matching by number prefix
            try:
                prefix = int(action.split('_')[0])
                action_class = prefix - 1  # 1-based to 0-based
                if action_class > 9:
                    action_class = 9
            except (ValueError, IndexError):
                print(f'  WARNING: unknown action "{action}", skipping')
                continue

        global_img_id += 1
        new_img_id = global_img_id

        # Create unique file name with synth_ prefix to avoid collisions
        unique_name = f'synth_{fixed_name}'

        # Map activity_folder for grouping
        # Use scene + avatar + session timestamp as person_id
        # so each rendering session has its own temporal sequence
        session = img_info.get('person_id', '')  # timestamp from annotation
        person_id = f'synth_{scene}_{avatar}_{session}'
        activity_folder = f'synth_class_{action_class}'

        new_img = {
            'id': new_img_id,
            'width': img_info['width'],
            'height': img_info['height'],
            'file_name': unique_name,
            'activity_folder': activity_folder,
            'person_id': person_id,
            'action_class': action_class,
            'source': 'synth',
        }
        all_images.append(new_img)

        src_img = os.path.join(img_dir, fixed_name)
        img_src_paths[new_img_id] = src_img

        # Process annotations
        for ann in anns:
            if ann.get('category_id', 1) != 1:
                continue

            global_ann_id += 1
            new_ann = copy.deepcopy(ann)
            new_ann['id'] = global_ann_id
            new_ann['image_id'] = new_img_id

            new_attrs = new_ann.get('attributes', {})
            if not isinstance(new_attrs, dict):
                new_attrs = {}
            new_attrs['action_class'] = action_class
            new_attrs['Activity'] = ACTION_NAMES[action_class]
            new_attrs['falling'] = action_class in FALLING_CLASSES
            new_ann['attributes'] = new_attrs

            # Remove segmentation to save space
            new_ann.pop('segmentation', None)

            all_annotations.append(new_ann)

    print(f'  Synthetic: {len(all_images)} images, '
          f'{len(all_annotations)} annotations')
    return all_images, all_annotations, img_src_paths, global_img_id, global_ann_id


# ===================== Temporal keypoint sequences =====================

def add_kpt_sequences(all_images, all_annotations, seq_len=8):
    """Add temporal keypoint sequences with velocity."""
    K = 7
    K3 = K * 3
    KD = K * 5

    ann_by_img = defaultdict(list)
    for ann in all_annotations:
        ann_by_img[ann['image_id']].append(ann)

    groups = defaultdict(list)
    for img in all_images:
        key = (img['activity_folder'], img['person_id'])
        groups[key].append(img)

    for key in groups:
        groups[key].sort(key=lambda x: x['file_name'])

    def normalize_kpts(ann):
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

    def add_velocity(seq_k3):
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
        frame_feats = []
        for img in imgs:
            anns = ann_by_img.get(img['id'], [])
            if len(anns) > 0:
                frame_feats.append(normalize_kpts(anns[0]))
            else:
                frame_feats.append([0.0] * K3)

        for frame_idx, img in enumerate(imgs):
            anns = ann_by_img.get(img['id'], [])
            for ann in anns:
                start = max(0, frame_idx - seq_len + 1)
                seq_k3 = [frame_feats[t] for t in range(start, frame_idx + 1)]
                while len(seq_k3) < seq_len:
                    seq_k3.insert(0, seq_k3[0])
                seq_k5 = add_velocity(seq_k3)
                ann['attributes']['kpt_sequence'] = seq_k5
                count += 1

    print(f'  Added kpt_sequence (T={seq_len}, dim={KD}) to {count} annotations')


# ===================== Split =====================

def split_by_person(all_images, all_annotations, train_ratio, seed):
    """Split by person_id with stratified sampling per action class."""
    random.seed(seed)

    groups = defaultdict(list)
    for img in all_images:
        key = (img['activity_folder'], img['person_id'])
        groups[key].append(img['id'])

    class_to_subs = defaultdict(list)
    for img in all_images:
        key = (img['activity_folder'], img['person_id'])
        action_class = img['action_class']
        if (action_class, key) not in [(a, k) for a, k in
                                        [(img2['action_class'],
                                          (img2['activity_folder'], img2['person_id']))
                                         for img2 in all_images
                                         if (img2['activity_folder'], img2['person_id']) in
                                         class_to_subs.get(img2['action_class'], [])]]:
            pass

    # Simpler approach: group by action_class -> set of (activity_folder, person_id)
    class_to_subs = defaultdict(set)
    for img in all_images:
        key = (img['activity_folder'], img['person_id'])
        class_to_subs[img['action_class']].add(key)

    train_img_ids = set()
    val_img_ids = set()

    for cls_id in sorted(class_to_subs.keys()):
        subs = list(class_to_subs[cls_id])
        random.shuffle(subs)
        n_val = max(1, int(len(subs) * (1 - train_ratio)))
        val_subs = subs[:n_val]
        train_subs = subs[n_val:]

        for key in train_subs:
            train_img_ids.update(groups[key])
        for key in val_subs:
            val_img_ids.update(groups[key])

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


# ===================== Utilities =====================

def make_coco_dict(images, annotations):
    return {
        'images': images,
        'annotations': annotations,
        'categories': CATEGORIES,
    }


def link_or_copy_images(images, img_src_paths, dst_dir):
    """Create symlinks for images."""
    os.makedirs(dst_dir, exist_ok=True)
    missing = 0
    linked = 0
    for img in images:
        src = img_src_paths.get(img['id'])
        if src is None or not os.path.exists(src):
            missing += 1
            continue
        dst = os.path.join(dst_dir, img['file_name'])
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        linked += 1
    if missing > 0:
        print(f'  WARNING: {missing} source images not found')
    print(f'  Linked {linked} images to {dst_dir}')


def print_stats(name, images, annotations):
    print(f'\n  {name}:')
    print(f'    Images: {len(images)}, Annotations: {len(annotations)}')

    class_counts = defaultdict(int)
    source_counts = defaultdict(int)
    for img in images:
        source_counts[img.get('source', 'unknown')] += 1

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
    print(f'    --- Sources: {dict(source_counts)}')


# ===================== Main =====================

def main():
    args = parse_args()
    print('=' * 60)
    print('Preparing combined V14 dataset (SenIR + Synthetic)')
    print('=' * 60)
    print(f'SenIR source: {args.senir_src}')
    print(f'Synthetic source: {args.synth_src}')
    print(f'Destination: {args.dst}')

    # 1. Collect SenIR data
    print('\n[1/5] Collecting SenIR data...')
    senir_imgs, senir_anns, senir_paths, max_img_id, max_ann_id = \
        collect_senir_data(args.senir_src)

    # 2. Collect Synthetic data (IDs continue from SenIR)
    print('\n[2/5] Collecting Synthetic data...')
    synth_imgs, synth_anns, synth_paths, _, _ = \
        collect_synth_data(args.synth_src, max_img_id, max_ann_id)

    # 3. Merge
    all_images = senir_imgs + synth_imgs
    all_annotations = senir_anns + synth_anns
    all_paths = {**senir_paths, **synth_paths}
    print(f'\n  Combined: {len(all_images)} images, '
          f'{len(all_annotations)} annotations')

    # 4. Add temporal sequences
    print(f'\n[3/5] Adding kpt_sequences (T={args.seq_len})...')
    add_kpt_sequences(all_images, all_annotations, seq_len=args.seq_len)

    # 5. Split train/val
    print('\n[4/5] Splitting train/val...')
    train_imgs, train_anns, val_imgs, val_anns = split_by_person(
        all_images, all_annotations, args.train_ratio, args.seed)

    print_stats('Train', train_imgs, train_anns)
    print_stats('Val', val_imgs, val_anns)

    # 6. Write output
    print('\n[5/5] Writing dataset...')
    ann_dir = os.path.join(args.dst, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)

    train_file = os.path.join(ann_dir, 'instances_train.json')
    val_file = os.path.join(ann_dir, 'instances_val.json')

    print(f'  Writing {train_file}...')
    with open(train_file, 'w') as f:
        json.dump(make_coco_dict(train_imgs, train_anns), f)

    print(f'  Writing {val_file}...')
    with open(val_file, 'w') as f:
        json.dump(make_coco_dict(val_imgs, val_anns), f)

    # Link images
    img_dir = os.path.join(args.dst, 'images')
    link_or_copy_images(all_images, all_paths, img_dir)

    print('\nDone!')
    print('=' * 60)


if __name__ == '__main__':
    main()

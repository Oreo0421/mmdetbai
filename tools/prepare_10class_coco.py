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
    """Split data by person_id (subfolder) to avoid data leakage.

    All frames from the same subfolder go into the same split.

    Returns:
        train_images, train_annotations, val_images, val_annotations
    """
    random.seed(seed)

    # Group images by (activity_folder, person_id)
    groups = defaultdict(list)
    for img in all_images:
        key = (img['activity_folder'], img['person_id'])
        groups[key].append(img['id'])

    # Shuffle groups and split
    group_keys = sorted(groups.keys())
    random.shuffle(group_keys)

    total_imgs = sum(len(groups[k]) for k in group_keys)
    target_train = int(total_imgs * train_ratio)

    train_img_ids = set()
    count = 0
    for key in group_keys:
        if count >= target_train:
            break
        train_img_ids.update(groups[key])
        count += len(groups[key])

    # Build annotation lookup by image_id
    ann_by_img = defaultdict(list)
    for ann in all_annotations:
        ann_by_img[ann['image_id']].append(ann)

    train_images, val_images = [], []
    train_annotations, val_annotations = [], []

    for img in all_images:
        if img['id'] in train_img_ids:
            train_images.append(img)
            train_annotations.extend(ann_by_img.get(img['id'], []))
        else:
            val_images.append(img)
            val_annotations.extend(ann_by_img.get(img['id'], []))

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

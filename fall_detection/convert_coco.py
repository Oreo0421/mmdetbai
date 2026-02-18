"""Convert COCO-format sensir annotations to fall_detection dataset format.

Reads the per-frame COCO annotations, groups clips by person_id,
concatenates into longer mixed-activity videos, and generates
interval annotations for fall/sit/lie.

Usage:
    python -m fall_detection.convert_coco \
        --coco_root /home/tbai/Desktop/sensir_coco_v12 \
        --output_dir ./fall_dataset \
        --mode per_person

Modes:
    per_person: concatenate all clips per person → one long video per person
    per_clip:   each activity clip → one short video (simpler, less training signal)
"""
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from .utils import save_json

# Action class mapping
# 0: Standing still → upright
# 1: Walking → upright
# 2: Sitting down → sit interval
# 3: Standing up → upright
# 4: Lying down → lie interval (voluntary)
# 5: Getting up → upright
# 6-9: Falling variants → fall interval
ACTION_TO_INTERVAL = {
    0: None,      # standing still → upright (default)
    1: None,      # walking → upright
    2: 'sit',     # sitting down → sit
    3: None,      # standing up → upright
    4: 'lie',     # lying down → voluntary lie
    5: None,      # getting up → upright
    6: 'fall',    # falling while walking
    7: 'fall',    # falling while standing
    8: 'fall',    # falling while sitting
    9: 'fall',    # falling while standing up
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Convert COCO sensir to fall_detection format')
    p.add_argument('--coco_root', type=str, required=True,
                   help='Path to sensir COCO dataset (e.g. sensir_coco_v12)')
    p.add_argument('--output_dir', type=str, default='./fall_dataset')
    p.add_argument('--mode', choices=['per_person', 'per_clip'], default='per_person',
                   help='per_person: concat clips per person; per_clip: each clip separate')
    p.add_argument('--splits', type=str, default='train,val',
                   help='Which splits to convert (comma-separated)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_coco_data(coco_root: str, split: str) -> Tuple[dict, dict, dict]:
    """Load COCO annotations and organize by video clip."""
    ann_path = Path(coco_root) / 'annotations' / f'instances_{split}.json'
    with open(ann_path) as f:
        coco = json.load(f)

    img_map = {img['id']: img for img in coco['images']}

    # Group annotations by video clip
    clips = defaultdict(list)  # vid_id → [(frame_num, keypoints, action_class, img_w, img_h)]

    for ann in coco['annotations']:
        img = img_map[ann['image_id']]
        fn = img['file_name']
        parts = fn.rsplit('_frame_', 1)
        if len(parts) != 2:
            continue
        vid_id = parts[0]
        frame_num = int(parts[1].replace('.png', ''))

        # Extract keypoints: 21 values = 7 joints × (x, y, visibility_flag)
        raw_kpts = ann['keypoints']
        action_class = ann['attributes'].get('action_class', -1)
        img_w = img['width']
        img_h = img['height']

        clips[vid_id].append((frame_num, raw_kpts, action_class, img_w, img_h))

    # Sort frames within each clip
    for vid_id in clips:
        clips[vid_id].sort(key=lambda x: x[0])

    return clips, img_map, coco


def extract_person_id(vid_id: str) -> str:
    """Extract person ID from video name like '1A-7_Falling_while_walking_AAA'."""
    parts = vid_id.split('_')
    return parts[-1]  # last part is person ID


def clip_to_keypoints(clip_data: list) -> Tuple[np.ndarray, int]:
    """Convert clip annotation data to keypoint array.

    Returns:
        kpts: (T, 7, 3) array with (norm_x, norm_y, score).
        action_class: the action class for this clip.
    """
    T = len(clip_data)
    kpts = np.zeros((T, 7, 3), dtype=np.float32)
    action_class = clip_data[0][2]  # constant within clip

    for t, (frame_num, raw_kpts, ac, img_w, img_h) in enumerate(clip_data):
        for j in range(7):
            x = raw_kpts[j * 3]
            y = raw_kpts[j * 3 + 1]
            vis = raw_kpts[j * 3 + 2]  # COCO: 0=not labeled, 1=occluded, 2=visible

            # Normalize by image dimensions (image is already a person crop)
            kpts[t, j, 0] = x / max(img_w, 1)
            kpts[t, j, 1] = y / max(img_h, 1)

            # Convert visibility to confidence score
            if vis == 2:
                kpts[t, j, 2] = 1.0
            elif vis == 1:
                kpts[t, j, 2] = 0.5
            else:
                kpts[t, j, 2] = 0.0

    return kpts, action_class


def convert_per_clip(clips: dict, output_dir: Path, fps: int = 5, suffix: str = ''):
    """Convert each clip as a separate video."""
    videos_dir = output_dir / 'videos'
    labels_dir = output_dir / 'labels'
    videos_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)

    for vid_id, clip_data in sorted(clips.items()):
        kpts, action_class = clip_to_keypoints(clip_data)
        T = kpts.shape[0]

        # Create interval based on action class
        intervals = {'fall': [], 'sit': [], 'lie': []}
        interval_type = ACTION_TO_INTERVAL.get(action_class)
        if interval_type is not None:
            intervals[interval_type].append([0, T - 1])
            stats[interval_type] += 1
        else:
            stats['upright'] += 1

        # Save
        safe_name = vid_id.replace(' ', '_').replace('/', '_')
        if suffix:
            safe_name = f'{safe_name}_{suffix}'
        np.savez_compressed(str(videos_dir / f'{safe_name}.npz'), keypoints=kpts)
        save_json({
            'video_id': safe_name,
            'fps': fps,
            'source_video': vid_id,
            'action_class': int(action_class),
            'total_frames': T,
            'intervals': intervals,
        }, str(labels_dir / f'{safe_name}.json'))

    print(f"  Converted {len(clips)} clips: {dict(stats)}")


def convert_per_person(clips: dict, output_dir: Path, fps: int = 5, seed: int = 42,
                       suffix: str = ''):
    """Concatenate all clips per person into longer mixed-activity videos.

    Ordering: upright activities first, then sit/lie, then falls.
    This creates more natural transitions for training.
    """
    videos_dir = output_dir / 'videos'
    labels_dir = output_dir / 'labels'
    videos_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    # Group clips by person
    person_clips = defaultdict(list)  # person_id → [(vid_id, clip_data)]
    for vid_id, clip_data in clips.items():
        pid = extract_person_id(vid_id)
        person_clips[pid].append((vid_id, clip_data))

    stats = defaultdict(int)
    gap_frames = 3  # small gap between clips (0.6s at 5fps)

    for pid in sorted(person_clips.keys()):
        clip_list = person_clips[pid]

        # Sort: upright first, then sit/lie, then falls (with some randomness within)
        def sort_key(item):
            _, cd = item
            ac = cd[0][2]
            if ac in (0, 1, 3, 5):
                return (0, rng.random())  # upright activities
            elif ac in (2, 4):
                return (1, rng.random())  # sit/lie (voluntary)
            else:
                return (2, rng.random())  # falls
        clip_list.sort(key=sort_key)

        # Concatenate
        all_kpts = []
        intervals = {'fall': [], 'sit': [], 'lie': []}
        offset = 0
        clip_info = []

        for vid_id, clip_data in clip_list:
            kpts, action_class = clip_to_keypoints(clip_data)
            T_clip = kpts.shape[0]

            if offset > 0:
                # Insert small gap (repeat last frame with low confidence)
                gap = np.zeros((gap_frames, 7, 3), dtype=np.float32)
                if len(all_kpts) > 0:
                    gap[:] = all_kpts[-1][-1:]  # repeat last frame
                    gap[:, :, 2] *= 0.3  # low confidence during gap
                all_kpts.append(gap)
                offset += gap_frames

            # Record interval
            interval_type = ACTION_TO_INTERVAL.get(action_class)
            clip_start = offset
            clip_end = offset + T_clip - 1

            if interval_type is not None:
                intervals[interval_type].append([int(clip_start), int(clip_end)])
                stats[interval_type] += 1
            else:
                stats['upright'] += 1

            all_kpts.append(kpts)
            clip_info.append({
                'source': vid_id,
                'action_class': int(action_class),
                'start_frame': int(clip_start),
                'end_frame': int(clip_end),
            })
            offset += T_clip

        # Stack
        full_kpts = np.concatenate(all_kpts, axis=0)  # (T_total, 7, 3)
        T_total = full_kpts.shape[0]

        # Save
        video_name = f'person_{pid}' if not suffix else f'person_{pid}_{suffix}'
        np.savez_compressed(str(videos_dir / f'{video_name}.npz'), keypoints=full_kpts)
        save_json({
            'video_id': video_name,
            'fps': fps,
            'person_id': pid,
            'total_frames': T_total,
            'duration_sec': T_total / fps,
            'intervals': intervals,
            'clips': clip_info,
        }, str(labels_dir / f'{video_name}.json'))

        n_fall = len(intervals['fall'])
        n_sit = len(intervals['sit'])
        n_lie = len(intervals['lie'])
        print(f"  {video_name}: {T_total} frames ({T_total/fps:.1f}s), "
              f"{len(clip_list)} clips, falls={n_fall} sits={n_sit} lies={n_lie}")

    print(f"\n  Total: {len(person_clips)} person videos, stats: {dict(stats)}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    for split in args.splits.split(','):
        split = split.strip()
        print(f"\n=== Processing {split} split ===")

        clips, _, _ = load_coco_data(args.coco_root, split)
        print(f"  Found {len(clips)} clips")

        split_dir = output_dir  # single output dir (train/val split done at training time)

        if args.mode == 'per_person':
            convert_per_person(clips, split_dir, fps=5, seed=args.seed, suffix=split)
        else:
            convert_per_clip(clips, split_dir, fps=5, suffix=split)

    print(f"\n=== Dataset saved to {output_dir} ===")
    print(f"  videos/  (.npz files with keypoints)")
    print(f"  labels/  (.json files with intervals)")
    print(f"\nTo train:")
    print(f"  python -m fall_detection.train --data_root {output_dir} --fps 5 --window_sec 2.0 --stride_sec 0.2 --epochs 80")


if __name__ == '__main__':
    main()

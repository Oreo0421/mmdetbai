"""Generate synthetic dataset for debugging and testing.

Creates fake keypoint sequences with known fall/sit/lie events.
Useful for verifying the pipeline end-to-end before using real data.

Usage:
    python -m fall_detection.generate_synthetic --output_dir ./synthetic_dataset --num_videos 20
"""
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

from .utils import save_json, NUM_JOINTS


# ─── Canonical poses (image coords, y increases downward) ───
# Joint order: head, shoulder, hand_R, hand_L, hips, foot_R, foot_L

POSE_UPRIGHT = np.array([
    [0.50, 0.10],  # head
    [0.50, 0.25],  # shoulder
    [0.35, 0.40],  # hand_R
    [0.65, 0.40],  # hand_L
    [0.50, 0.50],  # hips
    [0.40, 0.90],  # foot_R
    [0.60, 0.90],  # foot_L
], dtype=np.float32)

POSE_SIT = np.array([
    [0.50, 0.30],  # head
    [0.50, 0.40],  # shoulder
    [0.35, 0.50],  # hand_R
    [0.65, 0.50],  # hand_L
    [0.50, 0.60],  # hips
    [0.40, 0.85],  # foot_R
    [0.60, 0.85],  # foot_L
], dtype=np.float32)

POSE_LIE = np.array([
    [0.15, 0.70],  # head
    [0.25, 0.70],  # shoulder
    [0.18, 0.65],  # hand_R
    [0.30, 0.75],  # hand_L
    [0.50, 0.70],  # hips
    [0.75, 0.72],  # foot_R
    [0.85, 0.72],  # foot_L
], dtype=np.float32)


def add_noise(pose: np.ndarray, noise_std: float = 0.02,
              rng: np.random.RandomState = None) -> np.ndarray:
    """Add Gaussian noise to a pose."""
    if rng is None:
        rng = np.random.RandomState()
    return pose + rng.randn(*pose.shape).astype(np.float32) * noise_std


def interpolate_poses(pose_a: np.ndarray, pose_b: np.ndarray,
                      n_frames: int) -> np.ndarray:
    """Linearly interpolate between two poses over n_frames."""
    t = np.linspace(0, 1, n_frames).reshape(-1, 1, 1).astype(np.float32)
    return pose_a[None] * (1 - t) + pose_b[None] * t


def generate_confidence(n_frames: int, n_joints: int,
                        base_conf: float = 0.85,
                        noise_std: float = 0.1,
                        dropout_prob: float = 0.05,
                        rng: np.random.RandomState = None) -> np.ndarray:
    """Generate realistic confidence scores with occasional dropouts."""
    if rng is None:
        rng = np.random.RandomState()
    conf = base_conf + rng.randn(n_frames, n_joints).astype(np.float32) * noise_std
    # Random dropouts
    dropout = rng.rand(n_frames, n_joints) < dropout_prob
    conf[dropout] = rng.rand(dropout.sum()).astype(np.float32) * 0.15
    return np.clip(conf, 0.0, 1.0)


def generate_video(
    fps: int = 25,
    duration_sec: float = 30.0,
    rng: np.random.RandomState = None,
) -> Tuple[np.ndarray, dict]:
    """Generate a single synthetic video with random events.

    Returns:
        kpts: (T, 7, 3) keypoint array.
        label_data: dict with fps, intervals.
    """
    if rng is None:
        rng = np.random.RandomState()

    T = int(fps * duration_sec)
    kpts = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)

    # Plan events
    intervals = {'fall': [], 'sit': [], 'lie': []}
    events = _plan_events(T, fps, rng)

    # Generate keypoint sequence
    t = 0
    for event_type, start, end in events:
        # Fill gap with upright pose
        if t < start:
            gap = start - t
            for frame in range(t, start):
                kpts[frame, :, :2] = add_noise(POSE_UPRIGHT, 0.015, rng)
            t = start

        # Generate event
        duration = end - start
        if event_type == 'stand':
            for frame in range(start, end):
                kpts[frame, :, :2] = add_noise(POSE_UPRIGHT, 0.015, rng)

        elif event_type == 'sit':
            trans = min(int(fps * 1.5), duration // 3)
            # Transition down
            poses_down = interpolate_poses(POSE_UPRIGHT, POSE_SIT, trans)
            for i in range(trans):
                kpts[start + i, :, :2] = add_noise(poses_down[i], 0.015, rng)
            # Stay sitting
            for frame in range(start + trans, end):
                kpts[frame, :, :2] = add_noise(POSE_SIT, 0.015, rng)
            intervals['sit'].append([int(start), int(end - 1)])

        elif event_type == 'lie_voluntary':
            # Slow transition to lying (over ~2 seconds)
            trans = min(int(fps * 2.0), duration // 2)
            poses_down = interpolate_poses(POSE_UPRIGHT, POSE_LIE, trans)
            for i in range(trans):
                kpts[start + i, :, :2] = add_noise(poses_down[i], 0.015, rng)
            for frame in range(start + trans, end):
                kpts[frame, :, :2] = add_noise(POSE_LIE, 0.015, rng)
            intervals['lie'].append([int(start), int(end - 1)])

        elif event_type == 'fall':
            # RAPID transition (0.3-0.5 seconds) → then lying
            fall_dur = int(fps * rng.uniform(0.3, 0.5))
            fall_dur = min(fall_dur, duration - 1)
            poses_fall = interpolate_poses(POSE_UPRIGHT, POSE_LIE, fall_dur)
            for i in range(fall_dur):
                kpts[start + i, :, :2] = add_noise(poses_fall[i], 0.025, rng)
            for frame in range(start + fall_dur, end):
                kpts[frame, :, :2] = add_noise(POSE_LIE, 0.02, rng)
            intervals['fall'].append([int(start), int(end - 1)])

        t = end

    # Fill remaining with upright
    for frame in range(t, T):
        kpts[frame, :, :2] = add_noise(POSE_UPRIGHT, 0.015, rng)

    # Generate confidence scores
    kpts[:, :, 2] = generate_confidence(T, NUM_JOINTS, rng=rng)

    label_data = {
        'fps': fps,
        'intervals': intervals,
    }
    return kpts, label_data


def _plan_events(T: int, fps: int,
                 rng: np.random.RandomState) -> List[Tuple[str, int, int]]:
    """Plan a sequence of events for a video."""
    events = []
    t = int(fps * 2)  # start with some standing
    min_gap = int(fps * 2)

    while t < T - int(fps * 3):
        # Choose event type
        r = rng.rand()
        if r < 0.25:
            # Fall event
            dur = int(fps * rng.uniform(3.0, 6.0))
            end = min(t + dur, T)
            events.append(('fall', t, end))
        elif r < 0.50:
            # Sit event
            dur = int(fps * rng.uniform(4.0, 8.0))
            end = min(t + dur, T)
            events.append(('sit', t, end))
        elif r < 0.70:
            # Voluntary lie (slow)
            dur = int(fps * rng.uniform(4.0, 8.0))
            end = min(t + dur, T)
            events.append(('lie_voluntary', t, end))
        else:
            # Just standing/walking
            dur = int(fps * rng.uniform(2.0, 5.0))
            end = min(t + dur, T)
            events.append(('stand', t, end))

        t = end + min_gap + rng.randint(0, int(fps * 2))

    return events


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate synthetic dataset')
    p.add_argument('--output_dir', type=str, default='./synthetic_dataset')
    p.add_argument('--num_videos', type=int, default=20)
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--duration_sec', type=float, default=30.0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / 'videos'
    labels_dir = output_dir / 'labels'
    videos_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    total_falls = 0
    total_sits = 0
    total_lies = 0

    for i in range(args.num_videos):
        vid = f'synth_{i:04d}'
        kpts, label_data = generate_video(
            fps=args.fps,
            duration_sec=args.duration_sec,
            rng=rng,
        )
        label_data['video_id'] = vid

        # Save
        np.savez_compressed(str(videos_dir / f'{vid}.npz'), keypoints=kpts)
        save_json(label_data, str(labels_dir / f'{vid}.json'))

        n_fall = len(label_data['intervals']['fall'])
        n_sit = len(label_data['intervals']['sit'])
        n_lie = len(label_data['intervals']['lie'])
        total_falls += n_fall
        total_sits += n_sit
        total_lies += n_lie

        T = kpts.shape[0]
        print(f"  {vid}: T={T} frames, "
              f"falls={n_fall} sits={n_sit} lies={n_lie}")

    print(f"\nGenerated {args.num_videos} videos in {output_dir}")
    print(f"Total events: falls={total_falls} sits={total_sits} lies={total_lies}")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    videos/  ({args.num_videos} .npz files)")
    print(f"    labels/  ({args.num_videos} .json files)")


if __name__ == '__main__':
    main()

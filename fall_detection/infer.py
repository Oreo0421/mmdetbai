"""Inference script: process a single video and output fall detection results.

Usage:
    python -m fall_detection.infer \
        --data_root ./dataset \
        --video_id video_001 \
        --checkpoint work_dirs/fall_detection/best.pth \
        --plot

Outputs:
    - JSON report with per-frame probabilities and detected events
    - Optional matplotlib plot
"""
import argparse
import numpy as np
import torch
from pathlib import Path

from .utils import (set_seed, save_json, load_json, DEFAULT_FPS,
                    FEATURE_DIM, STATE_NAMES, frames_from_sec)
from .preprocess import preprocess_keypoints, extract_features
from .model import FallDetectionTCN
from .eval import run_video_inference, detect_fall_events


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fall detection inference')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--video_id', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output_dir', type=str, default=None,
                   help='Output directory (default: work_dirs/fall_detection/infer)')
    p.add_argument('--fps', type=float, default=None)
    p.add_argument('--window_sec', type=float, default=1.5)
    p.add_argument('--plot', action='store_true', help='Generate matplotlib plots')
    # Event detection
    p.add_argument('--impact_thr', type=float, default=0.6)
    p.add_argument('--lie_thr', type=float, default=0.6)
    p.add_argument('--lie_sustain_sec', type=float, default=1.5)
    p.add_argument('--cooldown_sec', type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir or 'work_dirs/fall_detection/infer')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    channels = [int(c) for c in ckpt_args.get('channels', '64,64,128,128').split(',')]
    model = FallDetectionTCN(
        in_features=FEATURE_DIM,
        channels=channels,
        dropout=ckpt_args.get('dropout', 0.15),
        use_confidence_gate=not ckpt_args.get('no_conf_gate', False),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    # ── Load video keypoints ──
    npz_path = data_root / 'videos' / f'{args.video_id}.npz'
    json_path = data_root / 'labels' / f'{args.video_id}.json'

    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found")
        return

    kpts_raw = np.load(str(npz_path))['keypoints'].astype(np.float32)
    T = kpts_raw.shape[0]
    print(f"Video: {args.video_id}, frames: {T}")

    # Get fps
    fps = args.fps
    if fps is None and json_path.exists():
        label_data = load_json(str(json_path))
        fps = label_data.get('fps', DEFAULT_FPS)
    elif fps is None:
        fps = DEFAULT_FPS
    print(f"FPS: {fps}")

    # ── Preprocess ──
    kpts_clean = preprocess_keypoints(kpts_raw, fps=fps)
    features = extract_features(kpts_clean)  # (T, 35)

    # ── Run inference ──
    window_sec = args.window_sec or ckpt_args.get('window_sec', 1.5)
    state_probs, impact_probs, pred_probs = run_video_inference(
        model, features, fps, device,
        window_sec=window_sec, stride_sec=0.1
    )

    # ── Detect fall events ──
    events = detect_fall_events(
        impact_probs, state_probs, fps,
        impact_thr=args.impact_thr,
        lie_thr=args.lie_thr,
        lie_sustain_sec=args.lie_sustain_sec,
        cooldown_sec=args.cooldown_sec,
    )

    # ── Per-frame state classification ──
    state_classes = state_probs.argmax(axis=1)
    state_names = [STATE_NAMES[int(c)] for c in state_classes]

    # ── Print results ──
    print(f"\nDetected {len(events)} fall event(s):")
    for i, ev in enumerate(events):
        print(f"  Event {i+1}: frame={ev['frame']} time={ev['time']:.2f}s "
              f"confidence={ev['confidence']:.3f} "
              f"impact_prob={ev['impact_prob']:.3f} "
              f"lie_ratio={ev['lie_ratio']:.3f}")

    # ── Save JSON report ──
    report = {
        'video_id': args.video_id,
        'fps': fps,
        'total_frames': T,
        'duration_sec': T / fps,
        'events': events,
        'per_frame': {
            'state_probs': state_probs.tolist(),
            'impact_probs': impact_probs.tolist(),
            'pred_probs': pred_probs.tolist(),
        },
    }
    report_path = output_dir / f'{args.video_id}_report.json'
    save_json(report, str(report_path))
    print(f"\nReport saved to {report_path}")

    # ── Optional plot ──
    if args.plot:
        _plot_results(args.video_id, fps, T, state_probs, impact_probs,
                      pred_probs, events, output_dir, json_path)


def _plot_results(video_id: str, fps: float, T: int,
                  state_probs: np.ndarray, impact_probs: np.ndarray,
                  pred_probs: np.ndarray, events: list,
                  output_dir: Path, json_path: Path):
    """Generate matplotlib visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    times = np.arange(T) / fps

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # 1. State probabilities
    ax = axes[0]
    for i, name in enumerate(STATE_NAMES):
        ax.plot(times, state_probs[:, i], label=name, alpha=0.8)
    ax.set_ylabel('State Probability')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.set_title(f'{video_id} - State Classification')

    # 2. Impact probability
    ax = axes[1]
    ax.plot(times, impact_probs, color='red', alpha=0.8, label='Impact')
    ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    for ev in events:
        ax.axvline(x=ev['time'], color='red', linestyle=':', alpha=0.7)
        ax.annotate(f"FALL\n{ev['confidence']:.2f}",
                    xy=(ev['time'], ev['impact_prob']),
                    fontsize=8, color='red', ha='center')
    ax.set_ylabel('Impact Probability')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.set_title('Impact Detection')

    # 3. Prediction probability
    ax = axes[2]
    ax.plot(times, pred_probs, color='orange', alpha=0.8, label='Fall Prediction')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Prediction Probability')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.set_title('Fall Prediction (1s horizon)')

    # Add GT intervals if available
    if json_path.exists():
        label_data = load_json(str(json_path))
        intervals = label_data.get('intervals', {})
        for start, end in intervals.get('fall', []):
            for a in axes:
                a.axvspan(start / fps, end / fps, color='red',
                          alpha=0.1, label='_')

    plt.tight_layout()
    plot_path = output_dir / f'{video_id}_plot.png'
    plt.savefig(str(plot_path), dpi=120)
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()

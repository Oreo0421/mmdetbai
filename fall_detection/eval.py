"""Evaluation script with window-level and event-level metrics.

Event-level fall detection:
  1. Convert impact probabilities to peaks (threshold th1).
  2. Verify post-condition: lie probability > th2 sustained for tau seconds.
  3. Apply cooldown to avoid duplicates.
  4. Match to ground-truth impact centers with temporal tolerance.

Usage:
    python -m fall_detection.eval --data_root ./dataset --checkpoint work_dirs/fall_detection/best.pth
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from .utils import (set_seed, save_json, load_json, DEFAULT_FPS,
                    DEFAULT_WINDOW_SEC, DEFAULT_STRIDE_SEC, FEATURE_DIM,
                    frames_from_sec, STATE_LIE)
from .dataset import FallDataset
from .model import FallDetectionTCN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate fall detection model')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='work_dirs/fall_detection/eval')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_ratio', type=float, default=0.2)
    # Event detection params
    p.add_argument('--impact_thr', type=float, default=0.6,
                   help='Threshold for impact peak detection')
    p.add_argument('--lie_thr', type=float, default=0.6,
                   help='Threshold for sustained lie verification')
    p.add_argument('--lie_sustain_sec', type=float, default=1.5,
                   help='How long lie must be sustained after impact')
    p.add_argument('--cooldown_sec', type=float, default=2.0,
                   help='Cooldown between detected events')
    p.add_argument('--tolerance_sec', type=float, default=0.5,
                   help='Temporal tolerance for matching GT events')
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────
# Window-level metrics
# ────────────────────────────────────────────────────────────────────

def compute_window_metrics(all_state_preds: np.ndarray,
                           all_state_labels: np.ndarray,
                           all_impact_probs: np.ndarray,
                           all_impact_labels: np.ndarray,
                           all_pred_probs: np.ndarray,
                           all_pred_labels: np.ndarray) -> dict:
    """Compute window-level accuracy, F1, AP."""
    metrics = {}

    # State accuracy and per-class F1
    state_acc = (all_state_preds == all_state_labels).mean()
    metrics['state_accuracy'] = float(state_acc)

    for cls_id, cls_name in enumerate(['upright', 'sit', 'lie']):
        tp = ((all_state_preds == cls_id) & (all_state_labels == cls_id)).sum()
        fp = ((all_state_preds == cls_id) & (all_state_labels != cls_id)).sum()
        fn = ((all_state_preds != cls_id) & (all_state_labels == cls_id)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        metrics[f'state_{cls_name}_f1'] = float(f1)

    # Impact F1 and AP
    metrics['impact_f1'] = float(_binary_f1_np(all_impact_probs, all_impact_labels))
    metrics['impact_ap'] = float(_average_precision(all_impact_probs, all_impact_labels))

    # Prediction F1 and AP
    metrics['pred_f1'] = float(_binary_f1_np(all_pred_probs, all_pred_labels))
    metrics['pred_ap'] = float(_average_precision(all_pred_probs, all_pred_labels))

    return metrics


def _binary_f1_np(probs: np.ndarray, labels: np.ndarray,
                  threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(float)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))


def _average_precision(probs: np.ndarray, labels: np.ndarray) -> float:
    """Simple average precision (area under PR curve)."""
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-probs)
    sorted_labels = labels[order]
    tp_cumsum = np.cumsum(sorted_labels)
    precision_at_k = tp_cumsum / np.arange(1, len(labels) + 1)
    ap = (precision_at_k * sorted_labels).sum() / labels.sum()
    return float(ap)


# ────────────────────────────────────────────────────────────────────
# Event-level fall detection
# ────────────────────────────────────────────────────────────────────

def detect_fall_events(
    impact_probs: np.ndarray,
    state_probs: np.ndarray,
    fps: float,
    impact_thr: float = 0.6,
    lie_thr: float = 0.6,
    lie_sustain_sec: float = 1.5,
    cooldown_sec: float = 2.0,
) -> List[Dict]:
    """Detect fall events from per-frame probabilities.

    Algorithm:
      1. Find frames where impact_prob > impact_thr (peaks).
      2. For each peak, verify: lie_prob > lie_thr sustained for tau seconds.
      3. Apply cooldown to avoid duplicates.

    Args:
        impact_probs: (T,) impact probability per frame.
        state_probs: (T, 3) state probabilities per frame.
        fps: frame rate.
        impact_thr: threshold for impact detection.
        lie_thr: threshold for lie state verification.
        lie_sustain_sec: seconds of sustained lying required.
        cooldown_sec: cooldown between events.

    Returns:
        events: list of dicts with 'frame', 'time', 'confidence'.
    """
    T = len(impact_probs)
    lie_sustain_frames = frames_from_sec(lie_sustain_sec, fps)
    cooldown_frames = frames_from_sec(cooldown_sec, fps)

    # 1. Find impact peaks (local maxima above threshold)
    peaks = []
    for t in range(1, T - 1):
        if (impact_probs[t] >= impact_thr and
                impact_probs[t] >= impact_probs[t - 1] and
                impact_probs[t] >= impact_probs[t + 1]):
            peaks.append(t)

    # Also check edges
    if T > 0 and impact_probs[0] >= impact_thr:
        if T == 1 or impact_probs[0] >= impact_probs[1]:
            peaks.insert(0, 0)
    if T > 1 and impact_probs[-1] >= impact_thr:
        if impact_probs[-1] >= impact_probs[-2]:
            peaks.append(T - 1)

    # Sort by confidence (descending)
    peaks.sort(key=lambda t: impact_probs[t], reverse=True)

    # 2. Verify post-condition and apply cooldown
    events = []
    used_frames = set()

    for t_peak in peaks:
        # Check cooldown
        if any(abs(t_peak - t_used) < cooldown_frames for t_used in used_frames):
            continue

        # Verify sustained lying after peak
        check_start = t_peak
        check_end = min(T, t_peak + lie_sustain_frames)
        if check_end - check_start < max(1, lie_sustain_frames // 2):
            # Not enough frames to verify (near end of video)
            # Still accept if impact is very confident
            if impact_probs[t_peak] < 0.8:
                continue

        lie_probs = state_probs[check_start:check_end, STATE_LIE]
        # Require majority of frames have lie_prob > threshold
        lie_ratio = (lie_probs > lie_thr).mean()
        if lie_ratio < 0.5:
            continue  # Not enough sustained lying → likely voluntary action

        confidence = float(impact_probs[t_peak] * (0.5 + 0.5 * lie_ratio))
        events.append({
            'frame': int(t_peak),
            'time': float(t_peak / fps),
            'confidence': confidence,
            'impact_prob': float(impact_probs[t_peak]),
            'lie_ratio': float(lie_ratio),
        })
        used_frames.add(t_peak)

    # Sort by time
    events.sort(key=lambda e: e['frame'])
    return events


def match_events_to_gt(
    detected: List[Dict],
    gt_impact_centers: List[int],
    fps: float,
    tolerance_sec: float = 0.5,
) -> Tuple[int, int, int]:
    """Match detected events to GT impact centers.

    A detected event matches a GT event if their frame difference
    is within tolerance_sec * fps.

    Returns:
        tp: true positives (matched detections).
        fp: false positives (unmatched detections).
        fn: false negatives (unmatched GT events).
    """
    tolerance_frames = frames_from_sec(tolerance_sec, fps)
    gt_matched = set()
    det_matched = set()

    # Greedy matching: for each GT, find closest unmatched detection
    for gt_idx, gt_frame in enumerate(gt_impact_centers):
        best_dist = float('inf')
        best_det = -1
        for det_idx, det in enumerate(detected):
            if det_idx in det_matched:
                continue
            dist = abs(det['frame'] - gt_frame)
            if dist < best_dist and dist <= tolerance_frames:
                best_dist = dist
                best_det = det_idx

        if best_det >= 0:
            gt_matched.add(gt_idx)
            det_matched.add(best_det)

    tp = len(gt_matched)
    fp = len(detected) - len(det_matched)
    fn = len(gt_impact_centers) - len(gt_matched)
    return tp, fp, fn


def compute_event_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, F1 from TP/FP/FN."""
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        'event_tp': tp,
        'event_fp': fp,
        'event_fn': fn,
        'event_precision': float(precision),
        'event_recall': float(recall),
        'event_f1': float(f1),
    }


# ────────────────────────────────────────────────────────────────────
# Per-frame inference on full video
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_video_inference(model: torch.nn.Module, features: np.ndarray,
                        fps: float, device: torch.device,
                        window_sec: float = 1.5,
                        stride_sec: float = 0.2
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run dense sliding-window inference on a full video.

    Aggregation strategy:
      - state: mean over overlapping windows (sustained property)
      - impact/predict: center-weighted Gaussian, then max across windows
        Each window assigns its probability to a narrow Gaussian around the
        window center, producing sharp peaks for event detection.

    Returns:
        state_probs: (T, 3) per-frame state probabilities.
        impact_probs: (T,) per-frame impact probabilities.
        pred_probs: (T,) per-frame prediction probabilities.
    """
    model.eval()
    T, F = features.shape
    W = frames_from_sec(window_sec, fps)
    S = max(1, frames_from_sec(stride_sec, fps))

    # Accumulators
    state_acc = np.zeros((T, 3), dtype=np.float64)
    state_count = np.zeros(T, dtype=np.float64)
    # For impact/predict: center-weighted accumulation
    impact_acc = np.zeros(T, dtype=np.float64)
    impact_wt = np.zeros(T, dtype=np.float64)
    pred_acc = np.zeros(T, dtype=np.float64)
    pred_wt = np.zeros(T, dtype=np.float64)

    # Gaussian weight centered at window midpoint (sigma = W/6)
    sigma = max(W / 6.0, 1.0)
    gauss = np.exp(-0.5 * ((np.arange(W) - W / 2.0) / sigma) ** 2)
    gauss /= gauss.max()  # normalize peak to 1.0

    # Dense sliding window
    windows = []
    starts = []
    for start in range(0, max(1, T - W + 1), S):
        end = start + W
        if end > T:
            end = T
            start = max(0, end - W)
        feat = features[start:end]
        if feat.shape[0] < W:
            pad = np.zeros((W - feat.shape[0], F), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)
        windows.append(feat)
        starts.append(start)

    if len(windows) == 0:
        return (np.zeros((T, 3)), np.zeros(T), np.zeros(T))

    # Batch inference
    batch_size = 128
    for i in range(0, len(windows), batch_size):
        batch = np.stack(windows[i:i + batch_size])
        batch_t = torch.from_numpy(batch).to(device)
        outputs = model(batch_t)

        state_p = torch.softmax(outputs['state'], dim=1).cpu().numpy()
        impact_p = torch.sigmoid(outputs['impact']).squeeze(-1).cpu().numpy()
        predict_p = torch.sigmoid(outputs['predict']).squeeze(-1).cpu().numpy()

        for j in range(len(batch)):
            s = starts[i + j]
            e = min(s + W, T)
            actual_len = e - s
            g = gauss[:actual_len]

            # State: uniform mean
            state_acc[s:e] += np.broadcast_to(state_p[j], (actual_len, 3))
            state_count[s:e] += 1.0

            # Impact/predict: Gaussian-weighted accumulation
            impact_acc[s:e] += impact_p[j] * g
            impact_wt[s:e] += g
            pred_acc[s:e] += predict_p[j] * g
            pred_wt[s:e] += g

    # Aggregate
    state_count = np.maximum(state_count, 1.0)
    state_probs = state_acc / state_count[:, None]

    impact_wt = np.maximum(impact_wt, 1e-8)
    impact_probs = impact_acc / impact_wt

    pred_wt = np.maximum(pred_wt, 1e-8)
    pred_probs = pred_acc / pred_wt

    return state_probs, impact_probs, pred_probs


# ────────────────────────────────────────────────────────────────────
# Main evaluation
# ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    # Build model
    channels = [int(c) for c in ckpt_args.get('channels', '64,64,128,128').split(',')]
    model = FallDetectionTCN(
        in_features=FEATURE_DIM,
        channels=channels,
        dropout=ckpt_args.get('dropout', 0.15),
        use_confidence_gate=not ckpt_args.get('no_conf_gate', False),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Build val dataset (same split as training)
    window_sec = ckpt_args.get('window_sec', 1.5)
    stride_sec = ckpt_args.get('stride_sec', 0.2)
    _, val_ds = FallDataset.build_splits(
        args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        fps=ckpt_args.get('fps'),
        window_sec=window_sec,
        stride_sec=stride_sec,
    )

    # ── Window-level metrics ──
    print("\n== Window-level evaluation ==")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    all_state_preds = []
    all_state_labels = []
    all_impact_probs = []
    all_impact_labels = []
    all_pred_probs = []
    all_pred_labels = []

    with torch.no_grad():
      for features, state, impact, pred in val_loader:
        features = features.to(device)
        outputs = model(features)
        all_state_preds.append(outputs['state'].argmax(1).cpu().numpy())
        all_state_labels.append(state.numpy())
        all_impact_probs.append(
            torch.sigmoid(outputs['impact']).squeeze(-1).cpu().numpy())
        all_impact_labels.append(impact.numpy())
        all_pred_probs.append(
            torch.sigmoid(outputs['predict']).squeeze(-1).cpu().numpy())
        all_pred_labels.append(pred.numpy())

    all_state_preds = np.concatenate(all_state_preds)
    all_state_labels = np.concatenate(all_state_labels)
    all_impact_probs = np.concatenate(all_impact_probs)
    all_impact_labels = np.concatenate(all_impact_labels)
    all_pred_probs = np.concatenate(all_pred_probs)
    all_pred_labels = np.concatenate(all_pred_labels)

    window_metrics = compute_window_metrics(
        all_state_preds, all_state_labels,
        all_impact_probs, all_impact_labels,
        all_pred_probs, all_pred_labels
    )

    for k, v in window_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Event-level metrics ──
    print("\n== Event-level evaluation ==")
    total_tp, total_fp, total_fn = 0, 0, 0

    for vid in val_ds.video_ids:
        if vid not in val_ds.video_data:
            continue
        vdata = val_ds.video_data[vid]
        features = vdata['features']
        vid_fps = vdata['fps']
        gt_centers = vdata['impact_centers']

        # Run dense inference
        state_probs, impact_probs, pred_probs = run_video_inference(
            model, features, vid_fps, device,
            window_sec=window_sec, stride_sec=0.1  # denser for evaluation
        )

        # Detect events
        events = detect_fall_events(
            impact_probs, state_probs, vid_fps,
            impact_thr=args.impact_thr,
            lie_thr=args.lie_thr,
            lie_sustain_sec=args.lie_sustain_sec,
            cooldown_sec=args.cooldown_sec,
        )

        # Match to GT
        tp, fp, fn = match_events_to_gt(
            events, gt_centers, vid_fps,
            tolerance_sec=args.tolerance_sec
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"  {vid}: GT={len(gt_centers)} Det={len(events)} "
              f"TP={tp} FP={fp} FN={fn}")

    event_metrics = compute_event_metrics(total_tp, total_fp, total_fn)
    print(f"\n  Event-level results:")
    for k, v in event_metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Save all metrics
    all_metrics = {**window_metrics, **event_metrics}
    save_json(all_metrics, str(output_dir / 'metrics.json'))
    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()

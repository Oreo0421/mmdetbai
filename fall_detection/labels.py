"""Label generation from interval annotations.

Generates:
  - Per-frame state labels: upright(0) / sit(1) / lie(2)
  - Per-frame impact labels: binary, derived from fall intervals
  - Per-frame prediction labels: binary, fall within H seconds
  - Sliding-window samples with aggregated labels
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from .utils import (STATE_UPRIGHT, STATE_SIT, STATE_LIE,
                    DEFAULT_FPS, DEFAULT_IMPACT_W_SEC, DEFAULT_HORIZON_SEC,
                    DEFAULT_WINDOW_SEC, DEFAULT_STRIDE_SEC, DEFAULT_OVERLAP_RATIO,
                    DEFAULT_CONF_THR, frames_from_sec)


@dataclass
class WindowSample:
    """A single sliding-window sample."""
    video_id: str
    start_frame: int
    end_frame: int       # exclusive
    features: np.ndarray  # (W, F)
    state_label: int      # majority state in window
    impact_label: float   # 1.0 if overlap with impact interval > ratio
    pred_label: float     # 1.0 if fall within horizon after window end


# ────────────────────────────────────────────────────────────────────
# Per-frame state labels
# ────────────────────────────────────────────────────────────────────

def generate_frame_states(label_data: Dict[str, Any], T: int) -> np.ndarray:
    """Generate per-frame state labels from interval annotations.

    Priority: fall (post-impact → lie) > sit > lie > upright.
    Within fall intervals, before impact center → upright, after → lie.
    Impact center is computed separately; here we use a simple heuristic:
    first 40% of fall interval = upright, last 60% = lie.

    Args:
        label_data: parsed JSON with 'intervals' containing 'fall', 'sit', 'lie'.
        T: total number of frames.

    Returns:
        states: (T,) int array with values in {0, 1, 2}.
    """
    states = np.full(T, STATE_UPRIGHT, dtype=np.int64)
    intervals = label_data.get('intervals', {})

    # Voluntary lie intervals
    for start, end in intervals.get('lie', []):
        start = max(0, int(start))
        end = min(T, int(end) + 1)  # make end exclusive
        states[start:end] = STATE_LIE

    # Sit intervals
    for start, end in intervals.get('sit', []):
        start = max(0, int(start))
        end = min(T, int(end) + 1)
        states[start:end] = STATE_SIT

    # Fall intervals: split into pre-impact (upright) and post-impact (lie)
    for start, end in intervals.get('fall', []):
        start = max(0, int(start))
        end = min(T, int(end) + 1)
        duration = end - start
        if duration <= 0:
            continue
        # Heuristic split: first 40% upright, last 60% lie
        # (actual impact detection refines this below)
        split = start + max(1, int(duration * 0.4))
        states[start:split] = STATE_UPRIGHT
        states[split:end] = STATE_LIE

    return states


# ────────────────────────────────────────────────────────────────────
# Impact detection from fall intervals
# ────────────────────────────────────────────────────────────────────

def compute_skeleton_height(kpts: np.ndarray, conf_thr: float = DEFAULT_CONF_THR
                            ) -> np.ndarray:
    """Compute per-frame skeleton "height" = max_y - min_y of confident joints.

    This measures vertical extent: large when standing, small when lying.

    Args:
        kpts: (T, J, 3) raw or preprocessed keypoints (use raw for better signal).

    Returns:
        height: (T,) array.
    """
    T, J, _ = kpts.shape
    height = np.zeros(T, dtype=np.float32)

    for t in range(T):
        mask = kpts[t, :, 2] >= conf_thr
        if mask.sum() >= 2:
            y_vals = kpts[t, mask, 1]
            height[t] = y_vals.max() - y_vals.min()
        elif mask.sum() == 1:
            height[t] = 0.0
        else:
            # No confident joints: interpolate later
            height[t] = np.nan

    # Interpolate NaN values
    nans = np.isnan(height)
    if nans.any() and not nans.all():
        valid_idx = np.where(~nans)[0]
        height = np.interp(np.arange(T), valid_idx, height[valid_idx])

    return height


def find_impact_frames(kpts_raw: np.ndarray, label_data: Dict[str, Any],
                       fps: float = DEFAULT_FPS,
                       w_sec: float = DEFAULT_IMPACT_W_SEC,
                       conf_thr: float = DEFAULT_CONF_THR
                       ) -> Tuple[np.ndarray, List[int]]:
    """Detect impact frames within each fall interval.

    For each fall interval:
      1. Compute h(t) = skeleton height (vertical extent)
      2. Compute v(t) = h(t) - h(t-1) (height velocity)
      3. t* = argmin v(t) (steepest drop)
      4. Impact interval = [t*-w, t*+w]

    Args:
        kpts_raw: (T, J, 3) raw keypoints (before centering/normalization).
        label_data: JSON labels with fall intervals.
        fps: frame rate.
        w_sec: half-width of impact interval in seconds.

    Returns:
        impact: (T,) binary array (1 = impact frame).
        impact_centers: list of t* values for each fall interval.
    """
    T = kpts_raw.shape[0]
    w = frames_from_sec(w_sec, fps)
    impact = np.zeros(T, dtype=np.float32)
    impact_centers = []

    height = compute_skeleton_height(kpts_raw, conf_thr=conf_thr)
    # Velocity: h(t) - h(t-1)
    velocity = np.zeros(T, dtype=np.float32)
    velocity[1:] = height[1:] - height[:-1]

    intervals = label_data.get('intervals', {})
    for start, end in intervals.get('fall', []):
        start = max(1, int(start))  # need t-1 for velocity
        end = min(T, int(end) + 1)
        if end - start < 2:
            continue

        # Find steepest drop (most negative velocity) in this interval
        v_slice = velocity[start:end]
        t_star_local = int(np.argmin(v_slice))
        t_star = start + t_star_local
        impact_centers.append(t_star)

        # Mark impact interval
        i_start = max(0, t_star - w)
        i_end = min(T, t_star + w + 1)
        impact[i_start:i_end] = 1.0

    return impact, impact_centers


def refine_frame_states_with_impact(states: np.ndarray,
                                    label_data: Dict[str, Any],
                                    impact_centers: List[int],
                                    T: int) -> np.ndarray:
    """Refine frame states using detected impact centers.

    Within each fall interval, frames before impact center → upright,
    frames from impact center onward → lie.
    """
    states = states.copy()
    intervals = label_data.get('intervals', {})
    fall_intervals = intervals.get('fall', [])

    for idx, (start, end) in enumerate(fall_intervals):
        start = max(0, int(start))
        end = min(T, int(end) + 1)
        if idx < len(impact_centers):
            t_star = impact_centers[idx]
            states[start:t_star] = STATE_UPRIGHT
            states[t_star:end] = STATE_LIE

    return states


# ────────────────────────────────────────────────────────────────────
# Prediction labels (fall within horizon)
# ────────────────────────────────────────────────────────────────────

def generate_prediction_labels(impact: np.ndarray, fps: float = DEFAULT_FPS,
                               horizon_sec: float = DEFAULT_HORIZON_SEC
                               ) -> np.ndarray:
    """Generate per-frame prediction labels.

    pred_label[t] = 1 if any impact frame exists in (t, t + horizon_frames].

    Args:
        impact: (T,) binary impact array.
        fps: frame rate.
        horizon_sec: prediction horizon in seconds.

    Returns:
        pred: (T,) binary array.
    """
    T = len(impact)
    H = frames_from_sec(horizon_sec, fps)
    pred = np.zeros(T, dtype=np.float32)

    # Find all impact frame indices
    impact_idx = np.where(impact > 0.5)[0]
    if len(impact_idx) == 0:
        return pred

    for t in range(T):
        # Check if any impact in (t, t+H]
        window_start = t + 1
        window_end = t + H + 1
        if np.any((impact_idx >= window_start) & (impact_idx < window_end)):
            pred[t] = 1.0

    return pred


# ────────────────────────────────────────────────────────────────────
# Sliding-window generation
# ────────────────────────────────────────────────────────────────────

def create_windows(
    features: np.ndarray,
    frame_states: np.ndarray,
    impact: np.ndarray,
    pred_labels: np.ndarray,
    video_id: str,
    fps: float = DEFAULT_FPS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> List[WindowSample]:
    """Create sliding-window samples from per-frame data.

    Args:
        features: (T, F) feature array.
        frame_states: (T,) state labels.
        impact: (T,) impact labels.
        pred_labels: (T,) prediction labels.
        video_id: identifier.
        fps, window_sec, stride_sec: window parameters.
        overlap_ratio: min overlap with impact interval for window impact=1.

    Returns:
        windows: list of WindowSample.
    """
    T = features.shape[0]
    W = frames_from_sec(window_sec, fps)
    S = max(1, frames_from_sec(stride_sec, fps))

    windows = []
    for start in range(0, T - W + 1, S):
        end = start + W

        # Features
        feat = features[start:end].copy()

        # State: majority vote
        state_counts = np.bincount(frame_states[start:end].astype(int),
                                   minlength=3)
        state_label = int(state_counts.argmax())

        # Impact: check overlap ratio
        impact_frames_in_window = impact[start:end].sum()
        impact_label = 1.0 if (impact_frames_in_window / W) >= overlap_ratio else 0.0

        # Prediction: label based on the END of the window
        # = does a fall happen within horizon after this window?
        pred_label = float(pred_labels[min(end - 1, T - 1)])

        windows.append(WindowSample(
            video_id=video_id,
            start_frame=start,
            end_frame=end,
            features=feat,
            state_label=state_label,
            impact_label=impact_label,
            pred_label=pred_label,
        ))

    return windows


def generate_all_labels(
    kpts_raw: np.ndarray,
    label_data: Dict[str, Any],
    fps: float = DEFAULT_FPS,
    impact_w_sec: float = DEFAULT_IMPACT_W_SEC,
    horizon_sec: float = DEFAULT_HORIZON_SEC,
    conf_thr: float = DEFAULT_CONF_THR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Generate all per-frame labels from raw keypoints and annotations.

    Returns:
        frame_states: (T,) int array.
        impact: (T,) float array.
        pred_labels: (T,) float array.
        impact_centers: list of impact center frame indices.
    """
    T = kpts_raw.shape[0]

    # Frame states from intervals
    frame_states = generate_frame_states(label_data, T)

    # Impact detection
    impact, impact_centers = find_impact_frames(
        kpts_raw, label_data, fps=fps, w_sec=impact_w_sec, conf_thr=conf_thr)

    # Refine states with actual impact centers
    frame_states = refine_frame_states_with_impact(
        frame_states, label_data, impact_centers, T)

    # Prediction labels
    pred_labels = generate_prediction_labels(impact, fps=fps,
                                             horizon_sec=horizon_sec)

    return frame_states, impact, pred_labels, impact_centers

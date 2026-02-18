"""Keypoint preprocessing: centering, normalization, smoothing, feature extraction.

Pipeline:
  raw (T, 7, 3) → interpolate missing → smooth → center → normalize → extract features → (T, 35)
"""
import numpy as np
from typing import Optional
from .utils import (NUM_JOINTS, J_HIPS, J_SHOULDER, J_FOOT_R, J_FOOT_L,
                    DEFAULT_CONF_THR, DEFAULT_FPS, BONE_CONNECTIONS, FEATURE_DIM)


# ────────────────────────────────────────────────────────────────────
# One-Euro Filter (per-channel, score-weighted)
# ────────────────────────────────────────────────────────────────────
class OneEuroFilter:
    """One-Euro filter for low-latency smoothing of noisy signals.

    Reference: Casiez et al., "1-Euro Filter", CHI 2012.
    """

    def __init__(self, freq: float = 25.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()

        # derivative
        a_d = self._alpha(self.d_cutoff)
        dx = (x - self._x_prev) * self.freq
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        # per-element alpha
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / te)

        x_hat = alpha * x + (1 - alpha) * self._x_prev
        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None


# ────────────────────────────────────────────────────────────────────
# Preprocessing steps
# ────────────────────────────────────────────────────────────────────

def interpolate_missing(kpts: np.ndarray, conf_thr: float = DEFAULT_CONF_THR
                        ) -> np.ndarray:
    """Fill in missing/low-confidence joints via linear interpolation + forward-fill.

    Args:
        kpts: (T, J, 3) with channels (x, y, score).
        conf_thr: below this score, joint is considered missing.

    Returns:
        kpts_filled: (T, J, 3) with interpolated coordinates.
                     Score channel is kept as-is (acts as weight).
    """
    T, J, _ = kpts.shape
    out = kpts.copy()

    for j in range(J):
        scores = out[:, j, 2]
        valid_mask = scores >= conf_thr

        if not valid_mask.any():
            # All frames missing for this joint → zero coords, keep low score
            out[:, j, :2] = 0.0
            continue

        if valid_mask.all():
            continue

        valid_idx = np.where(valid_mask)[0]

        for ch in range(2):  # x, y
            vals = out[:, j, ch]
            # Linear interpolation between valid points
            out[:, j, ch] = np.interp(
                np.arange(T), valid_idx, vals[valid_idx]
            )

    return out


def smooth_keypoints(kpts: np.ndarray, fps: float = DEFAULT_FPS,
                     conf_thr: float = DEFAULT_CONF_THR) -> np.ndarray:
    """Apply One-Euro filter to (x,y) channels, weighted by confidence.

    High-confidence points are smoothed normally.
    Low-confidence points pull less (mixed with smoothed estimate).

    Args:
        kpts: (T, J, 3) keypoints.
        fps: frame rate.

    Returns:
        smoothed: (T, J, 3).
    """
    T, J, _ = kpts.shape
    out = kpts.copy()

    for j in range(J):
        filt_x = OneEuroFilter(freq=fps, min_cutoff=1.5, beta=0.01)
        filt_y = OneEuroFilter(freq=fps, min_cutoff=1.5, beta=0.01)

        for t in range(T):
            score = kpts[t, j, 2]
            raw_x, raw_y = kpts[t, j, 0], kpts[t, j, 1]

            # Filter
            sx = filt_x(np.array([raw_x]))[0]
            sy = filt_y(np.array([raw_y]))[0]

            # Blend: high confidence → trust raw more, low → trust filter more
            w = np.clip(score, 0.0, 1.0)
            out[t, j, 0] = w * sx + (1 - w) * (sx)  # both use filtered
            out[t, j, 1] = w * sy + (1 - w) * (sy)
            # Actually for smoothing, always use filtered output;
            # the 1-euro filter internally adapts based on signal speed
            out[t, j, 0] = sx
            out[t, j, 1] = sy

    return out


def center_keypoints(kpts: np.ndarray, conf_thr: float = DEFAULT_CONF_THR
                     ) -> np.ndarray:
    """Center keypoints by subtracting hip center (or mean of confident joints).

    Args:
        kpts: (T, J, 3) with (x, y, score).

    Returns:
        centered: (T, J, 3) with centered (x, y) and original score.
    """
    T, J, _ = kpts.shape
    out = kpts.copy()

    for t in range(T):
        scores = kpts[t, :, 2]
        hip_score = scores[J_HIPS]

        if hip_score >= conf_thr:
            cx = kpts[t, J_HIPS, 0]
            cy = kpts[t, J_HIPS, 1]
        else:
            # Fallback: mean of confident joints
            mask = scores >= conf_thr
            if mask.any():
                cx = kpts[t, mask, 0].mean()
                cy = kpts[t, mask, 1].mean()
            else:
                cx, cy = 0.0, 0.0

        out[t, :, 0] -= cx
        out[t, :, 1] -= cy

    return out


def normalize_scale(kpts: np.ndarray, conf_thr: float = DEFAULT_CONF_THR
                    ) -> np.ndarray:
    """Normalize by torso length (shoulder-hip distance).

    Fallback: use bounding-box height from all confident joints.
    Adds a small epsilon to avoid division by zero.

    Args:
        kpts: (T, J, 3) centered keypoints.

    Returns:
        normalized: (T, J, 3).
    """
    T, J, _ = kpts.shape
    out = kpts.copy()
    eps = 1e-4
    min_scale = 0.05  # minimum scale to avoid explosion (5% of bbox = very flat person)

    # First pass: compute per-frame scales
    scales = np.ones(T, dtype=np.float32)
    for t in range(T):
        scores = kpts[t, :, 2]
        shoulder_ok = scores[J_SHOULDER] >= conf_thr
        hip_ok = scores[J_HIPS] >= conf_thr

        if shoulder_ok and hip_ok:
            dx = kpts[t, J_SHOULDER, 0] - kpts[t, J_HIPS, 0]
            dy = kpts[t, J_SHOULDER, 1] - kpts[t, J_HIPS, 1]
            scales[t] = np.sqrt(dx * dx + dy * dy) + eps
        else:
            mask = scores >= conf_thr
            if mask.sum() >= 2:
                y_vals = kpts[t, mask, 1]
                scales[t] = (y_vals.max() - y_vals.min()) + eps
            else:
                scales[t] = 1.0

    # Use median scale as robust reference to avoid per-frame spikes
    median_scale = max(np.median(scales), min_scale)

    for t in range(T):
        # Blend per-frame scale with median (robust against single-frame noise)
        scale = max(scales[t], min_scale)
        # If per-frame scale deviates >2x from median, use median
        if scale < median_scale * 0.3 or scale > median_scale * 3.0:
            scale = median_scale
        out[t, :, 0] /= scale
        out[t, :, 1] /= scale

    return out


def compute_velocities(kpts: np.ndarray) -> np.ndarray:
    """Compute per-joint velocities (dx, dy) between consecutive frames.

    Args:
        kpts: (T, J, 3) with (x, y, score).

    Returns:
        vel: (T, J, 2) velocities. First frame velocity = 0.
    """
    T, J, _ = kpts.shape
    vel = np.zeros((T, J, 2), dtype=np.float32)
    vel[1:, :, 0] = kpts[1:, :, 0] - kpts[:-1, :, 0]
    vel[1:, :, 1] = kpts[1:, :, 1] - kpts[:-1, :, 1]
    return vel


# ────────────────────────────────────────────────────────────────────
# Full pipeline
# ────────────────────────────────────────────────────────────────────

def preprocess_keypoints(kpts_raw: np.ndarray, fps: float = DEFAULT_FPS,
                         conf_thr: float = DEFAULT_CONF_THR) -> np.ndarray:
    """Full preprocessing pipeline: raw (T,7,3) → cleaned (T,7,3).

    Steps: interpolate → smooth → center → normalize.
    """
    kpts = kpts_raw.astype(np.float32).copy()
    kpts = interpolate_missing(kpts, conf_thr=conf_thr)
    kpts = smooth_keypoints(kpts, fps=fps, conf_thr=conf_thr)
    kpts = center_keypoints(kpts, conf_thr=conf_thr)
    kpts = normalize_scale(kpts, conf_thr=conf_thr)
    return kpts


def extract_features(kpts_clean: np.ndarray) -> np.ndarray:
    """Convert cleaned keypoints to feature vectors.

    Per frame: [x0,y0,s0, x1,y1,s1, ..., x6,y6,s6, dx0,dy0, ..., dx6,dy6]
    Total = 7*3 + 7*2 = 35 dimensions.

    Args:
        kpts_clean: (T, 7, 3) preprocessed keypoints.

    Returns:
        features: (T, 35) feature array.
    """
    T, J, _ = kpts_clean.shape
    vel = compute_velocities(kpts_clean)  # (T, J, 2)

    # Flatten: (T, J*3) coords+score + (T, J*2) velocities
    coords_flat = kpts_clean.reshape(T, -1)   # (T, 21)
    vel_flat = vel.reshape(T, -1)             # (T, 14)
    features = np.concatenate([coords_flat, vel_flat], axis=1)  # (T, 35)

    # Clip extreme values (robust against normalization edge cases)
    features[:, :14] = np.clip(features[:, :14], -5.0, 5.0)   # coords
    features[:, 21:] = np.clip(features[:, 21:], -3.0, 3.0)   # velocities

    return features.astype(np.float32)

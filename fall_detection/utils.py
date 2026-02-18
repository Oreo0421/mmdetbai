"""Shared constants and utility functions."""
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─── Joint layout (matches your thermal IR 7-keypoint model) ───
JOINT_NAMES = ['head', 'shoulder', 'hand_R', 'hand_L', 'hips', 'foot_R', 'foot_L']
NUM_JOINTS = len(JOINT_NAMES)
J_HEAD, J_SHOULDER, J_HAND_R, J_HAND_L, J_HIPS, J_FOOT_R, J_FOOT_L = range(7)

BONE_CONNECTIONS = [
    (J_HEAD, J_SHOULDER),      # neck
    (J_SHOULDER, J_HAND_R),    # right arm
    (J_SHOULDER, J_HAND_L),    # left arm
    (J_SHOULDER, J_HIPS),      # torso
    (J_HIPS, J_FOOT_R),        # right leg
    (J_HIPS, J_FOOT_L),        # left leg
]

# ─── State labels ───
STATE_UPRIGHT = 0
STATE_SIT = 1
STATE_LIE = 2
STATE_NAMES = ['upright', 'sit', 'lie']
NUM_STATES = 3

# ─── Defaults ───
DEFAULT_FPS = 25
DEFAULT_WINDOW_SEC = 1.5
DEFAULT_STRIDE_SEC = 0.2
DEFAULT_IMPACT_W_SEC = 0.3
DEFAULT_HORIZON_SEC = 1.0
DEFAULT_OVERLAP_RATIO = 0.3
DEFAULT_CONF_THR = 0.2       # below this, joint is "missing"

# Feature dim: 7 joints * (x, y, score, dx, dy) = 35
FEATURE_DIM = NUM_JOINTS * 5


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def split_video_ids(video_ids: List[str], val_ratio: float = 0.2,
                    seed: int = 42) -> tuple:
    """Deterministic train/val split by video_id."""
    rng = random.Random(seed)
    ids = sorted(video_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]


def frames_from_sec(sec: float, fps: float) -> int:
    """Convert seconds to frames (rounded)."""
    return max(1, round(sec * fps))

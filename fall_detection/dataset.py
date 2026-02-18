"""PyTorch Dataset for fall detection from offline keypoints.

Expected directory structure:
    dataset_root/
        videos/
            <video_id>.npz   # contains 'keypoints' array of shape (T, 7, 3)
        labels/
            <video_id>.json  # contains fps, intervals

Example JSON schema:
    {
        "video_id": "video_001",
        "fps": 25,
        "intervals": {
            "fall": [[100, 180], [500, 570]],
            "sit":  [[200, 350]],
            "lie":  [[400, 480]]
        }
    }

Example NPZ:
    np.savez('video_001.npz', keypoints=kpts)  # kpts.shape = (T, 7, 3)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from .utils import (load_json, split_video_ids, DEFAULT_FPS,
                    DEFAULT_WINDOW_SEC, DEFAULT_STRIDE_SEC,
                    DEFAULT_IMPACT_W_SEC, DEFAULT_HORIZON_SEC,
                    DEFAULT_OVERLAP_RATIO, DEFAULT_CONF_THR, FEATURE_DIM)
from .preprocess import preprocess_keypoints, extract_features
from .labels import generate_all_labels, create_windows, WindowSample


class FallDataset(Dataset):
    """Fall detection dataset from offline keypoints.

    Loads all videos, preprocesses keypoints, generates labels,
    creates sliding windows.

    Args:
        data_root: path to dataset root.
        video_ids: list of video IDs to include (for train/val split).
        fps: default fps if not in JSON. None = read from JSON (required).
        window_sec: sliding window length in seconds.
        stride_sec: sliding window stride in seconds.
        impact_w_sec: half-width of impact interval in seconds.
        horizon_sec: prediction horizon in seconds.
        overlap_ratio: min overlap for window impact label.
        conf_thr: confidence threshold for missing joints.
    """

    def __init__(
        self,
        data_root: str,
        video_ids: Optional[List[str]] = None,
        fps: Optional[float] = None,
        window_sec: float = DEFAULT_WINDOW_SEC,
        stride_sec: float = DEFAULT_STRIDE_SEC,
        impact_w_sec: float = DEFAULT_IMPACT_W_SEC,
        horizon_sec: float = DEFAULT_HORIZON_SEC,
        overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
        conf_thr: float = DEFAULT_CONF_THR,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.videos_dir = self.data_root / 'videos'
        self.labels_dir = self.data_root / 'labels'

        # Discover video IDs
        if video_ids is None:
            npz_files = sorted(self.videos_dir.glob('*.npz'))
            video_ids = [f.stem for f in npz_files]

        self.video_ids = video_ids
        self.samples: List[WindowSample] = []
        self.video_data: Dict[str, dict] = {}  # cache for inference

        # Stats for logging
        n_fall_windows = 0
        n_impact_windows = 0
        n_pred_windows = 0

        for vid in video_ids:
            npz_path = self.videos_dir / f'{vid}.npz'
            json_path = self.labels_dir / f'{vid}.json'

            if not npz_path.exists() or not json_path.exists():
                print(f"  [WARN] Missing files for {vid}, skipping.")
                continue

            # Load keypoints
            data = np.load(str(npz_path))
            kpts_raw = data['keypoints'].astype(np.float32)  # (T, 7, 3)
            T = kpts_raw.shape[0]

            # Load labels
            label_data = load_json(str(json_path))
            vid_fps = label_data.get('fps', fps)
            if vid_fps is None:
                vid_fps = DEFAULT_FPS
                print(f"  [WARN] No fps for {vid}, using default {DEFAULT_FPS}")

            # Preprocess
            kpts_clean = preprocess_keypoints(kpts_raw, fps=vid_fps,
                                              conf_thr=conf_thr)
            features = extract_features(kpts_clean)  # (T, F)

            # Generate labels
            frame_states, impact, pred_labels, impact_centers = \
                generate_all_labels(kpts_raw, label_data, fps=vid_fps,
                                    impact_w_sec=impact_w_sec,
                                    horizon_sec=horizon_sec,
                                    conf_thr=conf_thr)

            # Cache for inference
            self.video_data[vid] = {
                'kpts_raw': kpts_raw,
                'kpts_clean': kpts_clean,
                'features': features,
                'frame_states': frame_states,
                'impact': impact,
                'pred_labels': pred_labels,
                'impact_centers': impact_centers,
                'label_data': label_data,
                'fps': vid_fps,
            }

            # Create windows
            windows = create_windows(
                features, frame_states, impact, pred_labels,
                video_id=vid, fps=vid_fps,
                window_sec=window_sec, stride_sec=stride_sec,
                overlap_ratio=overlap_ratio,
            )
            self.samples.extend(windows)

            # Stats
            for w in windows:
                if w.impact_label > 0.5:
                    n_impact_windows += 1
                if w.pred_label > 0.5:
                    n_pred_windows += 1
                if w.state_label == 2:
                    n_fall_windows += 1

        n_total = len(self.samples)
        print(f"  Dataset: {len(video_ids)} videos, {n_total} windows")
        if n_total > 0:
            print(f"  Impact windows: {n_impact_windows}/{n_total} "
                  f"({100*n_impact_windows/n_total:.1f}%)")
            print(f"  Prediction windows: {n_pred_windows}/{n_total} "
                  f"({100*n_pred_windows/n_total:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample = self.samples[idx]
        features = torch.from_numpy(sample.features.copy())  # (W, F)

        if self.augment:
            features = self._augment(features)

        state = torch.tensor(sample.state_label, dtype=torch.long)
        impact = torch.tensor(sample.impact_label, dtype=torch.float32)
        pred = torch.tensor(sample.pred_label, dtype=torch.float32)
        return features, state, impact, pred

    def _augment(self, features: torch.Tensor) -> torch.Tensor:
        """Training augmentation for robustness.

        - Random scale jitter on coordinates (±20%)
        - Gaussian noise on coordinates
        - Random joint dropout (zero out joints with p=0.1)
        - Random temporal flip (reverse sequence, only if no impact)
        """
        W, F = features.shape
        J = 7

        # 1. Random scale jitter on (x, y) coords
        if torch.rand(1) < 0.5:
            scale = 0.8 + 0.4 * torch.rand(1)  # [0.8, 1.2]
            for j in range(J):
                features[:, j*3] *= scale      # x
                features[:, j*3+1] *= scale    # y
                features[:, 21+j*2] *= scale   # dx
                features[:, 21+j*2+1] *= scale # dy

        # 2. Gaussian noise on coordinates
        if torch.rand(1) < 0.5:
            noise_std = 0.02 + 0.06 * torch.rand(1)  # [0.02, 0.08]
            for j in range(J):
                features[:, j*3] += torch.randn(W) * noise_std
                features[:, j*3+1] += torch.randn(W) * noise_std

        # 3. Random joint dropout
        if torch.rand(1) < 0.3:
            for j in range(J):
                if torch.rand(1) < 0.15:
                    features[:, j*3] = 0
                    features[:, j*3+1] = 0
                    features[:, j*3+2] = 0  # score → 0
                    features[:, 21+j*2] = 0
                    features[:, 21+j*2+1] = 0

        # 4. Random horizontal flip (negate x coords and swap left/right)
        if torch.rand(1) < 0.5:
            for j in range(J):
                features[:, j*3] = -features[:, j*3]      # negate x
                features[:, 21+j*2] = -features[:, 21+j*2]  # negate dx
            # Swap hand_R(2) ↔ hand_L(3), foot_R(5) ↔ foot_L(6)
            for ja, jb in [(2, 3), (5, 6)]:
                for offset in [0, 1, 2]:  # x, y, score
                    features[:, ja*3+offset], features[:, jb*3+offset] = \
                        features[:, jb*3+offset].clone(), features[:, ja*3+offset].clone()
                for offset in [0, 1]:  # dx, dy
                    features[:, 21+ja*2+offset], features[:, 21+jb*2+offset] = \
                        features[:, 21+jb*2+offset].clone(), features[:, 21+ja*2+offset].clone()

        return features

    @staticmethod
    def build_splits(data_root: str, val_ratio: float = 0.2, seed: int = 42,
                     **kwargs) -> Tuple['FallDataset', 'FallDataset']:
        """Build train/val datasets with deterministic video-level split.

        Auto-detects _train/_val suffix in filenames. If found, uses those
        for splitting (ignores val_ratio). Otherwise, random split.
        """
        videos_dir = Path(data_root) / 'videos'
        all_ids = sorted([f.stem for f in videos_dir.glob('*.npz')])

        if len(all_ids) == 0:
            raise FileNotFoundError(f"No .npz files in {videos_dir}")

        # Auto-detect _train/_val suffix
        has_suffix = any(x.endswith('_train') for x in all_ids)
        if has_suffix:
            train_ids = [x for x in all_ids if x.endswith('_train')]
            val_ids = [x for x in all_ids if x.endswith('_val')]
            # Include unsuffixed files in train
            neither = [x for x in all_ids
                       if not x.endswith('_train') and not x.endswith('_val')]
            train_ids.extend(neither)
        else:
            train_ids, val_ids = split_video_ids(all_ids, val_ratio=val_ratio,
                                                 seed=seed)
        print(f"Train videos ({len(train_ids)}): {train_ids}")
        print(f"Val videos ({len(val_ids)}): {val_ids}")

        train_ds = FallDataset(data_root, video_ids=train_ids, **kwargs)
        val_ds = FallDataset(data_root, video_ids=val_ids, **kwargs)
        return train_ds, val_ds

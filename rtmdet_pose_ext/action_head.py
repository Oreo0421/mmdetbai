# -*- coding: utf-8 -*-
"""
ActionTemporalHead: GRU-based temporal action classifier on keypoint sequences.

Architecture:
  Per-frame encoder: Linear(K*3, 64) + ReLU + Linear(64, 64) + ReLU
  Temporal GRU:      GRU(64, 64, 1 layer) — skipped when T=1
  Classifier:        Linear(64, num_classes)

Training (single-frame, T=1):
  - Receives normalized keypoints from pose_head output
  - 10-class action classification (CrossEntropy loss)
  - GRU bypassed, encoder output goes directly to classifier

Inference with tracker (T>1):
  - Keypoints buffered per track_id over multiple frames
  - GRU processes full sequence for temporal action prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

from mmdet.registry import MODELS


# ---- 10 Action Classes ----
ACTION_CLASSES = [
    'Standing still',       # 0
    'Walking',              # 1
    'Sitting down',         # 2
    'Standing up',          # 3
    'Lying down',           # 4
    'Getting up',           # 5
    'Falling walking',      # 6
    'Falling standing',     # 7
    'Falling sitting',      # 8
    'Falling standing up',  # 9
]

FALLING_CLASS_IDS = {6, 7, 8, 9}


@MODELS.register_module(force=True)
class ActionTemporalHead(nn.Module):
    """Temporal action classification head based on keypoint sequences.

    Args:
        num_keypoints: Number of keypoints per person (default: 7).
        kpt_dim: Dimension per keypoint: x, y, visibility_score (default: 3).
        embed_dim: Per-frame encoder output dimension (default: 64).
        hidden_dim: GRU hidden dimension (default: 64).
        num_gru_layers: Number of GRU layers (default: 1).
        num_classes: Number of action classes. 1 = binary, 10 = multi-class.
        loss_weight: Weight for the action loss (default: 1.0).
        pos_weight: Positive class weight for BCE (binary mode only).
        class_weight: Per-class weights for CrossEntropy (multi-class mode).
        dropout: Dropout rate in encoder (default: 0.1).
    """

    def __init__(
        self,
        num_keypoints: int = 7,
        kpt_dim: int = 3,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_gru_layers: int = 1,
        num_classes: int = 10,
        loss_weight: float = 1.0,
        pos_weight: float = 1.0,
        class_weight: Optional[List[float]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.kpt_dim = kpt_dim
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.hidden_dim = hidden_dim

        input_dim = num_keypoints * kpt_dim  # 7 * 3 = 21

        # Per-frame encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

        # Temporal GRU (used when T > 1)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
        )

        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # BCE pos_weight for binary mode
        self.register_buffer(
            'pos_weight', torch.tensor([pos_weight], dtype=torch.float32)
        )

        # Per-class weights for multi-class CE
        if class_weight is not None:
            self.register_buffer(
                'class_weight',
                torch.tensor(class_weight, dtype=torch.float32)
            )
        else:
            self.class_weight = None

    def forward(self, kpt_features: Tensor) -> Tensor:
        """Forward pass.

        Args:
            kpt_features: (N, T, K*D) normalized keypoint features.
                T=1 for single-frame training.
                T>1 for temporal inference with tracker.

        Returns:
            logits: (N, num_classes)
        """
        N, T, D = kpt_features.shape

        # Encode each frame
        encoded = self.encoder(kpt_features.reshape(N * T, D))
        encoded = encoded.reshape(N, T, -1)  # (N, T, embed_dim)

        if T == 1:
            # Single frame: skip GRU, use encoder output directly
            out = encoded.squeeze(1)  # (N, embed_dim)
        else:
            # Temporal: run GRU, take last hidden state
            _, h_n = self.gru(encoded)
            out = h_n[-1]  # (N, hidden_dim)

        return self.classifier(out)  # (N, num_classes)

    def loss(self, kpt_features: Tensor, gt_labels: Tensor) -> dict:
        """Calculate action classification loss.

        Args:
            kpt_features: (N, T, K*D) normalized keypoint features.
            gt_labels: (N,) action class labels (0-9) or binary (0/1).

        Returns:
            dict with 'loss_action'.
        """
        logits = self.forward(kpt_features)  # (N, num_classes)

        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1),
                gt_labels.float(),
                pos_weight=self.pos_weight,
            )
        else:
            # Filter out invalid labels (action_class == -1)
            valid = gt_labels >= 0
            if valid.any():
                loss = F.cross_entropy(
                    logits[valid],
                    gt_labels[valid].long(),
                    weight=self.class_weight,
                )
            else:
                loss = logits.sum() * 0.0
        return {'loss_action': loss * self.loss_weight}

    def predict(self, kpt_features: Tensor) -> dict:
        """Predict action probabilities.

        Args:
            kpt_features: (N, T, K*D) normalized keypoint features.

        Returns:
            dict with:
                'action_probs': (N, num_classes) class probabilities
                'action_class': (N,) predicted class IDs
                'is_falling': (N,) bool tensor
            For binary (num_classes=1): returns legacy format (N,) probs.
        """
        logits = self.forward(kpt_features)
        if self.num_classes == 1:
            return torch.sigmoid(logits.squeeze(-1))  # (N,) backward compat

        probs = F.softmax(logits, dim=-1)  # (N, num_classes)
        action_class = probs.argmax(dim=-1)  # (N,)
        is_falling = action_class >= 6  # classes 6-9 are falling

        return {
            'action_probs': probs,
            'action_class': action_class,
            'is_falling': is_falling,
        }

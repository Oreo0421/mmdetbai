# -*- coding: utf-8 -*-
"""
ActionTemporalHead: GRU-based temporal action classifier on keypoint sequences.

Architecture:
  Per-frame encoder: Linear(K*3, 64) + ReLU + Linear(64, 64) + ReLU
  Temporal GRU:      GRU(64, 64, 1 layer) — skipped when T=1
  Classifier:        Linear(64, num_classes)

Training (single-frame, T=1):
  - Receives normalized keypoints from pose_head output
  - Binary falling classification (BCE loss)
  - GRU bypassed, encoder output goes directly to classifier

Inference with tracker (T>1):
  - Keypoints buffered per track_id over multiple frames
  - GRU processes full sequence for temporal action prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module(force=True)
class ActionTemporalHead(nn.Module):
    """Temporal action classification head based on keypoint sequences.

    Args:
        num_keypoints: Number of keypoints per person (default: 7).
        kpt_dim: Dimension per keypoint: x, y, visibility_score (default: 3).
        embed_dim: Per-frame encoder output dimension (default: 64).
        hidden_dim: GRU hidden dimension (default: 64).
        num_gru_layers: Number of GRU layers (default: 1).
        num_classes: Number of action classes. 1 = binary falling (default: 1).
        loss_weight: Weight for the action loss (default: 1.0).
        pos_weight: Positive class weight for BCE (handles class imbalance).
        dropout: Dropout rate in encoder (default: 0.1).
    """

    def __init__(
        self,
        num_keypoints: int = 7,
        kpt_dim: int = 3,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_gru_layers: int = 1,
        num_classes: int = 1,
        loss_weight: float = 1.0,
        pos_weight: float = 1.0,
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

        # BCE pos_weight for class imbalance (falling is rare)
        self.register_buffer(
            'pos_weight', torch.tensor([pos_weight], dtype=torch.float32)
        )

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
            gt_labels: (N,) binary falling labels (0 or 1).

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
            loss = F.cross_entropy(logits, gt_labels.long())

        return {'loss_action': loss * self.loss_weight}

    def predict(self, kpt_features: Tensor) -> Tensor:
        """Predict action probabilities.

        Args:
            kpt_features: (N, T, K*D) normalized keypoint features.

        Returns:
            probs: (N,) for binary, (N, num_classes) for multi-class.
        """
        logits = self.forward(kpt_features)
        if self.num_classes == 1:
            return torch.sigmoid(logits.squeeze(-1))  # (N,)
        else:
            return F.softmax(logits, dim=-1)  # (N, num_classes)

"""TCN-based multi-head model for fall detection.

Architecture:
  Input: (B, T, F=35)  →  permute to (B, F, T)
  →  Input projection: Conv1d(35, 64)
  →  4 TCN blocks with dilations [1, 2, 4, 8], kernel=3
  →  Adaptive average pool → (B, C)
  →  3 heads:
       state_head:   Linear → 3 classes (upright/sit/lie)
       impact_head:  Linear → 1 logit (binary)
       predict_head: Linear → 1 logit (binary)

Receptive field: with kernel=3 and dilations [1,2,4,8]:
  RF = 1 + 2*(1+2+4+8) = 31 frames ≈ 1.24s at 25fps

Confidence gating: per-joint scores are used to gate features
  before the temporal processing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple
from .utils import NUM_JOINTS, FEATURE_DIM


class ConfidenceGate(nn.Module):
    """Gate joint features by their confidence scores.

    Input features are arranged as [x0,y0,s0, x1,y1,s1, ..., dx0,dy0, ...].
    Extract scores from positions [2, 5, 8, 11, 14, 17, 20] and use them
    to weight the corresponding joint's features.
    """

    def __init__(self, num_joints: int = NUM_JOINTS):
        super().__init__()
        self.num_joints = num_joints
        # Score indices in the 35-dim feature vector
        # coords part: j*3+2 for j in 0..6 → [2, 5, 8, 11, 14, 17, 20]
        self.score_idx = [j * 3 + 2 for j in range(num_joints)]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, F) or (B, F, T) features.
               Assumes (B, T, F) input.

        Returns:
            gated: (B, T, F) features weighted by joint confidence.
        """
        B, T, F = x.shape
        scores = x[:, :, self.score_idx]  # (B, T, J)
        scores = scores.clamp(0, 1)

        # Build per-feature weight: each joint's features get that joint's score
        weights = torch.ones(B, T, F, device=x.device)

        for j in range(self.num_joints):
            s = scores[:, :, j:j+1]  # (B, T, 1)
            # Weight coord channels (j*3, j*3+1) — skip score channel itself
            weights[:, :, j*3] = s.squeeze(-1)
            weights[:, :, j*3+1] = s.squeeze(-1)
            # Weight velocity channels (21 + j*2, 21 + j*2+1)
            vel_base = self.num_joints * 3
            weights[:, :, vel_base + j*2] = s.squeeze(-1)
            weights[:, :, vel_base + j*2+1] = s.squeeze(-1)

        return x * weights


class TCNBlock(nn.Module):
    """Temporal Convolutional Block with residual connection.

    Conv1d(dilation) → BatchNorm → ReLU → Dropout → Conv1d(1) → BatchNorm → + residual → ReLU
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # same padding

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if channel dims differ
        self.residual = (nn.Conv1d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, T)

        Returns:
            out: (B, C_out, T)
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out + residual)
        return out


class FallDetectionTCN(nn.Module):
    """Multi-head TCN for fall detection, impact detection, and fall prediction.

    Args:
        in_features: input feature dimension per frame (default 35).
        channels: list of channel sizes for TCN blocks.
        kernel_size: temporal convolution kernel size.
        dropout: dropout rate.
        use_confidence_gate: whether to apply confidence gating.
    """

    def __init__(self, in_features: int = FEATURE_DIM,
                 channels: list = None,
                 kernel_size: int = 3,
                 dropout: float = 0.15,
                 use_confidence_gate: bool = True):
        super().__init__()

        if channels is None:
            channels = [64, 64, 128, 128]

        self.use_confidence_gate = use_confidence_gate
        if use_confidence_gate:
            self.conf_gate = ConfidenceGate()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_features, channels[0], 1),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # TCN blocks with increasing dilation
        dilations = [2 ** i for i in range(len(channels))]
        self.tcn_blocks = nn.ModuleList()
        in_ch = channels[0]
        for ch, dil in zip(channels, dilations):
            self.tcn_blocks.append(
                TCNBlock(in_ch, ch, kernel_size, dilation=dil, dropout=dropout)
            )
            in_ch = ch

        final_ch = channels[-1]

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Heads
        self.state_head = nn.Sequential(
            nn.Linear(final_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # upright, sit, lie
        )
        self.impact_head = nn.Sequential(
            nn.Linear(final_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.predict_head = nn.Sequential(
            nn.Linear(final_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: (B, T, F) feature tensor.

        Returns:
            dict with:
                'state': (B, 3) logits
                'impact': (B, 1) logits
                'predict': (B, 1) logits
        """
        # Confidence gating (operates on (B, T, F))
        if self.use_confidence_gate:
            x = self.conf_gate(x)

        # (B, T, F) → (B, F, T) for Conv1d
        x = x.permute(0, 2, 1)

        # Input projection
        x = self.input_proj(x)  # (B, C, T)

        # TCN blocks
        for block in self.tcn_blocks:
            x = block(x)  # (B, C, T)

        # Global average pool
        x = self.pool(x).squeeze(-1)  # (B, C)

        # Heads
        state_logits = self.state_head(x)     # (B, 3)
        impact_logits = self.impact_head(x)   # (B, 1)
        predict_logits = self.predict_head(x)  # (B, 1)

        return {
            'state': state_logits,
            'impact': impact_logits,
            'predict': predict_logits,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

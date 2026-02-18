# -*- coding: utf-8 -*-
"""
ActionTemporalHead: GRU-based temporal action classifier on keypoint sequences.

Architecture (V9):
  Per-frame encoder: Linear(K*3, 64) + ReLU + Linear(64, 64) + ReLU
  Temporal GRU:      GRU(64, 64, 1 layer) + LayerNorm
  Residual:          GRU output + last frame encoder output (skip connection)
  Classifier:        Linear(64, num_classes)

V8 mode (skip_gru_t1=True, default):
  T=1 → GRU bypassed, encoder output goes directly to classifier

V9 mode (skip_gru_t1=False):
  T=any → always use GRU + residual, even for single frames
  This ensures training (T=8), validation (T=1), and inference (T=1~30)
  all use the same pathway.
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

# ---- V13: State mapping (action_class → posture state) ----
# upright(0): Standing still(0), Walking(1), Standing up(3), Getting up(5)
# sit(1):     Sitting down(2)
# lie(2):     Lying down(4), Falling*(6-9) — person ends on ground
STATE_NAMES = ['upright', 'sit', 'lie']
ACTION_TO_STATE = [0, 0, 1, 0, 2, 0, 2, 2, 2, 2]  # len=10
ACTION_TO_FALL  = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # len=10

# ---- V10: Bone skeleton connections ----
# head(0), shoulder(1), hand_R(2), hand_L(3), hips(4), foot_R(5), foot_L(6)
BONE_CONNECTIONS = [
    (0, 1),  # head → shoulder (neck)
    (1, 2),  # shoulder → hand_right (right arm)
    (1, 3),  # shoulder → hand_left (left arm)
    (1, 4),  # shoulder → hips (torso)
    (4, 5),  # hips → foot_right (right leg)
    (4, 6),  # hips → foot_left (left leg)
]


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
        temporal_residual: Add skip connection (encoder + GRU) for stable
            temporal training. Default: True.
        skip_gru_t1: Skip GRU when T=1. True for V8 backward compat,
            False for V9 (always use GRU). Default: True.
        bone_mode: If True, input features are bone/skeleton features (V10)
            instead of raw keypoint features. Affects inference fallback.
            Default: False.
        appearance_dim: Dimension of RoI appearance features to concat (V11).
            0 = disabled (backward compatible). Default: 0.
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
        temporal_residual: bool = True,
        skip_gru_t1: bool = True,
        bone_mode: bool = False,
        appearance_dim: int = 0,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.kpt_dim = kpt_dim
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.hidden_dim = hidden_dim
        self.temporal_residual = temporal_residual
        self.skip_gru_t1 = skip_gru_t1
        self.bone_mode = bone_mode
        self.appearance_dim = appearance_dim

        input_dim = num_keypoints * kpt_dim + appearance_dim  # V11: 36+96=132

        # Per-frame encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

        # Temporal GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
        )

        # LayerNorm on GRU output for stable training
        self.gru_norm = nn.LayerNorm(hidden_dim)

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

        Returns:
            logits: (N, num_classes)
        """
        N, T, D = kpt_features.shape

        # Encode each frame
        encoded = self.encoder(kpt_features.reshape(N * T, D))
        encoded = encoded.reshape(N, T, -1)  # (N, T, embed_dim)

        if T == 1 and self.skip_gru_t1:
            # V8 behavior: skip GRU, use encoder output directly
            out = encoded.squeeze(1)  # (N, embed_dim)
        else:
            # Temporal path: GRU + LayerNorm
            _, h_n = self.gru(encoded)
            gru_out = self.gru_norm(h_n[-1])  # (N, hidden_dim)

            if self.temporal_residual:
                # Residual: last frame's encoder output + GRU temporal context
                # This ensures the classifier always gets useful encoder features,
                # even when GRU weights are freshly initialized.
                out = encoded[:, -1, :] + gru_out
            else:
                out = gru_out

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
            # Filter out invalid labels (-1) for binary mode
            valid = gt_labels >= 0
            if valid.any():
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1)[valid],
                    gt_labels[valid].float(),
                    pos_weight=self.pos_weight,
                )
            else:
                loss = logits.sum() * 0.0
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


@MODELS.register_module(force=True)
class ActionMultiHead(nn.Module):
    """V13: Multi-head temporal action classifier with shared GRU encoder.

    Three output heads sharing a common temporal encoder:
      - action_head: 10-class action classification (CE + class weights)
      - state_head:  3-class posture state (upright/sit/lie)
      - fall_head:   binary fall detection (focal loss)

    The shared encoder learns rich temporal representations.
    Auxiliary heads (state, fall) provide complementary supervision signals
    that help the model distinguish falls from similar non-fall actions.

    Label mapping is handled internally: the caller passes action_class (0-9)
    labels, and this module maps them to state and fall labels.

    Args:
        num_keypoints: Number of keypoints (or bones if bone_mode). Default: 6.
        kpt_dim: Dimension per keypoint/bone. Default: 6.
        embed_dim: Per-frame encoder output dimension. Default: 64.
        hidden_dim: GRU hidden dimension. Default: 64.
        num_gru_layers: Number of GRU layers. Default: 1.
        num_classes: Number of action classes. Default: 10.
        num_states: Number of posture states. Default: 3.
        loss_weight: Weight for action classification loss. Default: 1.0.
        state_loss_weight: Weight for state loss. Default: 0.5.
        fall_loss_weight: Weight for fall detection loss. Default: 2.0.
        fall_focal_alpha: Focal loss alpha (pos class weight). Default: 0.25.
        fall_focal_gamma: Focal loss gamma (hard example mining). Default: 2.0.
        class_weight: Per-class weights for action CE loss. Default: None.
        dropout: Dropout rate. Default: 0.1.
        temporal_residual: Add skip connection. Default: True.
        skip_gru_t1: Skip GRU when T=1. Default: False.
        bone_mode: Input is bone features. Default: True.
        appearance_dim: RoI appearance feature dim (V11). Default: 0.
    """

    def __init__(
        self,
        num_keypoints: int = 6,
        kpt_dim: int = 6,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_gru_layers: int = 1,
        num_classes: int = 10,
        num_states: int = 3,
        loss_weight: float = 1.0,
        state_loss_weight: float = 0.5,
        fall_loss_weight: float = 2.0,
        fall_focal_alpha: float = 0.25,
        fall_focal_gamma: float = 2.0,
        class_weight: Optional[List[float]] = None,
        dropout: float = 0.1,
        temporal_residual: bool = True,
        skip_gru_t1: bool = False,
        bone_mode: bool = True,
        appearance_dim: int = 0,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.kpt_dim = kpt_dim
        self.num_classes = num_classes
        self.num_states = num_states
        self.loss_weight = loss_weight
        self.state_loss_weight = state_loss_weight
        self.fall_loss_weight = fall_loss_weight
        self.fall_focal_alpha = fall_focal_alpha
        self.fall_focal_gamma = fall_focal_gamma
        self.hidden_dim = hidden_dim
        self.temporal_residual = temporal_residual
        self.skip_gru_t1 = skip_gru_t1
        self.bone_mode = bone_mode
        self.appearance_dim = appearance_dim

        input_dim = num_keypoints * kpt_dim + appearance_dim

        # Shared per-frame encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

        # Shared temporal GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
        )
        self.gru_norm = nn.LayerNorm(hidden_dim)

        # Three classification heads
        self.action_classifier = nn.Linear(hidden_dim, num_classes)
        self.state_classifier = nn.Linear(hidden_dim, num_states)
        self.fall_classifier = nn.Linear(hidden_dim, 1)

        # Per-class weights for action CE
        if class_weight is not None:
            self.register_buffer(
                'class_weight',
                torch.tensor(class_weight, dtype=torch.float32)
            )
        else:
            self.class_weight = None

        # Label mapping buffers
        self.register_buffer(
            '_action_to_state',
            torch.tensor(ACTION_TO_STATE, dtype=torch.long)
        )
        self.register_buffer(
            '_action_to_fall',
            torch.tensor(ACTION_TO_FALL, dtype=torch.float32)
        )

    def _encode(self, kpt_features: Tensor) -> Tensor:
        """Shared encoder: (N, T, D) -> (N, hidden_dim)."""
        N, T, D = kpt_features.shape
        encoded = self.encoder(kpt_features.reshape(N * T, D))
        encoded = encoded.reshape(N, T, -1)

        if T == 1 and self.skip_gru_t1:
            out = encoded.squeeze(1)
        else:
            _, h_n = self.gru(encoded)
            gru_out = self.gru_norm(h_n[-1])
            if self.temporal_residual:
                out = encoded[:, -1, :] + gru_out
            else:
                out = gru_out
        return out

    def forward(self, kpt_features: Tensor):
        """Forward pass returning all three head logits.

        Returns:
            action_logits: (N, num_classes)
            state_logits:  (N, num_states)
            fall_logits:   (N, 1)
        """
        h = self._encode(kpt_features)
        return (
            self.action_classifier(h),
            self.state_classifier(h),
            self.fall_classifier(h),
        )

    def loss(self, kpt_features: Tensor, gt_labels: Tensor) -> dict:
        """Calculate multi-head losses.

        Args:
            kpt_features: (N, T, K*D) input features.
            gt_labels: (N,) action class labels (0-9). -1 = invalid.

        Returns:
            dict with 'loss_action', 'loss_state', 'loss_fall'.
        """
        action_logits, state_logits, fall_logits = self.forward(kpt_features)

        valid = gt_labels >= 0
        if not valid.any():
            zero = action_logits.sum() * 0.0
            return {
                'loss_action': zero,
                'loss_state': zero,
                'loss_fall': zero,
            }

        gt_valid = gt_labels[valid].long()

        # Map action_class -> state and fall labels (vectorized)
        gt_state = self._action_to_state[gt_valid]
        gt_fall = self._action_to_fall[gt_valid]

        # 1. Action loss: 10-class CE with class weights
        loss_action = F.cross_entropy(
            action_logits[valid], gt_valid,
            weight=self.class_weight,
        )

        # 2. State loss: 3-class CE
        loss_state = F.cross_entropy(
            state_logits[valid], gt_state,
        )

        # 3. Fall loss: binary focal loss
        loss_fall = self._focal_loss(
            fall_logits.squeeze(-1)[valid], gt_fall,
        )

        return {
            'loss_action': loss_action * self.loss_weight,
            'loss_state': loss_state * self.state_loss_weight,
            'loss_fall': loss_fall * self.fall_loss_weight,
        }

    def _focal_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Binary focal loss for fall detection.

        Focal loss down-weights easy examples and focuses on hard ones.
        This is critical for falls which are rare (~40% of data).
        """
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = (self.fall_focal_alpha * targets
                   + (1 - self.fall_focal_alpha) * (1 - targets))
        focal_weight = alpha_t * (1 - p_t) ** self.fall_focal_gamma
        return (focal_weight * ce).mean()

    def predict(self, kpt_features: Tensor) -> dict:
        """Predict action, state, and fall probabilities.

        Returns:
            dict with:
                'action_probs': (N, 10)
                'action_class': (N,) predicted class IDs
                'is_falling':   (N,) bool (from action_class >= 6)
                'state_probs':  (N, 3) state probabilities
                'state_class':  (N,) predicted state (0=upright, 1=sit, 2=lie)
                'fall_prob':    (N,) fall probability from dedicated head
        """
        action_logits, state_logits, fall_logits = self.forward(kpt_features)

        action_probs = F.softmax(action_logits, dim=-1)
        action_class = action_probs.argmax(dim=-1)
        is_falling = action_class >= 6

        state_probs = F.softmax(state_logits, dim=-1)
        state_class = state_probs.argmax(dim=-1)

        fall_prob = torch.sigmoid(fall_logits.squeeze(-1))

        return {
            'action_probs': action_probs,
            'action_class': action_class,
            'is_falling': is_falling,
            'state_probs': state_probs,
            'state_class': state_class,
            'fall_prob': fall_prob,
        }

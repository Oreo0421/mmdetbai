# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module(force=True)
class KeypointMSELoss(nn.Module):
    """MSE Loss for Keypoint Heatmaps.

    Args:
        use_target_weight (bool): Whether to use per-keypoint weights.
        loss_weight (float): Global loss weight multiplier.
        reduction (str): Loss reduction method ('mean', 'sum', or 'none').
        invisible_weight (float): Loss weight for invisible keypoints (v=0).
            When > 0, invisible keypoints are supervised with GT=0 heatmap,
            forcing the model to predict zero response for unseen keypoints.
            Default: 0.1. Set to 0.0 to fully ignore invisible keypoints
            (original behavior).
    """

    def __init__(
        self,
        use_target_weight: bool = True,
        loss_weight: float = 1.0,
        reduction: str = 'mean',
        invisible_weight: float = 0.3,
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.invisible_weight = invisible_weight
        self.criterion = nn.MSELoss(reduction='none')

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        target_weight: Tensor = None,
    ) -> Tensor:
        """Calculate keypoint heatmap loss.

        Args:
            pred (Tensor): Predicted heatmaps (B, K, H, W)
            target (Tensor): Ground truth heatmaps (B, K, H, W)
            target_weight (Tensor, optional): Per-keypoint weights (B, K)
        """
        B, K, H, W = pred.shape

        loss = self.criterion(pred, target)  # (B, K, H, W)

        if self.use_target_weight and target_weight is not None:
            if target_weight.dim() == 2:  # (B, K)
                target_weight = target_weight.unsqueeze(-1).unsqueeze(-1)
            # 不可见点 (weight=0) 也参与loss，权重为 invisible_weight
            # GT heatmap 已经是全零，MSE 会迫使模型在不可见位置输出 0
            if self.invisible_weight > 0:
                target_weight = torch.where(
                    target_weight > 0,
                    target_weight,
                    torch.full_like(target_weight, self.invisible_weight),
                )
            loss = loss * target_weight

        if self.reduction == 'mean':
            if self.use_target_weight and target_weight is not None:
                loss = loss.sum() / (target_weight.sum() + 1e-6)
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss * self.loss_weight


@MODELS.register_module(force=True)
class KeypointPerPointMSELoss(nn.Module):
    """Per-keypoint averaged MSE Loss.

    V4 loss: 对每个关键点分别计算空间维度上的MSE，再按关键点数K做平均。
    可见点权重=1.0，不可见点权重=invisible_weight。

    loss = (1/K) * Σ_k [ mean_HW(MSE(pred_k, gt_k)) * w_k ]
    其中 w_k = 1.0 (visible) 或 invisible_weight (invisible)

    Args:
        use_target_weight (bool): Whether to use per-keypoint visibility weights.
        loss_weight (float): Global loss weight multiplier.
        invisible_weight (float): Weight for invisible keypoints. Default: 0.3.
    """

    def __init__(
        self,
        use_target_weight: bool = True,
        loss_weight: float = 5.0,
        invisible_weight: float = 0.3,
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.invisible_weight = invisible_weight
        self.criterion = nn.MSELoss(reduction='none')

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        target_weight: Tensor = None,
    ) -> Tensor:
        """Calculate per-keypoint averaged MSE loss.

        Args:
            pred (Tensor): Predicted heatmaps (B, K, H, W)
            target (Tensor): Ground truth heatmaps (B, K, H, W)
            target_weight (Tensor, optional): Per-keypoint visibility (B, K)
                values > 0 = visible, 0 = invisible
        """
        B, K, H, W = pred.shape

        # per-pixel MSE: (B, K, H, W)
        loss = self.criterion(pred, target)

        # 对每个关键点在空间维度上取平均: (B, K)
        per_kpt_loss = loss.mean(dim=(-2, -1))

        # 构建每个关键点的权重: visible=1.0, invisible=invisible_weight
        if self.use_target_weight and target_weight is not None:
            # target_weight: (B, K) or (B, K, 1, 1)
            if target_weight.dim() > 2:
                target_weight = target_weight.squeeze(-1).squeeze(-1)
            # visible -> 1.0, invisible -> invisible_weight
            w = torch.where(
                target_weight > 0,
                torch.ones_like(target_weight),
                torch.full_like(target_weight, self.invisible_weight),
            )
        else:
            w = torch.ones(B, K, device=pred.device, dtype=pred.dtype)

        # 加权后按关键点数K做平均
        # loss = (1/K) * Σ_k(per_kpt_loss_k * w_k), 再对batch平均
        weighted = per_kpt_loss * w  # (B, K)
        loss = weighted.sum(dim=1) / K  # (B,)
        loss = loss.mean()  # scalar

        return loss * self.loss_weight

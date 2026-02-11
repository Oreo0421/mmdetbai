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

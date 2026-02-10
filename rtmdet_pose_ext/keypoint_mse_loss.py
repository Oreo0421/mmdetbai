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
    """

    def __init__(
        self,
        use_target_weight: bool = True,
        loss_weight: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.reduction = reduction
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
            loss = loss * target_weight

        if self.reduction == 'mean':
            if self.use_target_weight and target_weight is not None:
                loss = loss.sum() / (target_weight.sum() + 1e-6)
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss * self.loss_weight

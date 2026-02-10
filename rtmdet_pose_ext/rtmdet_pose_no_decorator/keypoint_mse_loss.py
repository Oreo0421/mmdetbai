# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS


class KeypointMSELoss(nn.Module):
    """MSE Loss for Keypoint Heatmaps.
    
    This loss computes the Mean Squared Error between predicted and
    ground truth heatmaps, with optional per-keypoint weighting.
    
    Args:
        use_target_weight (bool): Whether to use per-keypoint weights.
            If True, invisible/occluded keypoints will have lower weights.
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
        
        # Use MSE loss without reduction first
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
            target_weight (Tensor, optional): Per-keypoint weights (B, K) or (B, K, 1, 1)
                - 0.0: invisible/unlabeled keypoint
                - 0.5: occluded keypoint (v=1 in COCO format)
                - 1.0: visible keypoint (v=2 in COCO format)
        
        Returns:
            Tensor: Computed loss value (scalar if reduction='mean')
        """
        B, K, H, W = pred.shape
        
        # Compute per-pixel MSE
        loss = self.criterion(pred, target)  # (B, K, H, W)
        
        # Apply per-keypoint weighting if enabled
        if self.use_target_weight and target_weight is not None:
            # Reshape target_weight to (B, K, 1, 1) for broadcasting
            if target_weight.dim() == 2:  # (B, K)
                target_weight = target_weight.unsqueeze(-1).unsqueeze(-1)
            elif target_weight.dim() == 4:  # (B, K, H, W)
                pass  # Already correct shape
            
            # Apply weights
            loss = loss * target_weight
        
        # Apply reduction
        if self.reduction == 'mean':
            if self.use_target_weight and target_weight is not None:
                # Weighted mean: divide by sum of weights instead of number of elements
                loss = loss.sum() / (target_weight.sum() + 1e-6)
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass  # Keep per-pixel losses
        
        # Apply global loss weight
        return loss * self.loss_weight


class KeypointOHKMMSELoss(nn.Module):
    """Online Hard Keypoint Mining MSE Loss.
    
    This loss focuses on hard examples by selecting the top-k most difficult
    keypoints per sample based on their loss values.
    
    Useful for handling imbalanced keypoint difficulties (e.g., hands are
    harder to detect than head in thermal images).
    
    Args:
        use_target_weight (bool): Whether to use per-keypoint weights
        loss_weight (float): Global loss weight
        topk (int): Number of hardest keypoints to select per sample
            If topk=7, all keypoints are used (equivalent to standard MSE)
            If topk=5, only the 5 hardest keypoints contribute to loss
    """
    
    def __init__(
        self,
        use_target_weight: bool = True,
        loss_weight: float = 1.0,
        topk: int = 5,
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.topk = topk
        self.criterion = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        target_weight: Tensor = None,
    ) -> Tensor:
        """Calculate OHKM keypoint loss.
        
        Args:
            pred (Tensor): Predicted heatmaps (B, K, H, W)
            target (Tensor): Ground truth heatmaps (B, K, H, W)
            target_weight (Tensor, optional): Per-keypoint weights (B, K)
        
        Returns:
            Tensor: Loss value focusing on hardest keypoints
        """
        B, K, H, W = pred.shape
        
        # Compute per-pixel MSE
        loss = self.criterion(pred, target)  # (B, K, H, W)
        
        # Reduce to per-keypoint loss
        loss_per_kpt = loss.mean(dim=[2, 3])  # (B, K)
        
        # Apply target weights if enabled
        if self.use_target_weight and target_weight is not None:
            loss_per_kpt = loss_per_kpt * target_weight
        
        # Online Hard Keypoint Mining: select top-k hardest keypoints
        topk = min(self.topk, K)
        
        # Get indices of top-k hardest keypoints per sample
        _, topk_indices = loss_per_kpt.topk(topk, dim=1)
        
        # Create selection mask
        mask = torch.zeros_like(loss_per_kpt)
        mask.scatter_(1, topk_indices, 1.0)
        
        # Apply mask
        loss_ohkm = (loss_per_kpt * mask).sum() / (mask.sum() + 1e-6)
        
        return loss_ohkm * self.loss_weight

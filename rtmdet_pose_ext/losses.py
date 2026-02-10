import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class KeypointOHKMMSELoss(nn.Module):
    """Online Hard Keypoint Mining MSE Loss"""
    
    def __init__(self, topk: int = 8, loss_weight: float = 1.0):
        super().__init__()
        self.topk = topk
        self.loss_weight = loss_weight
        self.criterion = nn.MSELoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                target_weight: torch.Tensor = None) -> torch.Tensor:
        B, K, H, W = pred.shape
        
        loss = self.criterion(pred, target)  # (B, K, H, W)
        
        if target_weight is not None:
            target_weight = target_weight.view(B, K, 1, 1)
            loss = loss * target_weight
        
        # Per-keypoint loss
        loss = loss.mean(dim=[2, 3])  # (B, K)
        
        # OHKM: select topk hardest keypoints
        if self.topk < K:
            topk_loss, _ = torch.topk(loss, self.topk, dim=1)
            loss = topk_loss.mean()
        else:
            loss = loss.mean()
        
        return loss * self.loss_weight

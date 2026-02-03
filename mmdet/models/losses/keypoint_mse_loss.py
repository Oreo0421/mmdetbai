# mmdet/models/losses/keypoint_mse_loss.py

import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class KeypointMSELoss(nn.Module):
    """MSE Loss for Keypoint Heatmaps."""
    
    def __init__(self, use_target_weight=True, loss_weight=1.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.criterion = nn.MSELoss(reduction='mean')
    
    def forward(self, pred, target, target_weight=None):
        if self.use_target_weight and target_weight is not None:
            # target_weight: (B, num_keypoints, 1, 1)
            pred = pred * target_weight
            target = target * target_weight
        
        loss = self.criterion(pred, target)
        return loss * self.loss_weight

import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class KeypointMSELoss(nn.Module):
    """Keypoint MSE Loss with proper dimension handling"""
    
    def __init__(self, use_target_weight: bool = True, loss_weight: float = 1.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.criterion = nn.MSELoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                target_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: (B, K, H, W) 预测热图
            target: (B, K, H, W) 目标热图
            target_weight: (B, K) 关键点权重
        Returns:
            loss: scalar
        """
        B, K, H, W = pred.shape
        
        # 计算 MSE loss
        loss = self.criterion(pred, target)  # (B, K, H, W)
        
        # 应用关键点权重
        if self.use_target_weight and target_weight is not None:
            # 扩展 target_weight 维度: (B, K) -> (B, K, 1, 1)
            target_weight = target_weight.view(B, K, 1, 1)
            loss = loss * target_weight
        
        # 计算平均 loss
        loss = loss.mean()
        
        return loss * self.loss_weight

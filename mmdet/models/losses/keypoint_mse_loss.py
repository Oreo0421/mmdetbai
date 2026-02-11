import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class KeypointMSELoss(nn.Module):
    """Keypoint MSE Loss with proper dimension handling.

    Args:
        use_target_weight (bool): Whether to use per-keypoint weights.
        loss_weight (float): Global loss weight multiplier.
        invisible_weight (float): Loss weight for invisible keypoints (v=0).
            Default: 0.3. Set to 0.0 to fully ignore invisible keypoints.
    """

    def __init__(self, use_target_weight: bool = True, loss_weight: float = 1.0,
                 invisible_weight: float = 0.3):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.invisible_weight = invisible_weight
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
            # 不可见点也参与loss，权重为 invisible_weight
            if self.invisible_weight > 0:
                target_weight = torch.where(
                    target_weight > 0,
                    target_weight,
                    torch.full_like(target_weight, self.invisible_weight),
                )
            loss = loss * target_weight

        # 计算平均 loss
        if self.use_target_weight and target_weight is not None:
            loss = loss.sum() / (target_weight.sum() + 1e-6)
        else:
            loss = loss.mean()

        return loss * self.loss_weight

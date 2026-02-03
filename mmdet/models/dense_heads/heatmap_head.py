# mmdet/models/dense_heads/heatmap_head.py

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType


@MODELS.register_module()
class HeatmapHead(nn.Module):
    """Simple Heatmap-based Pose Estimation Head.
    
    Args:
        num_keypoints: 关键点数量 (例如 COCO 17个)
        in_channels: 输入特征通道数
        feat_channels: 中间特征通道数
        loss_keypoint: 关键点损失配置
    """
    
    def __init__(
        self,
        num_keypoints: int = 17,
        in_channels: int = 96,
        feat_channels: int = 128,
        loss_keypoint: ConfigType = dict(
            type='KeypointMSELoss',
            use_target_weight=True,
            loss_weight=1.0,
        ),
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        
        # 简单的卷积层: 只用最高分辨率特征 (P3, stride=8)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        
        # 上采样 + 预测层
        self.deconv = nn.ConvTranspose2d(
            feat_channels, feat_channels, 4, stride=2, padding=1
        )
        self.pred_layer = nn.Conv2d(
            feat_channels, num_keypoints, 1, stride=1, padding=0
        )
        
        # 损失函数
        self.loss_keypoint = MODELS.build(loss_keypoint)
    
    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """前向传播.
        
        Args:
            feats: 来自 neck 的多尺度特征 [P3, P4, P5]
        
        Returns:
            heatmaps: (B, num_keypoints, H, W)
        """
        # 只用 P3 (stride=8, 最高分辨率)
        x = feats[0]
        
        x = self.conv_layers(x)
        x = self.deconv(x)  # 上采样 2x
        heatmaps = self.pred_layer(x)
        
        return heatmaps
    
    def loss(self, feats: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """计算关键点损失."""
        
        heatmaps = self.forward(feats)
        
        # 从 data_samples 提取 GT heatmaps 和 weights
        gt_heatmaps = []
        keypoint_weights = []
        
        for data_sample in batch_data_samples:
            if hasattr(data_sample, 'gt_instances'):
                gt = data_sample.gt_instances
                if hasattr(gt, 'keypoints_heatmap'):
                    gt_heatmaps.append(gt.keypoints_heatmap)
                    keypoint_weights.append(gt.keypoint_weights)
        
        if len(gt_heatmaps) == 0:
            # 没有 pose 标注，返回 0 loss
            return dict(loss_keypoint=heatmaps.sum() * 0.0)
        
        gt_heatmaps = torch.stack(gt_heatmaps)
        keypoint_weights = torch.stack(keypoint_weights)
        
        # 计算损失
        loss_kpt = self.loss_keypoint(
            heatmaps, gt_heatmaps, keypoint_weights
        )
        
        return dict(loss_keypoint=loss_kpt)
    
    def predict(self, feats: Tuple[Tensor], batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """预测关键点."""
        
        heatmaps = self.forward(feats)
        
        # 从 heatmap 提取关键点坐标
        keypoints, scores = self._decode_heatmap(heatmaps)
        
        # 填充到 data_samples
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_instances.keypoints = keypoints[i]
            data_sample.pred_instances.keypoint_scores = scores[i]
        
        return batch_data_samples
    
    def _decode_heatmap(self, heatmaps: Tensor) -> Tuple[Tensor, Tensor]:
        """从 heatmap 解码关键点坐标.
        
        Returns:
            keypoints: (B, num_keypoints, 2) [x, y]
            scores: (B, num_keypoints)
        """
        B, K, H, W = heatmaps.shape
        
        # 找到每个 heatmap 的最大值位置
        heatmaps_flat = heatmaps.view(B, K, -1)
        scores, indices = heatmaps_flat.max(dim=2)
        
        y = (indices // W).float()
        x = (indices % W).float()
        
        keypoints = torch.stack([x, y], dim=2)  # (B, K, 2)
        
        return keypoints, scores

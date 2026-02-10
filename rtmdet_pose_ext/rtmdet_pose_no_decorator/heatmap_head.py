# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType


class HeatmapHead(nn.Module):
    """Heatmap-based Pose Estimation Head for 7-keypoint thermal IR human detection.
    
    Designed for 40x114 thermal images with 7 body keypoints:
    - head, shoulder, hand_right, hand_left, hips, foot_right, foot_left
    
    Architecture:
        feats[0] (P3, stride=8) → Conv → BatchNorm → ReLU
                                 → Conv → BatchNorm → ReLU
                                 → Deconv (upsample 2x)
                                 → Conv1x1 → heatmaps (num_keypoints channels)
    
    Args:
        num_keypoints (int): Number of keypoints (default: 7)
        in_channels (int): Input feature channels from neck
        feat_channels (int): Intermediate feature channels
        deconv_out_channels (int): Output channels of deconv layer
        use_deconv (bool): Whether to use deconvolution for upsampling
        loss_keypoint (dict): Keypoint loss config
    """
    
    def __init__(
        self,
        num_keypoints: int = 7,
        in_channels: int = 96,
        feat_channels: int = 128,
        deconv_out_channels: int = 64,
        use_deconv: bool = True,
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
        self.use_deconv = use_deconv
        
        # Feature refinement layers (only use P3 - highest resolution)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling + prediction
        if use_deconv:
            # Deconvolution for better upsampling (learnable)
            self.deconv = nn.ConvTranspose2d(
                feat_channels, deconv_out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            )
            self.deconv_bn = nn.BatchNorm2d(deconv_out_channels)
            self.deconv_relu = nn.ReLU(inplace=True)
            pred_in_channels = deconv_out_channels
        else:
            # Simple bilinear upsampling
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            pred_in_channels = feat_channels
        
        # Final heatmap prediction layer
        self.pred_layer = nn.Conv2d(
            pred_in_channels, num_keypoints,
            kernel_size=1, stride=1, padding=0
        )
        
        # Loss function
        self.loss_keypoint = MODELS.build(loss_keypoint)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for conv and deconv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
    
    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward pass to generate keypoint heatmaps.
        
        Args:
            feats (tuple): Multi-scale features from neck [P3, P4, P5]
                P3: stride=8  (highest resolution)
                P4: stride=16
                P5: stride=32
        
        Returns:
            Tensor: Predicted heatmaps of shape (B, num_keypoints, H, W)
                For 192x192 input:
                    - P3 input: 24x24 (stride=8)
                    - After deconv 2x: 48x48 (stride=4)
        """
        # Use only P3 (highest resolution feature, stride=8)
        x = feats[0]  # (B, in_channels, H/8, W/8)
        
        # Feature refinement
        x = self.conv_layers(x)  # (B, feat_channels, H/8, W/8)
        
        # Upsampling
        if self.use_deconv:
            x = self.deconv(x)       # (B, deconv_out_channels, H/4, W/4)
            x = self.deconv_bn(x)
            x = self.deconv_relu(x)
        else:
            x = self.upsample(x)     # (B, feat_channels, H/4, W/4)
        
        # Generate heatmaps
        heatmaps = self.pred_layer(x)  # (B, num_keypoints, H/4, W/4)
        
        return heatmaps
    
    def loss(self, feats: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Calculate keypoint heatmap loss.
        
        Args:
            feats (tuple): Multi-scale features from neck
            batch_data_samples (list): GT data samples with keypoint annotations
        
        Returns:
            dict: Loss dictionary
                - loss_keypoint: MSE loss between pred and GT heatmaps
        """
        # Forward pass
        heatmaps = self.forward(feats)
        
        # Extract GT heatmaps and weights from data samples
        gt_heatmaps = []
        keypoint_weights = []
        
        for data_sample in batch_data_samples:
            if hasattr(data_sample, 'gt_instances'):
                gt = data_sample.gt_instances
                if hasattr(gt, 'keypoints_heatmap'):
                    gt_heatmaps.append(gt.keypoints_heatmap)
                    if hasattr(gt, 'keypoint_weights'):
                        keypoint_weights.append(gt.keypoint_weights)
                    else:
                        # Default: all keypoints have weight 1.0
                        keypoint_weights.append(
                            torch.ones(self.num_keypoints, device=heatmaps.device)
                        )
        
        if len(gt_heatmaps) == 0:
            # No pose annotations in this batch
            # Return zero loss (don't break gradient flow)
            return dict(loss_keypoint=heatmaps.sum() * 0.0)
        
        # Stack GT data
        gt_heatmaps = torch.stack(gt_heatmaps).to(heatmaps.device)
        keypoint_weights = torch.stack(keypoint_weights).to(heatmaps.device)
        
        # Resize GT heatmaps to match prediction size if needed
        if gt_heatmaps.shape[-2:] != heatmaps.shape[-2:]:
            gt_heatmaps = F.interpolate(
                gt_heatmaps,
                size=heatmaps.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Calculate loss
        loss_kpt = self.loss_keypoint(
            heatmaps, gt_heatmaps, keypoint_weights
        )
        
        return dict(loss_keypoint=loss_kpt)
    
    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: SampleList,
        rescale: bool = True
    ) -> SampleList:
        """Predict keypoints from heatmaps.
        
        Args:
            feats (tuple): Multi-scale features from neck
            batch_data_samples (list): Data samples
            rescale (bool): Whether to rescale to original image size
        
        Returns:
            list: Updated data samples with predicted keypoints
        """
        # Forward pass
        heatmaps = self.forward(feats)
        
        # Decode heatmaps to keypoint coordinates
        keypoints, scores = self._decode_heatmap(heatmaps)
        
        # Rescale to original image size if needed
        if rescale:
            for i, data_sample in enumerate(batch_data_samples):
                if hasattr(data_sample, 'scale_factor'):
                    scale_factor = data_sample.scale_factor
                    keypoints[i] = keypoints[i] / torch.tensor(
                        scale_factor, device=keypoints.device
                    )
        
        # Fill results into data_samples
        for i, data_sample in enumerate(batch_data_samples):
            if not hasattr(data_sample, 'pred_instances'):
                from mmdet.structures import DetDataSample
                data_sample.pred_instances = DetDataSample().pred_instances
            
            data_sample.pred_instances.keypoints = keypoints[i]
            data_sample.pred_instances.keypoint_scores = scores[i]
        
        return batch_data_samples
    
    def _decode_heatmap(self, heatmaps: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode keypoint coordinates from heatmaps.
        
        Simple argmax decoding: find the location with maximum activation.
        
        Args:
            heatmaps (Tensor): Predicted heatmaps (B, K, H, W)
        
        Returns:
            tuple:
                - keypoints (Tensor): Keypoint coordinates (B, K, 2) in [x, y] format
                - scores (Tensor): Keypoint confidence scores (B, K)
        """
        B, K, H, W = heatmaps.shape
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, K, -1)
        
        # Find maximum values and their indices
        scores, indices = heatmaps_flat.max(dim=2)
        scores = torch.sigmoid(scores)  # Normalize scores to [0, 1]
        
        # Convert flat indices to (x, y) coordinates
        y = (indices // W).float()
        x = (indices % W).float()
        
        # Stack to (B, K, 2)
        keypoints = torch.stack([x, y], dim=2)
        
        # Apply sub-pixel refinement (optional, for better accuracy)
        # keypoints = self._refine_keypoints(heatmaps, keypoints)
        
        return keypoints, scores

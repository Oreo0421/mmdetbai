# -*- coding: utf-8 -*-
"""
CoordinateRegressionHead: Direct coordinate regression pose head.

Drop-in replacement for HeatmapHead. Instead of predicting heatmaps
and decoding via argmax, this head directly regresses normalized
(x, y) coordinates + confidence scores via FC layers.

Architecture:
    RoI features [N, 96, 48, 48]
      → Conv1-4 (3x3, BN, ReLU)
      → AdaptiveAvgPool2d(1)
      → FC1 → FC2
      ├→ fc_coords → sigmoid → [N, K, 2] normalized coords [0,1]
      └→ fc_scores → [N, K] confidence logits
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import InstanceData
from mmengine.logging import MMLogger
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps, get_box_tensor


@MODELS.register_module(force=True)
class CoordinateRegressionHead(nn.Module):
    """Direct coordinate regression head for keypoint estimation.

    Args:
        num_keypoints (int): Number of keypoints. Default: 7.
        in_channels (int): Input feature channels. Default: 96.
        feat_channels (int): Conv feature channels. Default: 128.
        fc_channels (int): FC hidden layer channels. Default: 256.
        loss_keypoint (dict, optional): Ignored, kept for config compatibility.
        coord_loss_type (str): 'smooth_l1' or 'l1'. Default: 'smooth_l1'.
        coord_loss_weight (float): Weight for coordinate loss. Default: 5.0.
        score_loss_weight (float): Weight for score loss. Default: 1.0.
        invisible_weight (float): Loss weight for invisible keypoints. Default: 0.3.
        match_iou_thr (float): IoU threshold for RoI-GT matching. Default: 0.1.
    """

    def __init__(
        self,
        num_keypoints: int = 7,
        in_channels: int = 96,
        feat_channels: int = 128,
        fc_channels: int = 256,
        loss_keypoint: Optional[Dict] = None,
        coord_loss_type: str = 'smooth_l1',
        coord_loss_weight: float = 5.0,
        score_loss_weight: float = 1.0,
        invisible_weight: float = 0.3,
        match_iou_thr: float = 0.1,
        log_stats: bool = False,
        log_interval: int = 20,
        # Ignored args from HeatmapHead for config compatibility
        upsample_factor: int = 2,
        sigma: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.match_iou_thr = match_iou_thr
        self.invisible_weight = invisible_weight
        self.coord_loss_weight = coord_loss_weight
        self.score_loss_weight = score_loss_weight
        self.log_stats = log_stats
        self.log_interval = int(log_interval)
        self._stat_iter = 0
        self._fwd_count = 0

        K = num_keypoints

        # Spatial feature extraction (same as HeatmapHead)
        self.conv1 = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(feat_channels)
        self.conv2 = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(feat_channels)
        self.conv3 = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(feat_channels)
        self.conv4 = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(feat_channels)

        # Global Average Pooling → FC layers → output
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(feat_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, fc_channels)
        self.fc_coords = nn.Linear(fc_channels, K * 2)  # normalized (x, y)
        self.fc_scores = nn.Linear(fc_channels, K)       # confidence logits

        # Loss functions
        if coord_loss_type == 'smooth_l1':
            self.coord_criterion = nn.SmoothL1Loss(reduction='none')
        else:
            self.coord_criterion = nn.L1Loss(reduction='none')
        self.score_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # For config compatibility (not used)
        self.loss_keypoint = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        # 坐标输出初始化为 0.5（RoI 中心），加速收敛
        nn.init.zeros_(self.fc_coords.weight)
        nn.init.constant_(self.fc_coords.bias, 0.0)  # sigmoid(0)=0.5 → 中心
        print(f"[CoordinateRegressionHead] init done: K={self.num_keypoints}")

    def forward(self, feats: Union[Tuple[Tensor, ...], Tensor]) -> Tuple[Tensor, Tensor]:
        """Extract features and predict coordinates + scores.

        Args:
            feats: [N, C, H, W] RoI features or multi-scale tuple

        Returns:
            coords: [N, K, 2] normalized coordinates in [0, 1]
            score_logits: [N, K] raw logits
        """
        if isinstance(feats, (list, tuple)):
            x = feats[0]
        else:
            x = feats

        N = x.size(0)
        K = self.num_keypoints

        # Debug: log first few forward passes
        self._fwd_count += 1
        if self._fwd_count <= 3:
            print(f"[CoordRegHead.forward] call#{self._fwd_count} "
                  f"input={x.shape} N={N} device={x.device}")
            import sys; sys.stdout.flush()

        # Handle N=0 case safely (BN and CUDA can segfault on empty tensors)
        if N == 0:
            device = x.device
            dtype = x.dtype
            empty_coords = torch.zeros((0, K, 2), device=device, dtype=dtype)
            empty_scores = torch.zeros((0, K), device=device, dtype=dtype)
            return empty_coords, empty_scores

        # Spatial feature extraction
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)

        # GAP + FC
        x = self.gap(x).view(N, -1)  # [N, feat_channels]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # Predict
        coords = torch.sigmoid(self.fc_coords(x))  # [N, K*2] → [0, 1]
        coords = coords.view(N, K, 2)
        score_logits = self.fc_scores(x)  # [N, K]

        return coords, score_logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def loss(
        self,
        feats: Union[Tuple[Tensor, ...], Tensor],
        rois_or_samples=None,
        batch_data_samples: Optional[List] = None,
    ) -> Dict[str, Tensor]:
        """Drop-in interface matching HeatmapHead.loss().

        Full-image mode: loss(feats, batch_data_samples)
        RoI mode:        loss(roi_feats, rois, batch_data_samples)
        """
        if torch.is_tensor(rois_or_samples):
            return self._loss_roi(feats, rois_or_samples, batch_data_samples)
        return self._loss_full(feats, rois_or_samples)

    def _loss_roi(
        self,
        roi_feats: Tensor,
        rois: Tensor,
        batch_data_samples: List,
    ) -> Dict[str, Tensor]:
        """RoI-based coordinate regression loss."""
        N = roi_feats.size(0)

        if N == 0:
            zero = roi_feats.sum() * 0
            return {'loss_keypoint': zero}

        coords, score_logits = self.forward(roi_feats)  # [N, K, 2], [N, K]

        gt_coords, gt_vis = self._build_coord_targets(
            rois, batch_data_samples, coords.dtype, coords.device
        )

        # Coordinate loss (SmoothL1)
        coord_loss = self.coord_criterion(coords, gt_coords)  # [N, K, 2]
        coord_loss = coord_loss.sum(dim=2)  # [N, K]

        # Visibility weighting
        vis_weight = torch.where(
            gt_vis > 0,
            torch.ones_like(gt_vis),
            torch.full_like(gt_vis, self.invisible_weight),
        )
        coord_loss = (coord_loss * vis_weight).sum() / (vis_weight.sum() + 1e-6)

        # Score loss (BCE)
        score_target = (gt_vis > 0).float()
        score_loss = self.score_criterion(score_logits, score_target).mean()

        total_loss = (self.coord_loss_weight * coord_loss +
                      self.score_loss_weight * score_loss)

        return {'loss_keypoint': total_loss}

    def _loss_full(
        self,
        feats: Union[Tuple[Tensor, ...], Tensor],
        batch_data_samples: List,
    ) -> Dict[str, Tensor]:
        """Full-image mode loss (entire image as single RoI)."""
        coords, score_logits = self.forward(feats)  # [B, K, 2], [B, K]
        B, K, _ = coords.shape

        gt_coords = torch.zeros_like(coords)
        gt_vis = torch.zeros(B, K, device=coords.device, dtype=coords.dtype)

        for b, ds in enumerate(batch_data_samples):
            meta = getattr(ds, 'metainfo', None)
            img_shape = meta.get('img_shape', None) if meta else None
            if img_shape is None:
                continue
            img_h, img_w = float(img_shape[0]), float(img_shape[1])

            kpts = None
            if hasattr(ds, 'gt_keypoints'):
                kpts = ds.gt_keypoints
            elif hasattr(ds, 'gt_instances') and hasattr(ds.gt_instances, 'keypoints'):
                kpts = ds.gt_instances.keypoints

            if kpts is None:
                continue

            if isinstance(kpts, torch.Tensor):
                kpts = kpts.to(device=coords.device, dtype=coords.dtype)
            else:
                kpts = torch.as_tensor(kpts, device=coords.device, dtype=coords.dtype)

            if kpts.dim() == 3:
                kpts = kpts[0]

            for k in range(min(K, kpts.size(0))):
                x, y, v = float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])
                if v > 0:
                    gt_coords[b, k, 0] = x / img_w
                    gt_coords[b, k, 1] = y / img_h
                    gt_vis[b, k] = 1.0

        coord_loss = self.coord_criterion(coords, gt_coords).sum(dim=2)
        vis_weight = torch.where(
            gt_vis > 0,
            torch.ones_like(gt_vis),
            torch.full_like(gt_vis, self.invisible_weight),
        )
        coord_loss = (coord_loss * vis_weight).sum() / (vis_weight.sum() + 1e-6)

        score_target = (gt_vis > 0).float()
        score_loss = self.score_criterion(score_logits, score_target).mean()

        total_loss = self.coord_loss_weight * coord_loss + self.score_loss_weight * score_loss
        return {'loss_keypoint': total_loss}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        feats: Union[Tuple[Tensor, ...], Tensor],
        rois_or_results,
        batch_data_samples=None,
        rescale: bool = True,
    ):
        """Drop-in interface matching HeatmapHead.predict().

        RoI mode:        predict(roi_feats, rois, ...) → (kpts[N,K,2], scores[N,K])
        Full-image mode: predict(feats, batch_results, ...) → modified batch_results
        """
        coords, score_logits = self.forward(feats)
        scores = torch.sigmoid(score_logits)

        if torch.is_tensor(rois_or_results):
            # ===== RoI mode =====
            rois = rois_or_results
            if rois.numel() == 0:
                return coords, scores

            # Denormalize: [0,1] within RoI → image coords
            x1 = rois[:, 1].unsqueeze(1)
            y1 = rois[:, 2].unsqueeze(1)
            x2 = rois[:, 3].unsqueeze(1)
            y2 = rois[:, 4].unsqueeze(1)
            roi_w = (x2 - x1).clamp(min=1.0)
            roi_h = (y2 - y1).clamp(min=1.0)

            batch_keypoints = coords.clone()
            batch_keypoints[:, :, 0] = coords[:, :, 0] * roi_w + x1
            batch_keypoints[:, :, 1] = coords[:, :, 1] * roi_h + y1

            # Rescale: resized image → original image
            if batch_data_samples is not None:
                for i in range(batch_keypoints.size(0)):
                    img_idx = int(rois[i, 0])
                    if img_idx >= len(batch_data_samples):
                        continue
                    meta = getattr(batch_data_samples[img_idx], 'metainfo', None)
                    if meta is None:
                        continue
                    img_shape = meta.get('img_shape', None)
                    ori_shape = meta.get('ori_shape', None)
                    if img_shape is None or ori_shape is None:
                        continue
                    img_h, img_w = float(img_shape[0]), float(img_shape[1])
                    ori_h, ori_w = float(ori_shape[0]), float(ori_shape[1])
                    if rescale and ori_w > 0 and ori_h > 0:
                        batch_keypoints[i, :, 0] *= (ori_w / img_w)
                        batch_keypoints[i, :, 1] *= (ori_h / img_h)

            return batch_keypoints, scores

        # ===== Full-image mode =====
        batch_results = rois_or_results
        batch_keypoints = coords.clone()

        if batch_data_samples is not None:
            for b, ds in enumerate(batch_data_samples):
                meta = getattr(ds, 'metainfo', None)
                if meta is None:
                    continue
                img_shape = meta.get('img_shape', None)
                ori_shape = meta.get('ori_shape', None)
                if img_shape is None or ori_shape is None:
                    continue
                img_h, img_w = float(img_shape[0]), float(img_shape[1])
                ori_h, ori_w = float(ori_shape[0]), float(ori_shape[1])

                batch_keypoints[b, :, 0] *= img_w
                batch_keypoints[b, :, 1] *= img_h

                if rescale and ori_w > 0 and ori_h > 0:
                    batch_keypoints[b, :, 0] *= (ori_w / img_w)
                    batch_keypoints[b, :, 1] *= (ori_h / img_h)

        # Attach to results
        for i, item in enumerate(batch_results):
            keypoints = batch_keypoints[i]
            kp_scores = scores[i]

            if isinstance(item, InstanceData):
                if len(item) == 0:
                    item.keypoints = keypoints.new_zeros((0, self.num_keypoints, 2))
                    item.keypoint_scores = kp_scores.new_zeros((0, self.num_keypoints))
                    continue
                if hasattr(item, 'scores') and item.scores is not None and len(item.scores) > 0:
                    top1 = int(item.scores.argmax())
                else:
                    top1 = 0
                item = item[top1:top1 + 1]
                batch_results[i] = item
                item.keypoints = keypoints.unsqueeze(0)
                item.keypoint_scores = kp_scores.unsqueeze(0)
                continue

            if hasattr(item, 'pred_instances'):
                pred = item.pred_instances
                if len(pred) == 0:
                    pred.keypoints = keypoints.new_zeros((0, self.num_keypoints, 2))
                    pred.keypoint_scores = kp_scores.new_zeros((0, self.num_keypoints))
                    continue
                if hasattr(pred, 'scores') and pred.scores is not None and len(pred.scores) > 0:
                    top1 = int(pred.scores.argmax())
                else:
                    top1 = 0
                pred = pred[top1:top1 + 1]
                item.pred_instances = pred
                pred.keypoints = keypoints.unsqueeze(0)
                pred.keypoint_scores = kp_scores.unsqueeze(0)
                continue

        return batch_results

    # ------------------------------------------------------------------
    # Target building
    # ------------------------------------------------------------------
    def _build_coord_targets(
        self,
        rois: Tensor,
        batch_data_samples: List,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Build coordinate regression targets from GT keypoints via IoU matching.

        Returns:
            gt_coords: [N, K, 2] normalized coords in [0, 1] within each RoI
            gt_vis: [N, K] visibility flags (1=visible, 0=invisible)
        """
        num_rois = rois.size(0)
        K = self.num_keypoints
        matched = 0

        gt_coords = torch.zeros((num_rois, K, 2), device=device, dtype=dtype)
        gt_vis = torch.zeros((num_rois, K), device=device, dtype=dtype)

        if num_rois == 0:
            return gt_coords, gt_vis

        for i in range(num_rois):
            img_idx = int(rois[i, 0])
            if img_idx >= len(batch_data_samples):
                continue
            ds = batch_data_samples[img_idx]
            gt_instances = getattr(ds, 'gt_instances', None)
            if gt_instances is None or len(gt_instances) == 0:
                continue

            gt_bboxes = getattr(gt_instances, 'bboxes', None)
            if gt_bboxes is None or len(gt_bboxes) == 0:
                continue

            gt_bboxes_t = get_box_tensor(gt_bboxes).to(device=device)
            if gt_bboxes_t.numel() == 0:
                continue

            # IoU matching
            roi_box = rois[i, 1:].unsqueeze(0)
            ious = bbox_overlaps(roi_box, gt_bboxes_t)
            max_iou, gt_idx = ious.max(dim=1)
            if float(max_iou.item()) < float(self.match_iou_thr):
                continue

            # Get GT keypoints
            gt_kpts = None
            if hasattr(gt_instances, 'keypoints'):
                gt_kpts = gt_instances.keypoints
            elif hasattr(ds, 'gt_keypoints'):
                gt_kpts = ds.gt_keypoints

            if gt_kpts is None:
                continue

            if isinstance(gt_kpts, torch.Tensor):
                kpts = gt_kpts.to(device=device, dtype=dtype)
            else:
                kpts = torch.as_tensor(gt_kpts, device=device, dtype=dtype)

            if kpts.dim() == 2:
                kpts = kpts.unsqueeze(0)

            if gt_idx.item() >= kpts.size(0):
                continue

            kpts = kpts[int(gt_idx.item())]  # [K, 3]

            # Map from original coords to resized image coords
            meta = getattr(ds, 'metainfo', None)
            kpts = self._map_keypoints_to_img(kpts, meta)

            if kpts.size(0) > 0 and torch.any(kpts[:, 2] > 0):
                matched += 1

            # Convert to normalized [0, 1] within RoI
            x1, y1, x2, y2 = rois[i, 1:].tolist()
            roi_w = max(x2 - x1, 1.0)
            roi_h = max(y2 - y1, 1.0)

            for k in range(min(K, kpts.size(0))):
                x, y, v = float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])
                if v <= 0:
                    continue

                x_norm = max(0.0, min(1.0, (x - x1) / roi_w))
                y_norm = max(0.0, min(1.0, (y - y1) / roi_h))

                gt_coords[i, k, 0] = x_norm
                gt_coords[i, k, 1] = y_norm
                gt_vis[i, k] = 1.0

        # Logging
        self._stat_iter += 1
        if self.log_stats:
            interval = max(1, self.log_interval)
            if self._stat_iter % interval == 0:
                logger = MMLogger.get_current_instance()
                logger.info(f"pose_roi_stats: total={num_rois} matched={matched}")

        return gt_coords, gt_vis

    def _map_keypoints_to_img(self, kpts: Tensor, meta: Optional[dict]) -> Tensor:
        """Map keypoints from original image coords to resized/flipped coords.
        Identical to HeatmapHead._map_keypoints_to_img.
        """
        if meta is None:
            return kpts

        scale_factor = meta.get('scale_factor', None)
        if scale_factor is None:
            return kpts

        if torch.is_tensor(scale_factor):
            scale_factor = scale_factor.detach().cpu().numpy()

        if isinstance(scale_factor, np.ndarray):
            scale_factor = scale_factor.tolist()

        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) >= 2:
                w_scale = float(scale_factor[0])
                h_scale = float(scale_factor[1])
            else:
                w_scale = h_scale = float(scale_factor[0])
        else:
            w_scale = h_scale = float(scale_factor)

        kpts = kpts.clone()
        kpts[..., 0] = kpts[..., 0] * w_scale
        kpts[..., 1] = kpts[..., 1] * h_scale

        if meta.get('flip', False):
            flip_dir = meta.get('flip_direction', 'horizontal')
            img_shape = meta.get('img_shape', None)
            if img_shape is not None:
                img_h, img_w = float(img_shape[0]), float(img_shape[1])
                if flip_dir == 'horizontal':
                    kpts[..., 0] = img_w - 1.0 - kpts[..., 0]
                    for left, right in [(2, 3), (5, 6)]:
                        if kpts.size(0) > max(left, right):
                            kpts[[left, right]] = kpts[[right, left]].clone()
                elif flip_dir == 'vertical':
                    kpts[..., 1] = img_h - 1.0 - kpts[..., 1]
                elif flip_dir == 'diagonal':
                    kpts[..., 0] = img_w - 1.0 - kpts[..., 0]
                    kpts[..., 1] = img_h - 1.0 - kpts[..., 1]
                    for left, right in [(2, 3), (5, 6)]:
                        if kpts.size(0) > max(left, right):
                            kpts[[left, right]] = kpts[[right, left]].clone()

        return kpts

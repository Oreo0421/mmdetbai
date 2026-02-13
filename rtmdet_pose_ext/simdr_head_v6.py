# -*- coding: utf-8 -*-
"""
HeatmapHead V6: SimDR (Simple Disentangled Representation) + visibility.

Based on ECCV 2022 Oral: "Is 2D Heatmap Representation Even Necessary?"
Key idea: replace 2D heatmap with two 1D classification vectors (x, y).

Changes from V4:
1. No 2D heatmap, no deconv upsampling
2. Disentangled 1D x/y prediction via axis-pooling + 1D conv heads
3. KL divergence loss with 1D Gaussian targets (better than MSE)
4. Sub-pixel precision via simdr_scale factor (k=2 → 96 bins from 48 features)
5. Keep visibility branch (BCE) with 3-level weights

Architecture:
  shared 4×Conv(128) → pool_H → x_head(1D conv+deconv) → x_logits (N,K,96)
                      → pool_W → y_head(1D conv+deconv) → y_logits (N,K,96)
                      → GAP → FC → vis_logits (N,K)
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.logging import MMLogger
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps, get_box_tensor

from .heatmap_head import HeatmapHead


@MODELS.register_module(force=True)
class HeatmapHeadV6(HeatmapHead):
    """SimDR-based keypoint head with visibility prediction."""

    def __init__(
        self,
        num_keypoints: int = 7,
        in_channels: int = 96,
        feat_channels: int = 128,
        simdr_scale: int = 2,
        sigma_1d: float = 1.5,
        simdr_loss_weight: float = 5.0,
        vis_loss_weight: float = 1.0,
        occluded_weight: float = 0.3,
        match_iou_thr: float = 0.1,
        log_stats: bool = False,
        log_interval: int = 20,
        loss_keypoint: Optional[Dict] = None,  # ignored
    ):
        super().__init__(
            num_keypoints=num_keypoints,
            in_channels=in_channels,
            feat_channels=feat_channels,
            loss_keypoint=None,
            upsample_factor=1,   # no 2D upsampling
            sigma=1.0,           # unused
            match_iou_thr=match_iou_thr,
            log_stats=log_stats,
            log_interval=log_interval,
        )
        self.simdr_scale = simdr_scale
        self.sigma_1d = sigma_1d
        self.simdr_loss_weight = simdr_loss_weight
        self.vis_loss_weight = vis_loss_weight
        self.occluded_weight = occluded_weight

        # --- SimDR x/y 1D heads ---
        # Each: Conv1d(3×1) + BN + ReLU [+ ConvTranspose1d for upsampling] + Conv1d(1×1)
        self.x_head = self._build_1d_head(feat_channels, num_keypoints, simdr_scale)
        self.y_head = self._build_1d_head(feat_channels, num_keypoints, simdr_scale)

        # --- Visibility branch (same as V4) ---
        self.vis_gap = nn.AdaptiveAvgPool2d(1)
        self.vis_fc = nn.Linear(feat_channels, num_keypoints)

    @staticmethod
    def _build_1d_head(in_ch, out_ch, scale):
        layers = [
            nn.Conv1d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
        ]
        factor = scale
        while factor > 1:
            layers.extend([
                nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1),
                nn.BatchNorm1d(in_ch),
                nn.ReLU(inplace=True),
            ])
            factor //= 2
        layers.append(nn.Conv1d(in_ch, out_ch, 1))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _extract_shared(self, feats):
        if isinstance(feats, (list, tuple)):
            x = feats[0]
        else:
            x = feats
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, feats) -> Tuple[Tensor, Tensor, Tensor]:
        """Return (x_logits [N,K,Lx], y_logits [N,K,Ly], vis_logits [N,K])."""
        shared = self._extract_shared(feats)  # (N, C, H, W)

        # SimDR: axis-pooling + 1D heads
        x_feat = shared.mean(dim=2)   # (N, C, W) pool over height
        y_feat = shared.mean(dim=3)   # (N, C, H) pool over width

        x_logits = self.x_head(x_feat)  # (N, K, W*scale)
        y_logits = self.y_head(y_feat)  # (N, K, H*scale)

        # Visibility
        vis_feat = self.vis_gap(shared).flatten(1)
        vis_logits = self.vis_fc(vis_feat)

        return x_logits, y_logits, vis_logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def loss(self, feats, rois_or_samples=None, batch_data_samples=None):
        if torch.is_tensor(rois_or_samples):
            return self._loss_roi_v6(feats, rois_or_samples, batch_data_samples)
        return {}

    def _loss_roi_v6(self, roi_feats, rois, batch_data_samples):
        x_logits, y_logits, vis_logits = self.forward(roi_feats)
        N, K, Lx = x_logits.shape
        Ly = y_logits.shape[-1]
        device = x_logits.device
        dtype = x_logits.dtype

        gt_x_dist, gt_y_dist, loc_weights, vis_labels = self._build_targets_v6(
            rois, batch_data_samples, Lx, Ly, dtype, device
        )

        # ---- 1. SimDR: KL divergence ----
        x_log_prob = F.log_softmax(x_logits, dim=-1)
        y_log_prob = F.log_softmax(y_logits, dim=-1)

        # KL(target || pred) summed over bins → per-keypoint scalar
        kl_x = F.kl_div(x_log_prob, gt_x_dist, reduction='none').sum(dim=-1)  # (N, K)
        kl_y = F.kl_div(y_log_prob, gt_y_dist, reduction='none').sum(dim=-1)  # (N, K)

        kl_per_kpt = (kl_x + kl_y) * loc_weights
        loss_simdr = (kl_per_kpt.sum(dim=1) / K).mean()

        # ---- 2. Visibility BCE ----
        loss_vis = F.binary_cross_entropy_with_logits(
            vis_logits, vis_labels, reduction='mean'
        )

        # ---- Total ----
        loss_kpt = self.simdr_loss_weight * loss_simdr + self.vis_loss_weight * loss_vis

        # Logging
        self._stat_iter += 1
        if self.log_stats and self._stat_iter % max(1, self.log_interval) == 0:
            logger = MMLogger.get_current_instance()
            logger.info(
                f"v6_loss: simdr={loss_simdr.item():.4f} "
                f"vis={loss_vis.item():.4f} "
                f"kl_x={kl_x.mean().item():.4f} kl_y={kl_y.mean().item():.4f}"
            )

        return {"loss_keypoint": loss_kpt}

    # ------------------------------------------------------------------
    # Target building
    # ------------------------------------------------------------------
    def _build_targets_v6(self, rois, batch_data_samples, Lx, Ly, dtype, device):
        """Build 1D Gaussian targets for SimDR.

        Returns:
            gt_x_dist  (Tensor): (N, K, Lx) probability distributions
            gt_y_dist  (Tensor): (N, K, Ly) probability distributions
            loc_weights (Tensor): (N, K)
            vis_labels  (Tensor): (N, K)
        """
        num_rois = rois.size(0)
        K = self.num_keypoints
        feat_w = Lx // self.simdr_scale  # 48
        feat_h = Ly // self.simdr_scale  # 48
        matched = 0

        # Default: uniform distribution (no training signal)
        gt_x_dist = torch.full((num_rois, K, Lx), 1.0 / Lx, device=device, dtype=dtype)
        gt_y_dist = torch.full((num_rois, K, Ly), 1.0 / Ly, device=device, dtype=dtype)
        loc_weights = torch.zeros((num_rois, K), device=device, dtype=dtype)
        vis_labels = torch.zeros((num_rois, K), device=device, dtype=dtype)

        if num_rois == 0:
            return gt_x_dist, gt_y_dist, loc_weights, vis_labels

        for i in range(num_rois):
            img_idx = int(rois[i, 0])
            if img_idx >= len(batch_data_samples):
                continue
            ds = batch_data_samples[img_idx]
            gt_instances = getattr(ds, "gt_instances", None)
            if gt_instances is None or len(gt_instances) == 0:
                continue

            gt_bboxes = getattr(gt_instances, "bboxes", None)
            if gt_bboxes is None or len(gt_bboxes) == 0:
                continue

            gt_bboxes_t = get_box_tensor(gt_bboxes).to(device=device)
            if gt_bboxes_t.numel() == 0:
                continue

            roi_box = rois[i, 1:].unsqueeze(0)
            ious = bbox_overlaps(roi_box, gt_bboxes_t)
            max_iou, gt_idx = ious.max(dim=1)
            if float(max_iou.item()) < float(self.match_iou_thr):
                continue

            gt_kpts = None
            if hasattr(gt_instances, "keypoints"):
                gt_kpts = gt_instances.keypoints
            elif hasattr(ds, "gt_keypoints"):
                gt_kpts = ds.gt_keypoints
            if gt_kpts is None:
                continue

            if isinstance(gt_kpts, Tensor):
                kpts = gt_kpts.to(device=device, dtype=dtype)
            else:
                kpts = torch.as_tensor(gt_kpts, device=device, dtype=dtype)
            if kpts.dim() == 2:
                kpts = kpts.unsqueeze(0)
            if gt_idx.item() >= kpts.size(0):
                continue

            kpts = kpts[int(gt_idx.item())]
            meta = getattr(ds, "metainfo", None)
            kpts = self._map_keypoints_to_img(kpts, meta)

            if kpts.size(0) > 0 and torch.any(kpts[:, 2] > 0):
                matched += 1

            x1, y1, x2, y2 = rois[i, 1:].tolist()
            roi_w = max(x2 - x1, 1.0)
            roi_h = max(y2 - y1, 1.0)

            for k in range(min(K, kpts.size(0))):
                x, y, v = float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])
                if v == 0:
                    continue

                # Coordinate in SimDR bin space
                bin_x = (x - x1) * feat_w / roi_w * self.simdr_scale
                bin_y = (y - y1) * feat_h / roi_h * self.simdr_scale

                if 0 <= bin_x < Lx and 0 <= bin_y < Ly:
                    gt_x_dist[i, k] = self._gaussian_1d(Lx, bin_x, device, dtype)
                    gt_y_dist[i, k] = self._gaussian_1d(Ly, bin_y, device, dtype)

                    if v == 1:
                        loc_weights[i, k] = self.occluded_weight
                        vis_labels[i, k] = 0.0
                    elif v >= 2:
                        loc_weights[i, k] = 1.0
                        vis_labels[i, k] = 1.0

        # Logging
        if self.log_stats and self._stat_iter % max(1, self.log_interval) == 0:
            logger = MMLogger.get_current_instance()
            logger.info(f"pose_roi_stats: total={num_rois} matched={matched}")

        return gt_x_dist, gt_y_dist, loc_weights, vis_labels

    def _gaussian_1d(self, length, center, device, dtype):
        """Normalized 1D Gaussian distribution."""
        x = torch.arange(length, device=device, dtype=dtype)
        g = torch.exp(-(x - center) ** 2 / (2 * self.sigma_1d ** 2))
        return g / (g.sum() + 1e-8)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, feats, rois_or_results, batch_data_samples=None, rescale=True):
        x_logits, y_logits, vis_logits = self.forward(feats)

        # Decode: argmax → coordinate in feature space
        pred_x = x_logits.argmax(dim=-1).float() / self.simdr_scale  # (N, K)
        pred_y = y_logits.argmax(dim=-1).float() / self.simdr_scale
        batch_scores = torch.sigmoid(vis_logits)  # (N, K)

        feat_w = x_logits.size(-1) // self.simdr_scale  # 48
        feat_h = y_logits.size(-1) // self.simdr_scale

        # --- ROI-based prediction ---
        if torch.is_tensor(rois_or_results):
            rois = rois_or_results
            if rois.numel() == 0:
                batch_keypoints = torch.stack([pred_x, pred_y], dim=2)
                return batch_keypoints, batch_scores

            x1 = rois[:, 1].unsqueeze(1)
            y1 = rois[:, 2].unsqueeze(1)
            x2 = rois[:, 3].unsqueeze(1)
            y2 = rois[:, 4].unsqueeze(1)
            roi_w = (x2 - x1).clamp(min=1.0)
            roi_h = (y2 - y1).clamp(min=1.0)

            # Feature space → image space
            img_x = pred_x * roi_w / feat_w + x1
            img_y = pred_y * roi_h / feat_h + y1

            batch_keypoints = torch.stack([img_x, img_y], dim=2)  # (N, K, 2)

            # Rescale to original image
            if batch_data_samples is not None:
                for i in range(batch_keypoints.size(0)):
                    img_idx = int(rois[i, 0])
                    if img_idx >= len(batch_data_samples):
                        continue
                    meta = getattr(batch_data_samples[img_idx], "metainfo", None)
                    if meta is None:
                        continue
                    img_shape = meta.get("img_shape", None)
                    ori_shape = meta.get("ori_shape", None)
                    if img_shape is None or ori_shape is None:
                        continue
                    img_h, img_w = float(img_shape[0]), float(img_shape[1])
                    ori_h, ori_w = float(ori_shape[0]), float(ori_shape[1])
                    if rescale and ori_w > 0 and ori_h > 0:
                        batch_keypoints[i, :, 0] *= (ori_w / img_w)
                        batch_keypoints[i, :, 1] *= (ori_h / img_h)

            return batch_keypoints, batch_scores

        # --- Full-image fallback (attach to result instances) ---
        from mmengine.structures import InstanceData
        batch_results = rois_or_results

        for i, item in enumerate(batch_results):
            kpts = torch.stack([pred_x[i], pred_y[i]], dim=1)  # (K, 2)
            scores = batch_scores[i]  # (K,)

            target = None
            if isinstance(item, InstanceData):
                target = item
            elif hasattr(item, "pred_instances"):
                target = item.pred_instances

            if target is not None:
                if len(target) == 0:
                    target.keypoints = kpts.new_zeros((0, self.num_keypoints, 2))
                    target.keypoint_scores = scores.new_zeros((0, self.num_keypoints))
                else:
                    target.keypoints = kpts.unsqueeze(0)
                    target.keypoint_scores = scores.unsqueeze(0)

        return batch_results

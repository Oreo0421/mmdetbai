# -*- coding: utf-8 -*-
"""
HeatmapHead V5: V4 + soft-argmax coordinate regression loss.

Changes from V4:
1. Soft-argmax: differentiable coordinate extraction from heatmap
2. SmoothL1 coordinate loss: directly optimizes keypoint position accuracy
3. Three-loss total:
     loss_kpt = loc_w * MSE_heatmap + vis_w * BCE_vis + coord_w * SmoothL1_coord
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmengine.logging import MMLogger
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps, get_box_tensor

from .heatmap_head_v4 import HeatmapHeadV4


@MODELS.register_module(force=True)
class HeatmapHeadV5(HeatmapHeadV4):
    """HeatmapHeadV4 + soft-argmax coordinate regression loss.

    The soft-argmax converts heatmap to differentiable coordinates,
    then applies SmoothL1 loss against GT coordinates in heatmap space.
    This directly optimizes the metric we care about (coordinate accuracy).
    """

    def __init__(self, coord_loss_weight: float = 2.0, beta: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.coord_loss_weight = coord_loss_weight
        self.beta = beta  # soft-argmax temperature (higher = sharper)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def loss(self, feats, rois_or_samples=None, batch_data_samples=None):
        if torch.is_tensor(rois_or_samples):
            return self._loss_roi_v5(feats, rois_or_samples, batch_data_samples)
        return self._loss_full_v4(feats, rois_or_samples)

    def _loss_roi_v5(self, roi_feats, rois, batch_data_samples):
        heatmaps, vis_logits = self.forward(roi_feats)
        N, K, H, W = heatmaps.shape
        device = heatmaps.device
        dtype = heatmaps.dtype

        gt_heatmaps, loc_weights, vis_labels, gt_coords = self._build_targets_v5(
            rois, batch_data_samples, (H, W), dtype, device
        )

        # ---- 1. Foreground-weighted MSE (same as V4) ----
        mse = F.mse_loss(heatmaps, gt_heatmaps, reduction='none')
        fg_mask = (gt_heatmaps > 0.01).float()
        pixel_w = torch.where(
            fg_mask > 0,
            torch.full_like(fg_mask, self.fg_weight),
            torch.ones_like(fg_mask),
        )
        per_kpt_mse = (mse * pixel_w).mean(dim=(-2, -1))
        per_kpt_mse = per_kpt_mse * loc_weights
        loss_loc = (per_kpt_mse.sum(dim=1) / K).mean()

        # ---- 2. Visibility BCE (same as V4) ----
        loss_vis = F.binary_cross_entropy_with_logits(
            vis_logits, vis_labels, reduction='mean'
        )

        # ---- 3. Soft-argmax coordinate loss (NEW) ----
        # Build coordinate grids
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij',
        )
        grid_x = grid_x.reshape(1, 1, -1)  # (1, 1, H*W)
        grid_y = grid_y.reshape(1, 1, -1)

        # Soft-argmax: heatmap -> probability -> expected coordinate
        hm_prob = F.softmax(heatmaps.view(N, K, -1) * self.beta, dim=-1)
        pred_x = (hm_prob * grid_x).sum(-1)  # (N, K)
        pred_y = (hm_prob * grid_y).sum(-1)  # (N, K)

        # Normalize to [0, 1] for scale-invariant loss
        denom_w = max(W - 1, 1)
        denom_h = max(H - 1, 1)
        loss_x = F.smooth_l1_loss(
            pred_x / denom_w, gt_coords[:, :, 0] / denom_w, reduction='none')
        loss_y = F.smooth_l1_loss(
            pred_y / denom_h, gt_coords[:, :, 1] / denom_h, reduction='none')

        # Apply same visibility weighting as heatmap loss
        coord_per_kpt = (loss_x + loss_y) * loc_weights
        loss_coord = (coord_per_kpt.sum(dim=1) / K).mean()

        # ---- Total ----
        loss_kpt = (self.loc_loss_weight * loss_loc
                    + self.vis_loss_weight * loss_vis
                    + self.coord_loss_weight * loss_coord)

        # Logging
        self._stat_iter += 1
        if self.log_stats and self._stat_iter % max(1, self.log_interval) == 0:
            logger = MMLogger.get_current_instance()
            logger.info(
                f"v5_loss: loc={loss_loc.item():.4f} "
                f"vis={loss_vis.item():.4f} "
                f"coord={loss_coord.item():.4f}"
            )

        return {"loss_keypoint": loss_kpt}

    # ------------------------------------------------------------------
    # Target building (extends V4 with gt_coords)
    # ------------------------------------------------------------------
    def _build_targets_v5(self, rois, batch_data_samples, heatmap_size, dtype, device):
        """Build targets with 3-level visibility weights + GT coordinates.

        Returns:
            gt_heatmaps (Tensor): (N, K, H, W)
            loc_weights (Tensor): (N, K)
            vis_labels  (Tensor): (N, K)
            gt_coords   (Tensor): (N, K, 2) float coords in heatmap space
        """
        num_rois = rois.size(0)
        K = self.num_keypoints
        H, W = heatmap_size
        matched = 0

        gt_heatmaps = torch.zeros((num_rois, K, H, W), device=device, dtype=dtype)
        loc_weights = torch.zeros((num_rois, K), device=device, dtype=dtype)
        vis_labels = torch.zeros((num_rois, K), device=device, dtype=dtype)
        gt_coords = torch.zeros((num_rois, K, 2), device=device, dtype=dtype)

        if num_rois == 0:
            return gt_heatmaps, loc_weights, vis_labels, gt_coords

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

            if kpts.size(0) > 0:
                vis_any = kpts[:, 2] > 0
                if torch.any(vis_any):
                    matched += 1

            x1, y1, x2, y2 = rois[i, 1:].tolist()
            roi_w = max(x2 - x1, 1.0)
            roi_h = max(y2 - y1, 1.0)

            for k in range(min(K, kpts.size(0))):
                x, y, v = float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])

                if v == 0:
                    continue

                # Floating-point coords in heatmap space (exact, no quantization)
                hm_x = (x - x1) * W / roi_w
                hm_y = (y - y1) * H / roi_h
                hm_x_int = int(round(hm_x))
                hm_y_int = int(round(hm_y))

                if v == 1:
                    if 0 <= hm_x_int < W and 0 <= hm_y_int < H:
                        gt_heatmaps[i, k] = self._generate_gaussian(
                            H, W, hm_x_int, hm_y_int, device=device, dtype=dtype)
                        loc_weights[i, k] = self.occluded_weight
                        # Store exact float coords (clipped to valid range)
                        gt_coords[i, k, 0] = max(0.0, min(hm_x, W - 1.0))
                        gt_coords[i, k, 1] = max(0.0, min(hm_y, H - 1.0))
                    vis_labels[i, k] = 0.0

                elif v >= 2:
                    if 0 <= hm_x_int < W and 0 <= hm_y_int < H:
                        gt_heatmaps[i, k] = self._generate_gaussian(
                            H, W, hm_x_int, hm_y_int, device=device, dtype=dtype)
                        loc_weights[i, k] = 1.0
                        gt_coords[i, k, 0] = max(0.0, min(hm_x, W - 1.0))
                        gt_coords[i, k, 1] = max(0.0, min(hm_y, H - 1.0))
                    vis_labels[i, k] = 1.0

        return gt_heatmaps, loc_weights, vis_labels, gt_coords

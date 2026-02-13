# -*- coding: utf-8 -*-
"""
HeatmapHead V4: visibility prediction + foreground-weighted loss.

Changes from HeatmapHead:
1. Visibility branch: GAP → FC → sigmoid, supervised by BCE
2. Foreground-weighted MSE: peak pixels ×fg_weight, background ×1
3. Per-keypoint averaged loss with 3-level weight:
   - v=0 (out-of-frame): loc_weight=0, vis_target=0
   - v=1 (occluded):     loc_weight=occluded_weight, vis_target=0
   - v=2 (visible):      loc_weight=1.0, vis_target=1
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import InstanceData
from mmengine.logging import MMLogger
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps, get_box_tensor

from .heatmap_head import HeatmapHead


@MODELS.register_module(force=True)
class HeatmapHeadV4(HeatmapHead):
    """HeatmapHead with visibility prediction and foreground-weighted loss."""

    def __init__(
        self,
        num_keypoints: int = 7,
        in_channels: int = 96,
        feat_channels: int = 128,
        upsample_factor: int = 2,
        sigma: float = 2.0,
        match_iou_thr: float = 0.1,
        log_stats: bool = False,
        log_interval: int = 20,
        # V4 loss params
        loc_loss_weight: float = 5.0,
        vis_loss_weight: float = 1.0,
        fg_weight: float = 10.0,
        occluded_weight: float = 0.3,
        loss_keypoint: Optional[Dict] = None,  # ignored, kept for compat
    ):
        super().__init__(
            num_keypoints=num_keypoints,
            in_channels=in_channels,
            feat_channels=feat_channels,
            loss_keypoint=None,  # V4 computes loss internally
            upsample_factor=upsample_factor,
            sigma=sigma,
            match_iou_thr=match_iou_thr,
            log_stats=log_stats,
            log_interval=log_interval,
        )
        self.loc_loss_weight = loc_loss_weight
        self.vis_loss_weight = vis_loss_weight
        self.fg_weight = fg_weight
        self.occluded_weight = occluded_weight
        self._feat_channels = feat_channels

        # Visibility prediction branch: shared conv feats → GAP → FC → K logits
        self.vis_gap = nn.AdaptiveAvgPool2d(1)
        self.vis_fc = nn.Linear(feat_channels, num_keypoints)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _extract_shared(self, feats):
        """Conv feature extraction (shared by heatmap and visibility)."""
        if isinstance(feats, (list, tuple)):
            x = feats[0]
        else:
            x = feats
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, feats) -> Tuple[Tensor, Tensor]:
        """Return (heatmaps [B,K,H',W'], vis_logits [B,K])."""
        shared = self._extract_shared(feats)

        # Heatmap branch
        hm = self.deconv(shared) if self.deconv is not None else shared
        heatmaps = self.pred_layer(hm)

        # Visibility branch
        vis_feat = self.vis_gap(shared).flatten(1)  # [B, C]
        vis_logits = self.vis_fc(vis_feat)  # [B, K]

        return heatmaps, vis_logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def loss(self, feats, rois_or_samples=None, batch_data_samples=None):
        if torch.is_tensor(rois_or_samples):
            return self._loss_roi_v4(feats, rois_or_samples, batch_data_samples)
        return self._loss_full_v4(feats, rois_or_samples)

    def _loss_roi_v4(self, roi_feats, rois, batch_data_samples):
        heatmaps, vis_logits = self.forward(roi_feats)
        N, K, H, W = heatmaps.shape

        gt_heatmaps, loc_weights, vis_labels = self._build_targets_v4(
            rois, batch_data_samples, (H, W), heatmaps.dtype, heatmaps.device
        )

        # ---- 1. Foreground-weighted MSE (定位) ----
        #  mse = (pred_heatmap - gt_heatmap)²        # 逐像素差                                                                                                                                                             
        #前景像素 (peak区域) × 10，背景像素 × 1       # 解决91%背景稀释问题                                                                                                                                               
        #按7个点平均，再乘以可见性权重：
        # ifv=0 (out of image): × 0    → 不参与
        # v=1 (covaer): × 0.3  → 低权重学习
        # v=2 (visiable): × 1.0  → 全权重
        mse = F.mse_loss(heatmaps, gt_heatmaps, reduction='none')  # (N,K,H,W)

        # 前景 (peak) 像素加权，背景权重=1
        fg_mask = (gt_heatmaps > 0.01).float()
        pixel_w = torch.where(
            fg_mask > 0,
            torch.full_like(fg_mask, self.fg_weight),
            torch.ones_like(fg_mask),
        )
        weighted_mse = mse * pixel_w  # (N, K, H, W)

        # 按关键点空间平均 → (N, K)
        per_kpt_mse = weighted_mse.mean(dim=(-2, -1))
        # 乘以定位权重 (v=0→0, v=1→0.3, v=2→1.0)
        per_kpt_mse = per_kpt_mse * loc_weights
        # 按 K 平均 → (N,), 再 batch 平均 → scalar
        loss_loc = (per_kpt_mse.sum(dim=1) / K).mean()

        # ---- 2. Visibility BCE (可见性分类) ----
        #  vis_branch: 共享conv特征 → GAP(全局平均池化) → FC → 7个logit
        #  GT label:  v=2 → 1 (可见)，v=0/v=1 → 0 (不可见)
        # loss_vis = BCE(pred_logit, gt_label)
        loss_vis = F.binary_cross_entropy_with_logits(
            vis_logits, vis_labels, reduction='mean'
        )

        # ---- Total ----
        loss_kpt = self.loc_loss_weight * loss_loc + self.vis_loss_weight * loss_vis

        return {"loss_keypoint": loss_kpt}

    def _loss_full_v4(self, feats, batch_data_samples):
        """Full-image mode (fallback, not typically used with ROI extractor)."""
        heatmaps, vis_logits = self.forward(feats)
        B, K, H, W = heatmaps.shape

        gt_heatmaps = []
        loc_weights_list = []
        vis_labels_list = []

        for ds in (batch_data_samples or []):
            # GT heatmap
            if hasattr(ds, "gt_keypoints_heatmap"):
                hm = ds.gt_keypoints_heatmap
            else:
                hm = heatmaps.new_zeros((K, H, W))
            if isinstance(hm, Tensor):
                hm_t = hm.to(device=heatmaps.device, dtype=heatmaps.dtype)
            else:
                hm_t = heatmaps.new_tensor(hm)
            if hm_t.shape[-2:] != (H, W):
                hm_t = F.interpolate(
                    hm_t.unsqueeze(0), size=(H, W), mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            gt_heatmaps.append(hm_t)

            # Visibility from keypoints
            if hasattr(ds, "gt_keypoints"):
                kpts = ds.gt_keypoints
                if isinstance(kpts, Tensor) and kpts.numel() >= K * 3:
                    v = kpts[:, 2].to(device=heatmaps.device, dtype=heatmaps.dtype)
                    lw = torch.where(v >= 2, torch.ones_like(v),
                          torch.where(v >= 1,
                                      torch.full_like(v, self.occluded_weight),
                                      torch.zeros_like(v)))
                    vl = (v >= 2).float()
                else:
                    lw = heatmaps.new_ones((K,))
                    vl = heatmaps.new_ones((K,))
            else:
                lw = heatmaps.new_ones((K,))
                vl = heatmaps.new_ones((K,))
            loc_weights_list.append(lw)
            vis_labels_list.append(vl)

        if not gt_heatmaps:
            return {}

        gt_heatmaps = torch.stack(gt_heatmaps, dim=0)
        loc_weights = torch.stack(loc_weights_list, dim=0)
        vis_labels = torch.stack(vis_labels_list, dim=0)

        # Same loss computation
        mse = F.mse_loss(heatmaps, gt_heatmaps, reduction='none')
        fg_mask = (gt_heatmaps > 0.01).float()
        pixel_w = torch.where(fg_mask > 0,
                              torch.full_like(fg_mask, self.fg_weight),
                              torch.ones_like(fg_mask))
        per_kpt_mse = (mse * pixel_w).mean(dim=(-2, -1))
        per_kpt_mse = per_kpt_mse * loc_weights
        loss_loc = (per_kpt_mse.sum(dim=1) / K).mean()

        loss_vis = F.binary_cross_entropy_with_logits(
            vis_logits, vis_labels, reduction='mean')

        loss_kpt = self.loc_loss_weight * loss_loc + self.vis_loss_weight * loss_vis
        return {"loss_keypoint": loss_kpt}

    # ------------------------------------------------------------------
    # Target building (3-level visibility)
    # ------------------------------------------------------------------
    def _build_targets_v4(self, rois, batch_data_samples, heatmap_size, dtype, device):
        """Build targets with 3-level visibility weights.

        Returns:
            gt_heatmaps (Tensor): (N, K, H, W)
            loc_weights (Tensor): (N, K)  定位权重
            vis_labels  (Tensor): (N, K)  可见性 label (0/1)
        """
        num_rois = rois.size(0)
        K = self.num_keypoints
        H, W = heatmap_size
        matched = 0

        gt_heatmaps = torch.zeros((num_rois, K, H, W), device=device, dtype=dtype)
        loc_weights = torch.zeros((num_rois, K), device=device, dtype=dtype)
        vis_labels = torch.zeros((num_rois, K), device=device, dtype=dtype)

        if num_rois == 0:
            return gt_heatmaps, loc_weights, vis_labels

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
                    # 图外/未标注: loc=0, vis=0 (已初始化为0)
                    continue

                hm_x = (x - x1) * W / roi_w
                hm_y = (y - y1) * H / roi_h
                hm_x_int = int(round(hm_x))
                hm_y_int = int(round(hm_y))

                if v == 1:
                    # 遮挡: 生成heatmap(如果在ROI内), 低定位权重, vis=0
                    if 0 <= hm_x_int < W and 0 <= hm_y_int < H:
                        gt_heatmaps[i, k] = self._generate_gaussian(
                            H, W, hm_x_int, hm_y_int, device=device, dtype=dtype)
                        loc_weights[i, k] = self.occluded_weight
                    vis_labels[i, k] = 0.0

                elif v >= 2:
                    # 可见: 生成heatmap, 全定位权重, vis=1
                    if 0 <= hm_x_int < W and 0 <= hm_y_int < H:
                        gt_heatmaps[i, k] = self._generate_gaussian(
                            H, W, hm_x_int, hm_y_int, device=device, dtype=dtype)
                        loc_weights[i, k] = 1.0
                    vis_labels[i, k] = 1.0

        # Logging
        self._stat_iter += 1
        if self.log_stats:
            interval = max(1, self.log_interval)
            if self._stat_iter % interval == 0:
                logger = MMLogger.get_current_instance()
                logger.info(f"pose_roi_stats: total={num_rois} matched={matched}")

        return gt_heatmaps, loc_weights, vis_labels

    # ------------------------------------------------------------------
    # Inference (use visibility scores)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, feats, rois_or_results, batch_data_samples=None, rescale=True):
        heatmaps, vis_logits = self.forward(feats)
        batch_keypoints, _ = self._decode_heatmap(heatmaps)

        # 用 visibility 分支的 sigmoid 输出作为 keypoint score
        batch_scores = torch.sigmoid(vis_logits)  # [N, K]

        hm_h, hm_w = heatmaps.shape[-2], heatmaps.shape[-1]

        # --- ROI-based prediction ---
        if torch.is_tensor(rois_or_results):
            rois = rois_or_results
            if rois.numel() == 0:
                return batch_keypoints, batch_scores

            x1 = rois[:, 1].unsqueeze(1)
            y1 = rois[:, 2].unsqueeze(1)
            x2 = rois[:, 3].unsqueeze(1)
            y2 = rois[:, 4].unsqueeze(1)
            roi_w = (x2 - x1).clamp(min=1.0)
            roi_h = (y2 - y1).clamp(min=1.0)

            batch_keypoints[:, :, 0] = \
                (batch_keypoints[:, :, 0] + 0.5) * (roi_w / hm_w) - 0.5 + x1
            batch_keypoints[:, :, 1] = \
                (batch_keypoints[:, :, 1] + 0.5) * (roi_h / hm_h) - 0.5 + y1

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

        # --- Full-image prediction ---
        batch_results = rois_or_results

        if batch_data_samples is not None:
            for b, ds in enumerate(batch_data_samples):
                meta = getattr(ds, "metainfo", None)
                if meta is None:
                    continue
                img_shape = meta.get("img_shape", None)
                ori_shape = meta.get("ori_shape", None)
                if img_shape is None or ori_shape is None:
                    continue
                img_h, img_w = float(img_shape[0]), float(img_shape[1])
                ori_h, ori_w = float(ori_shape[0]), float(ori_shape[1])
                batch_keypoints[b, :, 0] = \
                    (batch_keypoints[b, :, 0] + 0.5) * (img_w / hm_w) - 0.5
                batch_keypoints[b, :, 1] = \
                    (batch_keypoints[b, :, 1] + 0.5) * (img_h / hm_h) - 0.5
                if rescale and ori_w > 0 and ori_h > 0:
                    batch_keypoints[b, :, 0] *= (ori_w / img_w)
                    batch_keypoints[b, :, 1] *= (ori_h / img_h)

        for i, item in enumerate(batch_results):
            keypoints = batch_keypoints[i]
            scores = batch_scores[i]

            if isinstance(item, InstanceData):
                if len(item) == 0:
                    item.keypoints = keypoints.new_zeros(
                        (0, self.num_keypoints, 2))
                    item.keypoint_scores = scores.new_zeros(
                        (0, self.num_keypoints))
                    continue
                if hasattr(item, "scores") and item.scores is not None \
                        and len(item.scores) > 0:
                    top1 = int(item.scores.argmax())
                else:
                    top1 = 0
                item = item[top1:top1 + 1]
                batch_results[i] = item
                item.keypoints = keypoints.unsqueeze(0)
                item.keypoint_scores = scores.unsqueeze(0)
                continue

            if hasattr(item, "pred_instances"):
                pred = item.pred_instances
                if len(pred) == 0:
                    pred.keypoints = keypoints.new_zeros(
                        (0, self.num_keypoints, 2))
                    pred.keypoint_scores = scores.new_zeros(
                        (0, self.num_keypoints))
                    continue
                if hasattr(pred, "scores") and pred.scores is not None \
                        and len(pred.scores) > 0:
                    top1 = int(pred.scores.argmax())
                else:
                    top1 = 0
                pred = pred[top1:top1 + 1]
                item.pred_instances = pred
                pred.keypoints = keypoints.unsqueeze(0)
                pred.keypoint_scores = scores.unsqueeze(0)
                continue

        return batch_results

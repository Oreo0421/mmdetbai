# -*- coding: utf-8 -*-
"""
Improved HeatmapHead with:
  1) Deconv upsampling: 24x24 → 48x48 output (halves quantization error)
  2) Sub-pixel refinement: shift argmax by neighbor difference (Dark-Pose style)
  3) Sigmoid scores: proper [0,1] range for keypoint_scores
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps, get_box_tensor


@MODELS.register_module(force=True)
class HeatmapHead(nn.Module):
    """
    Heatmap-based keypoint head with deconv upsampling.

    Input:  feats[0] from FPN, typically stride-8 (e.g. 24x24 for 192x192 img)
    Output: heatmaps at 2x resolution (e.g. 48x48) via one deconv layer

    Predict flow:
      heatmap argmax → sub-pixel refine → heatmap coords → img coords → original coords
    """

    def __init__(
        self,
        num_keypoints: int = 7,
        in_channels: int = 96,
        feat_channels: int = 128,
        loss_keypoint: Optional[Dict] = None,
        upsample_factor: int = 2,      # 上采样倍率，1=不上采样，2=deconv 2x
        sigma: float = 2.0,
        match_iou_thr: float = 0.1,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.upsample_factor = upsample_factor
        self.sigma = sigma
        self.match_iou_thr = match_iou_thr

        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feat_channels)
        self.conv2 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feat_channels)

        # Upsampling via deconv (24x24 → 48x48)
        if upsample_factor > 1:
            self.deconv = nn.ConvTranspose2d(
                feat_channels, feat_channels,
                kernel_size=4, stride=2, padding=1,  # exactly 2x upsample
            )
            self.bn_deconv = nn.BatchNorm2d(feat_channels)
        else:
            self.deconv = None

        # Final prediction
        self.pred_layer = nn.Conv2d(feat_channels, num_keypoints, kernel_size=1)

        # Loss
        self.loss_keypoint = MODELS.build(loss_keypoint) if loss_keypoint is not None else None

    def forward(self, feats: Union[Tuple[Tensor, ...], Tensor]) -> Tensor:
        """Return heatmaps [B, K, H', W'] where H'=H*upsample_factor."""
        if isinstance(feats, (list, tuple)):
            x = feats[0]  # [B, C, H, W], e.g. [B, 96, 24, 24]
        else:
            x = feats
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        if self.deconv is not None:
            x = F.relu(self.bn_deconv(self.deconv(x)), inplace=True)
            # Now x is [B, feat_channels, 48, 48]

        heatmaps = self.pred_layer(x)  # [B, K, 48, 48]
        return heatmaps

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def loss(
        self,
        feats: Union[Tuple[Tensor, ...], Tensor],
        rois_or_samples=None,
        batch_data_samples: Optional[List] = None,
    ) -> Dict[str, Tensor]:
        """Support both full-image and RoI-based pose losses.

        Full-image mode:
            loss(feats, batch_data_samples)
        RoI mode:
            loss(roi_feats, rois, batch_data_samples)
        """
        if torch.is_tensor(rois_or_samples):
            return self._loss_roi(feats, rois_or_samples, batch_data_samples)
        return self._loss_full(feats, rois_or_samples)

    def _loss_full(
        self,
        feats: Union[Tuple[Tensor, ...], Tensor],
        batch_data_samples: List,
    ) -> Dict[str, Tensor]:
        heatmaps = self.forward(feats)  # [B, K, H, W]
        B, K, H, W = heatmaps.shape

        if self.loss_keypoint is None:
            return {}

        gt_heatmaps = []
        target_weights = []

        for ds in batch_data_samples:
            if hasattr(ds, "gt_keypoints_heatmap"):
                hm = ds.gt_keypoints_heatmap
            else:
                hm = heatmaps.new_zeros((K, H, W))

            if isinstance(hm, torch.Tensor):
                hm_t = hm.to(device=heatmaps.device, dtype=heatmaps.dtype)
            else:
                hm_t = heatmaps.new_tensor(hm, dtype=heatmaps.dtype, device=heatmaps.device)

            # 如果 GT 热图尺寸和 pred 不匹配，用双线性插值对齐
            if hm_t.shape[-2:] != (H, W):
                hm_t = F.interpolate(
                    hm_t.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
                ).squeeze(0)

            gt_heatmaps.append(hm_t)

            # Target weight from visibility
            if hasattr(ds, "gt_keypoints"):
                kpts = ds.gt_keypoints
                if isinstance(kpts, torch.Tensor) and kpts.numel() >= K * 3:
                    v = kpts[:, 2].to(device=heatmaps.device, dtype=heatmaps.dtype)
                    w = (v > 0).to(dtype=heatmaps.dtype)
                else:
                    w = heatmaps.new_ones((K,))
            else:
                w = heatmaps.new_ones((K,))
            target_weights.append(w)

        gt_heatmaps = torch.stack(gt_heatmaps, dim=0)
        target_weights = torch.stack(target_weights, dim=0)

        try:
            loss_kpt = self.loss_keypoint(heatmaps, gt_heatmaps, target_weights)
        except TypeError:
            loss_kpt = self.loss_keypoint(heatmaps, gt_heatmaps)

        return {"loss_keypoint": loss_kpt}

    def _loss_roi(
        self,
        roi_feats: Tensor,
        rois: Tensor,
        batch_data_samples: List,
    ) -> Dict[str, Tensor]:
        heatmaps = self.forward(roi_feats)  # [N, K, H, W]
        N, K, H, W = heatmaps.shape

        if self.loss_keypoint is None:
            return {}

        gt_heatmaps, target_weights = self._build_targets_from_rois(
            rois, batch_data_samples, (H, W), heatmaps.dtype, heatmaps.device
        )

        try:
            loss_kpt = self.loss_keypoint(heatmaps, gt_heatmaps, target_weights)
        except TypeError:
            loss_kpt = self.loss_keypoint(heatmaps, gt_heatmaps)

        return {"loss_keypoint": loss_kpt}

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
        """Support both full-image and RoI-based pose prediction.

        Full-image mode:
            predict(feats, batch_results, batch_data_samples)
        RoI mode:
            predict(roi_feats, rois, batch_data_samples) -> (kpts, scores)
        """
        heatmaps = self.forward(feats)  # [B, K, H, W] or [N, K, H, W]
        batch_keypoints, batch_scores = self._decode_heatmap(heatmaps)

        hm_h, hm_w = heatmaps.shape[-2], heatmaps.shape[-1]

        if torch.is_tensor(rois_or_results):
            rois = rois_or_results
            if rois.numel() == 0:
                return batch_keypoints, batch_scores

            # heatmap -> roi coords
            x1 = rois[:, 1].unsqueeze(1)
            y1 = rois[:, 2].unsqueeze(1)
            x2 = rois[:, 3].unsqueeze(1)
            y2 = rois[:, 4].unsqueeze(1)
            roi_w = (x2 - x1).clamp(min=1.0)
            roi_h = (y2 - y1).clamp(min=1.0)

            batch_keypoints[:, :, 0] = (batch_keypoints[:, :, 0] + 0.5) * (roi_w / hm_w) - 0.5 + x1
            batch_keypoints[:, :, 1] = (batch_keypoints[:, :, 1] + 0.5) * (roi_h / hm_h) - 0.5 + y1

            # resized -> original
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

        batch_results = rois_or_results

        # ----- coordinate restore -----
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

                # heatmap → resized image coords
                batch_keypoints[b, :, 0] = (batch_keypoints[b, :, 0] + 0.5) * (img_w / hm_w) - 0.5
                batch_keypoints[b, :, 1] = (batch_keypoints[b, :, 1] + 0.5) * (img_h / hm_h) - 0.5

                # resized → original
                if rescale and ori_w > 0 and ori_h > 0:
                    batch_keypoints[b, :, 0] *= (ori_w / img_w)
                    batch_keypoints[b, :, 1] *= (ori_h / img_h)

        # ----- attach to results -----
        for i, item in enumerate(batch_results):
            keypoints = batch_keypoints[i]
            scores = batch_scores[i]

            if isinstance(item, InstanceData):
                if len(item) == 0:
                    item.keypoints = keypoints.new_zeros((0, self.num_keypoints, 2))
                    item.keypoint_scores = scores.new_zeros((0, self.num_keypoints))
                    continue

                if hasattr(item, "scores") and item.scores is not None and len(item.scores) > 0:
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
                    pred.keypoints = keypoints.new_zeros((0, self.num_keypoints, 2))
                    pred.keypoint_scores = scores.new_zeros((0, self.num_keypoints))
                    continue

                if hasattr(pred, "scores") and pred.scores is not None and len(pred.scores) > 0:
                    top1 = int(pred.scores.argmax())
                else:
                    top1 = 0

                pred = pred[top1:top1 + 1]
                item.pred_instances = pred
                pred.keypoints = keypoints.unsqueeze(0)
                pred.keypoint_scores = scores.unsqueeze(0)
                continue

        return batch_results

    def _build_targets_from_rois(
        self,
        rois: Tensor,
        batch_data_samples: List,
        heatmap_size: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Build ROI heatmap targets from GT keypoints using IoU matching."""
        num_rois = rois.size(0)
        K = self.num_keypoints
        H, W = heatmap_size

        gt_heatmaps = torch.zeros((num_rois, K, H, W), device=device, dtype=dtype)
        target_weights = torch.zeros((num_rois, K), device=device, dtype=dtype)

        if num_rois == 0:
            return gt_heatmaps, target_weights

        for i in range(num_rois):
            img_idx = int(rois[i, 0])
            if img_idx >= len(batch_data_samples):
                continue
            ds = batch_data_samples[img_idx]
            meta = getattr(ds, "metainfo", None)
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

            if isinstance(gt_kpts, torch.Tensor):
                kpts = gt_kpts.to(device=device, dtype=dtype)
            else:
                kpts = torch.as_tensor(gt_kpts, device=device, dtype=dtype)

            if kpts.dim() == 2:
                kpts = kpts.unsqueeze(0)

            if gt_idx.item() >= kpts.size(0):
                continue

            kpts = kpts[int(gt_idx.item())]
            kpts = self._map_keypoints_to_img(kpts, meta)

            x1, y1, x2, y2 = rois[i, 1:].tolist()
            roi_w = max(x2 - x1, 1.0)
            roi_h = max(y2 - y1, 1.0)

            for k in range(min(K, kpts.size(0))):
                x, y, v = float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])
                if v <= 0:
                    continue

                hm_x = (x - x1) * W / roi_w
                hm_y = (y - y1) * H / roi_h
                hm_x_int = int(round(hm_x))
                hm_y_int = int(round(hm_y))

                if 0 <= hm_x_int < W and 0 <= hm_y_int < H:
                    gt_heatmaps[i, k] = self._generate_gaussian(
                        H, W, hm_x_int, hm_y_int, device=device, dtype=dtype
                    )
                    target_weights[i, k] = 1.0

        return gt_heatmaps, target_weights

    def _map_keypoints_to_img(self, kpts: Tensor, meta: Optional[dict]) -> Tensor:
        """Map keypoints from original image coords to resized/flipped coords."""
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
                elif flip_dir == 'vertical':
                    kpts[..., 1] = img_h - 1.0 - kpts[..., 1]
                elif flip_dir == 'diagonal':
                    kpts[..., 0] = img_w - 1.0 - kpts[..., 0]
                    kpts[..., 1] = img_h - 1.0 - kpts[..., 1]

        return kpts

    def _generate_gaussian(
        self,
        h: int,
        w: int,
        cx: int,
        cy: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Generate a 2D Gaussian heatmap."""
        x = torch.arange(0, w, device=device, dtype=dtype)
        y = torch.arange(0, h, device=device, dtype=dtype).unsqueeze(1)
        return torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * self.sigma ** 2))

    def _decode_heatmap(self, heatmaps: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decode with argmax + sub-pixel refinement (Dark-Pose style).

        Sub-pixel: look at neighbors of argmax, shift by
          dx = 0.25 * sign(right - left)
          dy = 0.25 * sign(bottom - top)
        This reduces quantization error by ~50%.
        """
        B, K, H, W = heatmaps.shape
        hm = heatmaps.detach().clone()

        hm_flat = hm.view(B, K, -1)
        maxvals, idx = torch.max(hm_flat, dim=2)

        x = (idx % W).to(dtype=hm.dtype)
        y = (idx // W).to(dtype=hm.dtype)

        # --- Sub-pixel refinement ---
        xi = x.long()
        yi = y.long()

        for b in range(B):
            for k in range(K):
                px, py = int(xi[b, k]), int(yi[b, k])

                # X-direction refinement
                if 0 < px < W - 1:
                    left = hm[b, k, py, px - 1]
                    right = hm[b, k, py, px + 1]
                    diff = right - left
                    if diff.abs() > 1e-6:
                        x[b, k] += 0.25 * diff.sign()

                # Y-direction refinement
                if 0 < py < H - 1:
                    top = hm[b, k, py - 1, px]
                    bottom = hm[b, k, py + 1, px]
                    diff = bottom - top
                    if diff.abs() > 1e-6:
                        y[b, k] += 0.25 * diff.sign()

        keypoints = torch.stack([x, y], dim=2)

        # Sigmoid scores → proper [0, 1] range
        scores = torch.sigmoid(maxvals)

        return keypoints, scores

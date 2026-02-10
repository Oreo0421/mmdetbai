# -*- coding: utf-8 -*-
"""
Improved HeatmapHead with:
  1) Deconv upsampling: 24x24 → 48x48 output (halves quantization error)
  2) Sub-pixel refinement: shift argmax by neighbor difference (Dark-Pose style)
  3) Sigmoid scores: proper [0,1] range for keypoint_scores
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet.registry import MODELS


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
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.upsample_factor = upsample_factor

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

    def forward(self, feats: Tuple[Tensor, ...]) -> Tensor:
        """Return heatmaps [B, K, H', W'] where H'=H*upsample_factor."""
        x = feats[0]  # [B, C, H, W], e.g. [B, 96, 24, 24]
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
    def loss(self, feats: Tuple[Tensor, ...], batch_data_samples: List) -> Dict[str, Tensor]:
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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        feats: Tuple[Tensor, ...],
        batch_results,
        batch_data_samples=None,
        rescale: bool = True,
    ):
        heatmaps = self.forward(feats)  # [B, K, H, W]
        batch_keypoints, batch_scores = self._decode_heatmap(heatmaps)

        hm_h, hm_w = heatmaps.shape[-2], heatmaps.shape[-1]

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

# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.structures.bbox import get_box_tensor


@MODELS.register_module(force=True)
class RTMDetWithPose(SingleStageDetector):
    """RTMDet with 7-keypoint Pose Estimation Head for thermal IR human detection.
    
    Architecture:
        Image → Backbone → Neck → [P3, P4, P5]
                                    ├─→ bbox_head (RTMDetHead)
                                    └─→ pose_head (HeatmapHead)
    
    Args:
        backbone (dict): Backbone config (e.g., MobileViT)
        neck (dict): Neck config (e.g., CSPNeXtPAFPN)
        bbox_head (dict): Detection head config (RTMDetHead)
        pose_head (dict): Pose estimation head config (HeatmapHead)
        train_cfg (dict, optional): Training config
        test_cfg (dict, optional): Testing config
        data_preprocessor (dict, optional): Data preprocessor config
        init_cfg (dict, optional): Initialization config
    """
    
    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        pose_head: ConfigType,
        pose_roi_extractor: OptConfigType = None,
        pose_det_cfg: OptConfigType = None,
        pose_topk: int = 1,
        pose_min_box_size: float = 2.0,
        pose_use_gt_box: bool = True,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        # Build pose estimation head
        self.pose_head = MODELS.build(pose_head)

        # Optional: RoI-based pose head (top-down)
        self.pose_roi_extractor = (
            MODELS.build(pose_roi_extractor)
            if pose_roi_extractor is not None else None
        )
        self.pose_det_cfg = pose_det_cfg
        self.pose_topk = pose_topk
        self.pose_min_box_size = float(pose_min_box_size)
        self.pose_use_gt_box = bool(pose_use_gt_box)
    
    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> dict:
        """Calculate detection + pose estimation losses.
        
        Args:
            batch_inputs (Tensor): Images of shape (B, C, H, W)
            batch_data_samples (list): Ground truth data samples
        
        Returns:
            dict: Dictionary of losses from both tasks
                - loss_cls: Classification loss
                - loss_bbox: Bounding box regression loss
                - loss_keypoint: Keypoint heatmap loss
        """
        # Extract multi-scale features
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # 1. Detection losses (bbox + classification)
        if self.pose_roi_extractor is None or self.pose_use_gt_box:
            det_losses = self.bbox_head.loss(x, batch_data_samples)
            det_results = None
        else:
            # Also get det predictions to build pose RoIs (if enabled)
            det_losses, det_results = self.bbox_head.loss_and_predict(
                x, batch_data_samples, proposal_cfg=self.pose_det_cfg
            )
        losses.update(det_losses)

        # 2. Pose estimation losses
        if self.pose_roi_extractor is None:
            # Shared-feature pose (old behavior)
            pose_losses = self.pose_head.loss(x, batch_data_samples)
        else:
            if self.pose_use_gt_box:
                rois = self._build_gt_rois(batch_data_samples)
            else:
                rois, _ = self._build_pose_rois(
                    det_results, batch_data_samples, topk=self.pose_topk
                )
            if rois.numel() == 0:
                # no dets -> no pose loss
                zero = x[0].sum() * 0
                pose_losses = {"loss_keypoint": zero}
            else:
                roi_feats = self.pose_roi_extractor(x, rois)
                pose_losses = self.pose_head.loss(
                    roi_feats, rois, batch_data_samples
                )

        losses.update(pose_losses)

        return losses
    
    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True
    ) -> SampleList:
        """Predict both bboxes and keypoints.
        
        Args:
            batch_inputs (Tensor): Images of shape (B, C, H, W)
            batch_data_samples (list): Data samples
            rescale (bool): Whether to rescale results to original image size
        
        Returns:
            list: Prediction results with both bboxes and keypoints
        """
        x = self.extract_feat(batch_inputs)

        if self.pose_roi_extractor is None:
            # Detection prediction
            results_list = self.bbox_head.predict(
                x, batch_data_samples, rescale=rescale
            )

            # Pose prediction
            results_list = self.pose_head.predict(
                x, results_list, batch_data_samples, rescale=rescale
            )

            return self.add_pred_to_datasample(batch_data_samples, results_list)

        # --------- RoI-based pose path ---------
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self.bbox_head(x)
        cfg = self.pose_det_cfg if self.pose_det_cfg is not None else self.test_cfg

        # det results for pose RoIs (rescale=False for feature alignment)
        det_results_pose = self.bbox_head.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=cfg, rescale=False
        )

        # final det results for output (rescale as requested)
        det_results_out = self.bbox_head.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=cfg, rescale=rescale
        )

        rois, roi_map = self._build_pose_rois(
            det_results_pose, batch_data_samples, topk=self.pose_topk
        )
        if rois.numel() == 0:
            self._attach_empty_pose(det_results_out)
            return self.add_pred_to_datasample(batch_data_samples, det_results_out)

        roi_feats = self.pose_roi_extractor(x, rois)
        pose_kpts, pose_scores = self.pose_head.predict(
            roi_feats, rois, batch_data_samples, rescale=rescale
        )
        self._attach_pose(det_results_out, roi_map, pose_kpts, pose_scores)
        return self.add_pred_to_datasample(batch_data_samples, det_results_out)

    def _build_pose_rois(self, det_results, batch_data_samples=None, topk: int = 1):
        """Build RoIs from det results and keep mapping to instances."""
        rois = []
        roi_map = []  # list of (img_idx, inst_idx)

        for img_idx, inst in enumerate(det_results):
            if inst is None or len(inst) == 0:
                continue

            bboxes = inst.bboxes
            if bboxes is None or bboxes.numel() == 0:
                continue

            meta = None
            if batch_data_samples is not None and img_idx < len(batch_data_samples):
                meta = getattr(batch_data_samples[img_idx], "metainfo", None)

            if hasattr(inst, 'scores') and inst.scores is not None:
                order = torch.argsort(inst.scores, descending=True)
            else:
                order = torch.arange(len(inst), device=bboxes.device)

            if topk is not None and topk > 0:
                order = order[:topk]

            bboxes = bboxes[order]
            if bboxes.numel() == 0:
                continue

            # clamp to image bounds if available
            if meta is not None:
                img_shape = meta.get("img_shape", None)
                if img_shape is not None:
                    h, w = float(img_shape[0]), float(img_shape[1])
                    bboxes = bboxes.clone()
                    bboxes[:, 0].clamp_(0, w - 1.0)
                    bboxes[:, 2].clamp_(0, w - 1.0)
                    bboxes[:, 1].clamp_(0, h - 1.0)
                    bboxes[:, 3].clamp_(0, h - 1.0)

            # filter invalid boxes
            valid = torch.isfinite(bboxes).all(dim=1)
            ws = bboxes[:, 2] - bboxes[:, 0]
            hs = bboxes[:, 3] - bboxes[:, 1]
            valid = valid & (ws >= self.pose_min_box_size) & (hs >= self.pose_min_box_size)
            if valid.any():
                bboxes = bboxes[valid]
                order = order[valid]
            else:
                continue

            if bboxes.numel() == 0:
                continue

            batch_inds = bboxes.new_full((bboxes.size(0), 1), img_idx)
            rois.append(torch.cat([batch_inds, bboxes], dim=1))

            for idx in order.tolist():
                roi_map.append((img_idx, int(idx)))

        if len(rois) == 0:
            device = None
            for inst in det_results:
                if inst is not None and len(inst) > 0:
                    device = inst.bboxes.device
                    break
            if device is None:
                device = torch.device('cpu')
            empty = torch.zeros((0, 5), device=device)
            return empty, roi_map

        rois = torch.cat(rois, dim=0)
        return rois, roi_map

    def _build_gt_rois(self, batch_data_samples):
        """Build RoIs from GT boxes for training pose head."""
        rois = []
        for img_idx, ds in enumerate(batch_data_samples):
            gt_instances = getattr(ds, "gt_instances", None)
            if gt_instances is None or len(gt_instances) == 0:
                continue

            bboxes = getattr(gt_instances, "bboxes", None)
            if bboxes is None or len(bboxes) == 0:
                continue

            bboxes = get_box_tensor(bboxes)
            bboxes = bboxes.to(dtype=torch.float32)
            if bboxes.numel() == 0:
                continue

            meta = getattr(ds, "metainfo", None)
            if meta is not None:
                img_shape = meta.get("img_shape", None)
                if img_shape is not None:
                    h, w = float(img_shape[0]), float(img_shape[1])
                    bboxes = bboxes.clone()
                    bboxes[:, 0].clamp_(0, w - 1.0)
                    bboxes[:, 2].clamp_(0, w - 1.0)
                    bboxes[:, 1].clamp_(0, h - 1.0)
                    bboxes[:, 3].clamp_(0, h - 1.0)

            valid = torch.isfinite(bboxes).all(dim=1)
            ws = bboxes[:, 2] - bboxes[:, 0]
            hs = bboxes[:, 3] - bboxes[:, 1]
            valid = valid & (ws >= self.pose_min_box_size) & (hs >= self.pose_min_box_size)
            if not valid.any():
                continue

            bboxes = bboxes[valid]
            batch_inds = bboxes.new_full((bboxes.size(0), 1), img_idx)
            rois.append(torch.cat([batch_inds, bboxes], dim=1))

        if len(rois) == 0:
            device = None
            if len(batch_data_samples) > 0:
                device = batch_data_samples[0].gt_instances.bboxes.device
            if device is None:
                device = torch.device('cpu')
            return torch.zeros((0, 5), device=device)

        return torch.cat(rois, dim=0)

    def _attach_empty_pose(self, det_results):
        """Attach empty keypoints to det results when no RoIs exist."""
        num_kpts = getattr(self.pose_head, 'num_keypoints', 0)
        for inst in det_results:
            if inst is None or len(inst) == 0:
                continue
            device = inst.bboxes.device
            inst.keypoints = torch.zeros((len(inst), num_kpts, 2), device=device)
            inst.keypoint_scores = torch.zeros((len(inst), num_kpts), device=device)

    def _attach_pose(self, det_results, roi_map, keypoints, keypoint_scores):
        """Attach predicted keypoints to detection results by mapping."""
        num_kpts = keypoints.size(1) if keypoints.numel() > 0 else 0
        for inst in det_results:
            if inst is None or len(inst) == 0:
                continue
            device = inst.bboxes.device
            inst.keypoints = torch.zeros((len(inst), num_kpts, 2), device=device)
            inst.keypoint_scores = torch.zeros((len(inst), num_kpts), device=device)

        for i, (img_idx, inst_idx) in enumerate(roi_map):
            if img_idx >= len(det_results):
                continue
            inst = det_results[img_idx]
            if inst is None or inst_idx >= len(inst):
                continue
            inst.keypoints[inst_idx] = keypoints[i]
            inst.keypoint_scores[inst_idx] = keypoint_scores[i]

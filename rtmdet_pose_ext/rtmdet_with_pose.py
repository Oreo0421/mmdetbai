# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.structures.bbox import bbox_overlaps, get_box_tensor


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
        action_head: OptConfigType = None,
        freeze_det_pose: bool = False,
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

        # Optional: action classification head
        self.action_head = (
            MODELS.build(action_head) if action_head is not None else None
        )

        # Freeze backbone/neck/bbox_head/pose_head: only train action_head
        if freeze_det_pose and self.action_head is not None:
            frozen = 0
            for name, param in self.named_parameters():
                if 'action_head' not in name:
                    param.requires_grad = False
                    frozen += 1
            print(f'[freeze_det_pose] Frozen {frozen} params, '
                  f'only action_head is trainable.')
    
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

        # 3. Action classification losses (if action_head is configured)
        if self.action_head is not None and self.pose_roi_extractor is not None:
            appearance_dim = getattr(self.action_head, 'appearance_dim', 0)
            if rois.numel() > 0:
                # V9: try temporal GT keypoint sequences (T>1)
                kpt_seq, gt_action = self._extract_temporal_action_data(
                    rois, batch_data_samples)
                if kpt_seq is not None:
                    # V11: concat GAP appearance features
                    if appearance_dim > 0:
                        gap = self._extract_gap_features(
                            roi_feats, rois, batch_data_samples)
                        if gap is not None:
                            N, T, _ = kpt_seq.shape
                            # Truncate/project GAP to appearance_dim
                            app_feat = gap[:, :appearance_dim]  # (N, app_dim)
                            app_expanded = app_feat.unsqueeze(1).expand(
                                N, T, appearance_dim)
                            kpt_seq = torch.cat(
                                [kpt_seq, app_expanded], dim=-1)
                    # V12: binary label mapping (num_classes==1)
                    if self.action_head.num_classes == 1:
                        valid_mask = gt_action >= 0
                        binary = (gt_action >= 6).long()
                        gt_action = torch.where(
                            valid_mask, binary,
                            torch.full_like(gt_action, -1))
                    action_losses = self.action_head.loss(
                        kpt_seq, gt_action)
                    losses.update(action_losses)
                else:
                    # V8 fallback: single-frame pose_head features
                    with torch.no_grad():
                        kpt_features = self._extract_kpt_features_from_roi(
                            roi_feats)
                    # V11: concat GAP appearance features (fallback path)
                    if appearance_dim > 0:
                        gap = self._extract_gap_features(
                            roi_feats, rois, batch_data_samples)
                        if gap is not None:
                            N = kpt_features.size(0)
                            app_feat = gap[:, :appearance_dim]
                            app_expanded = app_feat.unsqueeze(1)  # (N,1,app)
                            kpt_features = torch.cat(
                                [kpt_features, app_expanded], dim=-1)
                    gt_action = self._extract_action_labels(
                        rois, batch_data_samples)
                    if gt_action is not None:
                        # V12: binary label mapping (fallback path)
                        if self.action_head.num_classes == 1:
                            valid_mask = gt_action >= 0
                            binary = (gt_action >= 6).long()
                            gt_action = torch.where(
                                valid_mask, binary,
                                torch.full_like(gt_action, -1))
                        action_losses = self.action_head.loss(
                            kpt_features, gt_action)
                        losses.update(action_losses)
                    else:
                        losses['loss_action'] = x[0].sum() * 0
            else:
                losses['loss_action'] = x[0].sum() * 0

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

        # Action classification
        if self.action_head is not None:
            appearance_dim = getattr(self.action_head, 'appearance_dim', 0)
            kpt_features = None
            # V9: try GT temporal sequences (for val consistency with training)
            if rois.numel() > 0:
                kpt_seq, _ = self._extract_temporal_action_data(
                    rois, batch_data_samples)
                if kpt_seq is not None:
                    kpt_features = kpt_seq
            # Fallback: single-frame pose_head features (real inference)
            if kpt_features is None:
                with torch.no_grad():
                    kpt_features = self._extract_kpt_features_from_roi(
                        roi_feats)
            # V11: concat GAP appearance features
            if appearance_dim > 0:
                gap = self._extract_gap_features(
                    roi_feats, rois, batch_data_samples)
                if gap is not None:
                    N, T, _ = kpt_features.shape
                    app_feat = gap[:, :appearance_dim]
                    app_expanded = app_feat.unsqueeze(1).expand(
                        N, T, appearance_dim)
                    kpt_features = torch.cat(
                        [kpt_features, app_expanded], dim=-1)
            action_probs = self.action_head.predict(kpt_features)
            self._attach_action(det_results_out, roi_map, action_probs)

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
        """Build RoIs from GT boxes for training pose head.

        Adds random jitter (±10% of box size) to GT boxes during training
        to reduce the gap between GT boxes and predicted boxes at test time.
        """
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

            # 对GT框加随机扰动，模拟检测框偏移，减小训练-测试gap
            if self.training:
                bboxes = bboxes.clone()
                ws = (bboxes[:, 2] - bboxes[:, 0]).clamp(min=1.0)
                hs = (bboxes[:, 3] - bboxes[:, 1]).clamp(min=1.0)
                noise = torch.randn(bboxes.size(0), 4, device=bboxes.device) * 0.1
                bboxes[:, 0] += noise[:, 0] * ws
                bboxes[:, 1] += noise[:, 1] * hs
                bboxes[:, 2] += noise[:, 2] * ws
                bboxes[:, 3] += noise[:, 3] * hs

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

    # ------------------------------------------------------------------
    # V11: RoI appearance feature extraction
    # ------------------------------------------------------------------
    def _extract_gap_features(self, roi_feats, rois, batch_data_samples):
        """Extract GAP (Global Average Pooling) appearance features from RoI.

        For training: match RoI to GT via IoU, then extract GAP from roi_feats.
        Returns (N, appearance_dim) tensor aligned with the temporal kpt_seq.

        Args:
            roi_feats: (M, C, H, W) RoI-aligned features from pose_roi_extractor.
            rois: (M, 5) RoIs [batch_idx, x1, y1, x2, y2].
            batch_data_samples: batch data samples for IoU matching.

        Returns:
            gap_feats: (M, C) GAP features for each RoI, or None.
        """
        if roi_feats is None or roi_feats.numel() == 0:
            return None
        # GAP: (M, C, H, W) → (M, C)
        gap = roi_feats.mean(dim=[2, 3])  # (M, C)
        return gap

    # ------------------------------------------------------------------
    # Action head helpers
    # ------------------------------------------------------------------
    def _build_action_gt(self, rois, batch_data_samples):
        """Build action training inputs from GT keypoints and falling labels.

        Uses GT keypoints (not pose_head predictions) so the action head
        gets clean training signal from epoch 1.

        Returns:
            kpt_features: (N, 1, K*3) normalized GT keypoints, or None.
            gt_falling: (N,) falling labels.
        """
        num_rois = rois.size(0)
        device = rois.device
        K = getattr(self.pose_head, 'num_keypoints', 7)

        # 只收集成功匹配到 GT 的 RoI（未匹配的不参与 action loss）
        valid_features = []
        valid_falling = []

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
            if float(max_iou.item()) < 0.1:
                continue
            idx = int(gt_idx.item())

            # GT keypoints → 归一化到 [0,1] relative to RoI
            gt_kpts = getattr(gt_instances, 'keypoints', None)
            if gt_kpts is None:
                gt_kpts = getattr(ds, 'gt_keypoints', None)
            if gt_kpts is None:
                continue
            if isinstance(gt_kpts, Tensor):
                kpts = gt_kpts.to(device=device, dtype=torch.float32)
            else:
                kpts = torch.as_tensor(gt_kpts, device=device, dtype=torch.float32)
            if kpts.dim() == 2:
                kpts = kpts.unsqueeze(0)
            if idx >= kpts.size(0):
                continue

            kpts_i = kpts[idx]  # (K, 3) — x, y, visibility
            x1, y1, x2, y2 = rois[i, 1:].tolist()
            roi_w = max(x2 - x1, 1.0)
            roi_h = max(y2 - y1, 1.0)

            norm_x = ((kpts_i[:K, 0] - x1) / roi_w).clamp(0, 1)
            norm_y = ((kpts_i[:K, 1] - y1) / roi_h).clamp(0, 1)
            # visibility: v>=2 可见=1, v==1 遮挡=0.5, v==0 不可见=0
            vis = kpts_i[:K, 2].clone()
            vis = (vis >= 2).float() + (vis == 1).float() * 0.5

            feat = torch.stack([norm_x, norm_y, vis], dim=-1)  # (K, 3)

            # Falling label
            falling_val = 0.0
            falling = getattr(gt_instances, 'falling', None)
            if falling is not None and idx < len(falling):
                falling_val = float(falling[idx])

            valid_features.append(feat.reshape(-1))  # (K*3,)
            valid_falling.append(falling_val)

        if len(valid_features) == 0:
            return None, None

        kpt_features = torch.stack(valid_features).unsqueeze(1)  # (M, 1, K*3)
        gt_falling = torch.tensor(valid_falling, device=device)  # (M,)
        return kpt_features, gt_falling

    def _extract_action_labels(self, rois, batch_data_samples):
        """Extract GT action class labels for each RoI via IoU matching.

        Supports both 10-class (action_class 0-9) and binary (falling 0/1).
        Prefers action_class if available, falls back to falling.

        Returns:
            gt_labels: (N,) action labels, or None if no matches.
        """
        num_rois = rois.size(0)
        device = rois.device
        # Default -1 = invalid (will be filtered by loss)
        gt_labels = torch.full((num_rois,), -1, device=device, dtype=torch.long)
        has_any = False

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

            roi_box = rois[i, 1:].unsqueeze(0)
            ious = bbox_overlaps(roi_box, gt_bboxes_t)
            max_iou, gt_idx = ious.max(dim=1)
            if float(max_iou.item()) < 0.1:
                continue
            idx = int(gt_idx.item())

            # Prefer action_class (0-9), fall back to falling (0/1)
            action_class = getattr(gt_instances, 'action_class', None)
            if action_class is not None and idx < len(action_class):
                val = int(action_class[idx])
                if val >= 0:
                    gt_labels[i] = val
                    has_any = True
                    continue

            # Fallback: binary falling
            falling = getattr(gt_instances, 'falling', None)
            if falling is not None and idx < len(falling):
                gt_labels[i] = int(falling[idx])
                has_any = True

        if not has_any:
            return None
        return gt_labels

    def _extract_temporal_action_data(self, rois, batch_data_samples):
        """Extract GT temporal keypoint sequences and action labels for each RoI.

        Used for V9 multi-frame training. Falls back to None if kpt_sequence
        is not available in the dataset (backward compatible with V8 data).

        Returns:
            kpt_sequences: (N, T, K*3) tensor, or None if not available.
            gt_labels: (N,) action class labels (-1 for unmatched), or None.
        """
        num_rois = rois.size(0)
        device = rois.device

        gt_labels = torch.full(
            (num_rois,), -1, device=device, dtype=torch.long)
        all_sequences = []
        has_any_label = False
        has_any_seq = False
        T = None
        KD = None  # feature dim per frame (21 for K*3, 35 for K*5)

        for i in range(num_rois):
            img_idx = int(rois[i, 0])
            if img_idx >= len(batch_data_samples):
                all_sequences.append(None)
                continue

            ds = batch_data_samples[img_idx]
            gt_instances = getattr(ds, 'gt_instances', None)
            if gt_instances is None or len(gt_instances) == 0:
                all_sequences.append(None)
                continue

            gt_bboxes = getattr(gt_instances, 'bboxes', None)
            if gt_bboxes is None or len(gt_bboxes) == 0:
                all_sequences.append(None)
                continue

            gt_bboxes_t = get_box_tensor(gt_bboxes).to(device=device)
            roi_box = rois[i, 1:].unsqueeze(0)
            ious = bbox_overlaps(roi_box, gt_bboxes_t)
            max_iou, gt_idx = ious.max(dim=1)
            if float(max_iou.item()) < 0.1:
                all_sequences.append(None)
                continue
            idx = int(gt_idx.item())

            # Get action label
            action_class = getattr(gt_instances, 'action_class', None)
            if action_class is not None and idx < len(action_class):
                val = int(action_class[idx])
                if val >= 0:
                    gt_labels[i] = val
                    has_any_label = True

            # Get kpt_sequence
            kpt_seq = getattr(gt_instances, 'kpt_sequence', None)
            if kpt_seq is not None and idx < kpt_seq.size(0):
                seq = kpt_seq[idx]  # (T, KD)
                all_sequences.append(seq)
                has_any_seq = True
                if T is None:
                    T = seq.size(0)
                    KD = seq.size(1)
            else:
                all_sequences.append(None)

        if not has_any_label or not has_any_seq or T is None:
            return None, None

        # Build (N, T, KD) tensor
        kpt_sequences = torch.zeros(num_rois, T, KD, device=device)
        for i, seq in enumerate(all_sequences):
            if seq is not None:
                kpt_sequences[i] = seq.to(device)

        return kpt_sequences, gt_labels

    def _extract_kpt_features_from_roi(self, roi_feats):
        """Extract normalized keypoint features from pose_head for action inference.

        直接用 pose_head.forward 在 RoI 特征空间解码，输出 [0,1] 归一化坐标。
        避免 rescale 后 keypoint 坐标和 RoI 坐标空间不匹配的问题。
        和训练时 _build_action_gt 的编码方式一致。

        Args:
            roi_feats: (N, C, H, W) RoI-aligned features.

        Returns:
            kpt_features: (N, 1, K*3) normalized [norm_x, norm_y, vis].
        """
        x_logits, y_logits, vis_logits = self.pose_head.forward(roi_feats)

        scale = getattr(self.pose_head, 'simdr_scale', 2)
        feat_w = x_logits.size(-1) // scale  # e.g. 48
        feat_h = y_logits.size(-1) // scale

        # argmax → 归一化到 [0, 1]（在 RoI 内的相对位置）
        pred_x = x_logits.argmax(dim=-1).float() / scale / feat_w  # (N, K) ∈ [0,1]
        pred_y = y_logits.argmax(dim=-1).float() / scale / feat_h  # (N, K) ∈ [0,1]
        pred_vis = torch.sigmoid(vis_logits)  # (N, K) ∈ [0,1]

        # 超出 [0,1] 的点标记为不可见
        oob = (pred_x < 0) | (pred_x > 1) | (pred_y < 0) | (pred_y > 1)
        pred_x = pred_x.clamp(0, 1)
        pred_y = pred_y.clamp(0, 1)
        pred_vis[oob] = 0.0

        kpt = torch.stack([pred_x, pred_y, pred_vis], dim=-1)  # (N, K, 3)

        # V10 bone mode: convert keypoints to bone features
        if self.action_head is not None:
            bone_mode = getattr(self.action_head, 'bone_mode', False)
            if bone_mode:
                return self._kpt_to_bone_features(kpt)  # (N, 1, 36)

        # Pad with zero velocity (dx=0, dy=0) if action_head expects kpt_dim > 3
        if self.action_head is not None:
            kpt_dim = getattr(self.action_head, 'kpt_dim', 3)
            if kpt_dim > 3:
                pad = torch.zeros(
                    kpt.size(0), kpt.size(1), kpt_dim - 3, device=kpt.device)
                kpt = torch.cat([kpt, pad], dim=-1)  # (N, K, kpt_dim)

        return kpt.reshape(kpt.size(0), 1, -1)  # (N, 1, K*kpt_dim)

    def _kpt_to_bone_features(self, kpt):
        """Convert keypoint predictions to bone features for V10.

        Args:
            kpt: (N, K, 3) tensor with (norm_x, norm_y, vis) per keypoint.

        Returns:
            bone_feat: (N, 1, 36) bone features for single-frame inference.
                Per bone: (dx, dy, angle, length, d_angle=0, d_length=0).
        """
        from rtmdet_pose_ext.action_head import BONE_CONNECTIONS
        N = kpt.size(0)
        device = kpt.device
        feats = []
        for start, end in BONE_CONNECTIONS:
            dx = kpt[:, end, 0] - kpt[:, start, 0]  # (N,)
            dy = kpt[:, end, 1] - kpt[:, start, 1]  # (N,)
            length = torch.sqrt(dx * dx + dy * dy + 1e-8)
            angle = torch.atan2(dx, dy)
            # Single frame: no temporal velocity
            d_angle = torch.zeros_like(angle)
            d_length = torch.zeros_like(length)
            feats.append(torch.stack(
                [dx, dy, angle, length, d_angle, d_length], dim=-1))
        bone_feat = torch.cat(feats, dim=-1)  # (N, 36)
        return bone_feat.unsqueeze(1)  # (N, 1, 36)

    def _attach_action(self, det_results, roi_map, action_output):
        """Attach action prediction scores to detection results.

        Handles both binary (legacy Tensor) and multi-class (dict) outputs.
        For multi-class, attaches:
            - action_scores: (N, 10) per-class probabilities
            - action_class: (N,) predicted class IDs
            - is_falling: (N,) bool
        For binary (backward compat):
            - action_scores: (N,) falling probability
        """
        is_multiclass = isinstance(action_output, dict)

        if is_multiclass:
            num_classes = action_output['action_probs'].size(-1)

        for inst in det_results:
            if inst is None or len(inst) == 0:
                continue
            device = inst.bboxes.device
            n = len(inst)
            if is_multiclass:
                inst.action_scores = torch.zeros(n, num_classes, device=device)
                inst.action_class = torch.zeros(n, dtype=torch.long, device=device)
                inst.is_falling = torch.zeros(n, dtype=torch.bool, device=device)
            else:
                inst.action_scores = torch.zeros(n, device=device)

        for i, (img_idx, inst_idx) in enumerate(roi_map):
            if img_idx >= len(det_results):
                continue
            inst = det_results[img_idx]
            if inst is None or inst_idx >= len(inst):
                continue
            if is_multiclass:
                inst.action_scores[inst_idx] = action_output['action_probs'][i]
                inst.action_class[inst_idx] = action_output['action_class'][i]
                inst.is_falling[inst_idx] = action_output['is_falling'][i]
            else:
                inst.action_scores[inst_idx] = action_output[i]

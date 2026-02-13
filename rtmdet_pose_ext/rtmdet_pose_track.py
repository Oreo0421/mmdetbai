# -*- coding: utf-8 -*-
"""
RTMDetPoseTracker: Inference wrapper for RTMDetWithPose + ByteTrack + temporal action.

Usage:
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile('rtmdet_pose_v7_withclass_track.py')
    # Build model from checkpoint
    model = ...  # load trained RTMDetWithPose with action_head

    tracker = RTMDetPoseTracker(
        model=model,
        tracker_cfg=cfg.tracker,
        max_seq_len=30,
    )

    for frame_id, (img_tensor, data_sample) in enumerate(video_frames):
        results = tracker.process_frame(img_tensor, data_sample, frame_id)
        # results.pred_instances has: bboxes, scores, keypoints, keypoint_scores,
        #                             instances_id (track ID), action_scores
"""
import collections

import torch
from torch import Tensor
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample


class RTMDetPoseTracker:
    """Inference wrapper: RTMDetWithPose + ByteTracker + temporal action classification.

    Args:
        model: Trained RTMDetWithPose model (with action_head).
        tracker_cfg: ByteTracker config dict.
        max_seq_len: Maximum keypoint sequence length per track (default: 30).
        action_threshold: Threshold for falling classification (default: 0.5).
    """

    def __init__(
        self,
        model,
        tracker_cfg: dict,
        max_seq_len: int = 30,
        action_threshold: float = 0.5,
    ):
        self.model = model
        self.model.eval()
        self.tracker = MODELS.build(tracker_cfg)
        self.max_seq_len = max_seq_len
        self.action_threshold = action_threshold

        # Per-track keypoint buffer: track_id -> deque of (K*3,) tensors
        self.kpt_buffers = {}

    def reset(self):
        """Reset tracker state and keypoint buffers."""
        self.tracker.reset()
        self.kpt_buffers.clear()

    @torch.no_grad()
    def process_frame(
        self,
        img_tensor: Tensor,
        data_sample: DetDataSample,
        frame_id: int,
    ) -> DetDataSample:
        """Process one frame: detect, pose, track, temporal action classify.

        Args:
            img_tensor: Image tensor of shape (1, C, H, W).
            data_sample: DetDataSample with metainfo.
            frame_id: Frame index (0-based, tracker resets at 0).

        Returns:
            DetDataSample with pred_track_instances containing:
                - bboxes, labels, scores, instances_id (from tracker)
                - keypoints, keypoint_scores (from pose head)
                - action_scores (temporal action classification)
                - is_falling (binary falling flag)
        """
        # 1. Per-frame detection + pose + single-frame action
        data_sample.set_metainfo({'frame_id': frame_id})
        results_list = self.model.predict(img_tensor, [data_sample], rescale=True)
        result = results_list[0]

        # 2. ByteTrack: associate persons across frames
        track_instances = self.tracker.track(result)
        device = track_instances.bboxes.device
        num_tracks = len(track_instances)

        # 3. Match tracked instances to predicted instances for keypoints
        pred_inst = result.pred_instances
        track_kpts = torch.zeros(num_tracks, self._num_kpts, 2, device=device)
        track_kpt_scores = torch.zeros(num_tracks, self._num_kpts, device=device)

        if num_tracks > 0 and len(pred_inst) > 0:
            # Match by IoU between tracked bboxes and predicted bboxes
            from mmdet.structures.bbox import bbox_overlaps
            ious = bbox_overlaps(track_instances.bboxes, pred_inst.bboxes)
            # Greedy assignment: each tracked box → best matching pred box
            for t_idx in range(num_tracks):
                if ious.size(1) == 0:
                    break
                best_pred = ious[t_idx].argmax()
                if ious[t_idx, best_pred] > 0.3:
                    if hasattr(pred_inst, 'keypoints'):
                        track_kpts[t_idx] = pred_inst.keypoints[best_pred]
                    if hasattr(pred_inst, 'keypoint_scores'):
                        track_kpt_scores[t_idx] = pred_inst.keypoint_scores[best_pred]

        track_instances.keypoints = track_kpts
        track_instances.keypoint_scores = track_kpt_scores

        # 4. Buffer keypoints per track_id & run temporal action classification
        action_scores = torch.zeros(num_tracks, device=device)
        is_falling = torch.zeros(num_tracks, dtype=torch.bool, device=device)

        if self.model.action_head is not None and num_tracks > 0:
            for i in range(num_tracks):
                tid = int(track_instances.instances_id[i])

                # Initialize buffer for new tracks
                if tid not in self.kpt_buffers:
                    self.kpt_buffers[tid] = collections.deque(
                        maxlen=self.max_seq_len)

                # Normalize keypoints relative to bbox
                bbox = track_instances.bboxes[i]  # (4,)
                kpt = track_kpts[i]                # (K, 2)
                vis = track_kpt_scores[i]          # (K,)
                norm_kpt = self._normalize_kpts(kpt, vis, bbox)  # (K*3,)
                self.kpt_buffers[tid].append(norm_kpt)

                # Build temporal sequence
                seq = torch.stack(list(self.kpt_buffers[tid]))  # (T, K*3)
                seq = seq.unsqueeze(0).to(device)  # (1, T, K*3)

                prob = self.model.action_head.predict(seq)  # (1,)
                action_scores[i] = prob.squeeze()
                is_falling[i] = prob.squeeze() > self.action_threshold

        track_instances.action_scores = action_scores
        track_instances.is_falling = is_falling

        # 5. Clean up stale tracks
        active_ids = set()
        if num_tracks > 0:
            active_ids = set(track_instances.instances_id.tolist())
        stale_ids = [tid for tid in self.kpt_buffers if tid not in active_ids]
        for tid in stale_ids:
            del self.kpt_buffers[tid]

        # Attach to result
        result.pred_track_instances = track_instances
        return result

    @property
    def _num_kpts(self):
        return getattr(self.model.pose_head, 'num_keypoints', 7)

    def _normalize_kpts(self, kpts, vis_scores, bbox):
        """Normalize keypoints to [0, 1] relative to bbox.

        Args:
            kpts: (K, 2) keypoints in image coords.
            vis_scores: (K,) visibility scores.
            bbox: (4,) [x1, y1, x2, y2].

        Returns:
            (K*3,) flattened [norm_x, norm_y, vis] per keypoint.
        """
        x1, y1, x2, y2 = bbox
        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)

        norm_x = ((kpts[:, 0] - x1) / w).clamp(0, 1)
        norm_y = ((kpts[:, 1] - y1) / h).clamp(0, 1)

        # (K, 3) → (K*3,)
        return torch.stack([norm_x, norm_y, vis_scores], dim=-1).reshape(-1)

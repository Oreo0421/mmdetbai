# mmdet/models/detectors/rtmdet_pose.py
from mmdet.registry import MODELS
from mmdet.models.detectors.single_stage import SingleStageDetector

@MODELS.register_module()
class RTMDetPose(SingleStageDetector):
    """RTMDet + Heatmap pose head (shared backbone+neck feats)."""

    def __init__(self, pose_head=None, **kwargs):
        super().__init__(**kwargs)  # builds backbone, neck, bbox_head, ...
        self.pose_head = MODELS.build(pose_head) if pose_head is not None else None

    def loss(self, batch_inputs, batch_data_samples):
        # 1) shared feats
        feats = self.extract_feat(batch_inputs)

        # 2) det losses (RTMDetHead already knows how to read gt bboxes/labels)
        losses = self.bbox_head.loss(feats, batch_data_samples)

        # 3) pose losses (you implement/choose a pose_head that can read gt keypoints)
        if self.pose_head is not None:
            pose_losses = self.pose_head.loss(feats, batch_data_samples)
            losses.update(pose_losses)

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        feats = self.extract_feat(batch_inputs)

        # det result
        det_results = self.bbox_head.predict(
            feats, batch_data_samples, rescale=rescale)

        # pose result (optional)
        if self.pose_head is not None:
            pose_results = self.pose_head.predict(
                feats, batch_data_samples, rescale=rescale)
            # 你可以把 pose_results 塞回 data_samples 或者返回 dict
            return det_results, pose_results

        return det_results


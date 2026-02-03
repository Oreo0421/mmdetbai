# mmdet/models/detectors/rtmdet_with_pose.py

from typing import List, Tuple, Union
import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class RTMDetWithPose(SingleStageDetector):
    """RTMDet with Pose Estimation Head.
    
    Multi-task detector for human detection + pose estimation.
    """
    
    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        pose_head: ConfigType,
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
        # 添加 pose head
        self.pose_head = MODELS.build(pose_head)
    
    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> dict:
        """计算 detection + pose 的联合损失."""
        
        # 提取特征
        x = self.extract_feat(batch_inputs)
        
        losses = dict()
        
        # 1. Detection losses
        det_losses = self.bbox_head.loss(x, batch_data_samples)
        losses.update(det_losses)
        
        # 2. Pose losses
        pose_losses = self.pose_head.loss(x, batch_data_samples)
        losses.update(pose_losses)
        
        return losses
    
    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """同时预测 bbox 和 pose."""
        
        x = self.extract_feat(batch_inputs)
        
        # Detection prediction
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale
        )
        
        # Pose prediction
        pose_results = self.pose_head.predict(
            x, batch_data_samples, rescale=rescale
        )
        
        # 合并结果
        for i, (det_result, pose_result) in enumerate(
            zip(results_list, pose_results)
        ):
            det_result.pred_instances.keypoints = pose_result.pred_instances.keypoints
            det_result.pred_instances.keypoint_scores = pose_result.pred_instances.keypoint_scores
        
        return results_list

from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs
import torch


@TRANSFORMS.register_module(force=True)
class PackDetInputsWithPose(PackDetInputs):
    """扩展 PackDetInputs 以支持关键点热图"""
    
    def transform(self, results: dict) -> dict:
        # 先调用父类
        packed = super().transform(results)
        
        # 添加关键点热图到 gt_instances
        if 'gt_keypoints_heatmap' in results:
            heatmap = torch.from_numpy(results['gt_keypoints_heatmap'])
            packed['data_samples'].gt_instances.keypoints_heatmap = heatmap
            
        if 'gt_keypoints' in results:
            kpts = torch.from_numpy(results['gt_keypoints'])
            packed['data_samples'].gt_instances.keypoints = kpts
        
        return packed

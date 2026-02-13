from .keypoint_transforms import GeneratePoseHeatmap, CopyImgIdToId, RandomRotateWithPose
from .custom_pack import PackDetInputsWithPose
from .coco_pose_dataset import CocoPoseDataset
from .losses import KeypointOHKMMSELoss

from .heatmap_head import HeatmapHead
from .heatmap_head_v4 import HeatmapHeadV4
from .regression_head import CoordinateRegressionHead
from .rtmdet_with_pose import RTMDetWithPose

__all__ = [
    'GeneratePoseHeatmap',
    'RandomRotateWithPose',
    'PackDetInputsWithPose',
    'CocoPoseDataset',
    'KeypointOHKMMSELoss',

    'HeatmapHead',
    'HeatmapHeadV4',
    'CoordinateRegressionHead',
    'RTMDetWithPose',
    'CopyImgIdToId',
]

print("✓ RTMDet Pose Extension loaded!")
print("  - GeneratePoseHeatmap")
print("  - RandomRotateWithPose")
print("  - PackDetInputsWithPose")
print("  - CocoPoseDataset")
print("  - KeypointOHKMMSELoss")

print("  - HeatmapHead")
print("  - HeatmapHeadV4")
print("  - CoordinateRegressionHead")
print("  - RTMDetWithPose")
print("  - CopyImgIdToId")

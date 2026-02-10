"""
RTMDet Pose Extension Package
将这个文件和所有 .py 文件放在同一个目录下，然后在配置文件中导入即可
"""

from mmdet.registry import MODELS, TRANSFORMS

# 使用 try-except 避免重复注册
def safe_register():
    """安全注册模块，避免重复注册错误"""
    
    # 导入所有自定义模块
    from .rtmdet_with_pose import RTMDetWithPose
    from .heatmap_head import HeatmapHead
    from .keypoint_mse_loss import KeypointMSELoss, KeypointOHKMMSELoss
    from .keypoint_transforms import GenerateKeypointHeatmap
    
    # 检查并注册模块
    modules_to_register = [
        ('RTMDetWithPose', RTMDetWithPose, MODELS),
        ('HeatmapHead', HeatmapHead, MODELS),
        ('KeypointMSELoss', KeypointMSELoss, MODELS),
        ('KeypointOHKMMSELoss', KeypointOHKMMSELoss, MODELS),
        ('GenerateKeypointHeatmap', GenerateKeypointHeatmap, TRANSFORMS),
    ]
    
    registered = []
    for name, module, registry in modules_to_register:
        if name not in registry.module_dict:
            registry.register_module(module=module, force=False)
            registered.append(name)
        else:
            print(f"  ⚠ {name} already registered, skipping")
    
    if registered:
        print("✓ RTMDet Pose Extension loaded successfully!")
        for name in registered:
            print(f"  - {name}")
    
    return RTMDetWithPose, HeatmapHead, KeypointMSELoss, KeypointOHKMMSELoss, GenerateKeypointHeatmap

# 执行注册
RTMDetWithPose, HeatmapHead, KeypointMSELoss, KeypointOHKMMSELoss, GenerateKeypointHeatmap = safe_register()

__all__ = [
    'RTMDetWithPose',
    'HeatmapHead',
    'KeypointMSELoss',
    'KeypointOHKMMSELoss',
    'GenerateKeypointHeatmap',
]

# ============================================================
# RTMDet + MobileViT + Pose Head on SensIR COCO
# Author: Taijing Bai
# Task: Human Detection (1 class) + 7-keypoint Pose Estimation
# Dataset: SensIR (40x114 thermal images -> COCO format)
# Keypoints: head, shoulder, hand_right, hand_left, hips, foot_right, foot_left
# ============================================================
import sys
sys.path.insert(0, '/path/to/rtmdet_pose_ext')  # ⚠️ 改成您的实际路径!

import rtmdet_pose_ext  # 这会自动注册所有模块
_base_ = 'configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'

# ============================================================
# 1. Dataset Configuration
# ============================================================

data_root = '/home/tbai/Desktop/sensir_coco/'

metainfo = dict(
    classes=('human',),
    # 定义7个关键点
    keypoint_info={
        0: {'name': 'head', 'id': 0, 'color': [255, 0, 0]},
        1: {'name': 'shoulder', 'id': 1, 'color': [0, 255, 0]},
        2: {'name': 'hand_right', 'id': 2, 'color': [0, 0, 255]},
        3: {'name': 'hand_left', 'id': 3, 'color': [255, 255, 0]},
        4: {'name': 'hips', 'id': 4, 'color': [255, 0, 255]},
        5: {'name': 'foot_right', 'id': 5, 'color': [0, 255, 255]},
        6: {'name': 'foot_left', 'id': 6, 'color': [128, 128, 128]},
    },
)

num_classes = 1
num_keypoints = 7

train_ann = 'annotations/instances_train.json'
val_ann   = 'annotations/instances_val.json'
test_ann  = 'annotations/instances_test.json'


# ============================================================
# 2. Model - Multi-task Detector (Detection + Pose)
# ============================================================

model = dict(
    type='RTMDetWithPose',  # 自定义的多任务检测器
    
    # Backbone: MobileViT
    backbone=dict(
        _delete_=True,
        type='TimmMobileViT',
        model_name='mobilevit_s',
        out_indices=(2, 3, 4),
    ),
    
    # Neck: CSPNeXtPAFPN
    neck=dict(
        _delete_=True,
        type='CSPNeXtPAFPN',
        in_channels=[96, 128, 640],   # MobileViT-S output channels
        out_channels=96,
        num_csp_blocks=1,
    ),
    
    # Detection Head: RTMDetHead
    bbox_head=dict(
        _delete_=True,
        type='RTMDetHead',
        num_classes=num_classes,
        in_channels=96,
        feat_channels=96,
        stacked_convs=2,
        with_objectness=False,
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32],
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type='GIoULoss',
            loss_weight=2.0,
        ),
    ),
    
    # Pose Head: HeatmapHead (NEW!)
    pose_head=dict(
        type='HeatmapHead',
        num_keypoints=num_keypoints,
        in_channels=96,           # 与 neck.out_channels 一致
        feat_channels=128,        # 中间特征通道数
        deconv_out_channels=64,   # 反卷积输出通道数
        use_deconv=True,          # 使用可学习的反卷积上采样
        loss_keypoint=dict(
            type='KeypointMSELoss',
            use_target_weight=True,  # 使用关键点权重 (可见性)
            loss_weight=2.0,         # 姿态损失权重 (可调)
        ),
    ),
)


# ============================================================
# 3. Image Size
# ============================================================

# 原始图像: 40x114 → 拉伸成正方形 192x192
img_scale = (192, 192)


# ============================================================
# 4. Data Pipelines
# ============================================================

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),  # ✅ 加载关键点
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    
    # ✅ 生成关键点热图 (核心步骤!)
    dict(
        type='GenerateKeypointHeatmap',
        heatmap_size=(48, 48),  # 输出热图大小 (stride=4, 192/4=48)
        sigma=2.0,              # 高斯核标准差 (控制热图宽度)
    ),
    
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),  # ✅ 加载关键点
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    
    # 测试时也需要生成热图 (用于验证)
    dict(
        type='GenerateKeypointHeatmap',
        heatmap_size=(48, 48),
        sigma=2.0,
    ),
    
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
    ),
]


# ============================================================
# 5. Dataloaders
# ============================================================

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann,
        data_prefix=dict(img=''),  # 文件名已包含 images/train/ 路径
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann,
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=test_ann,
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)


# ============================================================
# 6. Evaluators
# ============================================================

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}{val_ann}',
    metric=['bbox', 'keypoints'],  # ✅ 同时评估检测和姿态
    format_only=False,
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}{test_ann}',
    metric=['bbox', 'keypoints'],  # ✅ 同时评估检测和姿态
    format_only=False,
)


# ============================================================
# 7. Training Schedule
# ============================================================

max_epochs = 30

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# ============================================================
# 8. Optimizer & Learning Rate
# ============================================================

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=1e-4,
    ),
    clip_grad=dict(max_norm=10, norm_type=2),
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        by_epoch=True,
        eta_min=1e-6,
    ),
]


# ============================================================
# 9. Hooks & Visualization
# ============================================================

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=3,
    ),
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
)


# ============================================================
# 10. Custom Imports (重要!)
# ============================================================

custom_imports = dict(
    imports=[
        'path.to.rtmdet_with_pose',      # 替换为实际路径
        'path.to.heatmap_head',
        'path.to.keypoint_mse_loss',
        'path.to.keypoint_transforms',
    ],
    allow_failed_imports=False,
)

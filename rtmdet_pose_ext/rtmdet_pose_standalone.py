# ============================================================
# RTMDet + MobileViT + Pose Head (完全独立配置)
# 不依赖任何 _base_ 文件
# ============================================================

# ============================================================
# 0. 导入自定义模块
# ============================================================
# ✅ 正确方法
custom_imports = dict(
    imports=['rtmdet_pose_ext'],  # 自动导入并注册
    allow_failed_imports=False,
)

# ============================================================
# 1. 基础设置
# ============================================================
default_scope = 'mmdet'

# 数据路径
data_root = '/home/tbai/Desktop/sensir_coco/'
work_dir = './work_dirs/rtmdet_mobilevit_pose'

# 基本配置
num_classes = 1
num_keypoints = 7
img_scale = (192, 192)
max_epochs = 30

# Metainfo
metainfo = dict(
    classes=('human',),
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


# ============================================================
# 2. Model
# ============================================================
model = dict(
    type='RTMDetWithPose',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        batch_augments=None,
    ),
    backbone=dict(
        type='TimmMobileViT',
        model_name='mobilevit_s',
        out_indices=(2, 3, 4),
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 128, 640],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=num_classes,
        in_channels=96,
        feat_channels=96,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        with_objectness=False,
        exp_on_reg=False,
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
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    pose_head=dict(
        type='HeatmapHead',
        num_keypoints=num_keypoints,
        in_channels=96,
        feat_channels=128,
        deconv_out_channels=64,
        use_deconv=True,
        loss_keypoint=dict(
            type='KeypointMSELoss',
            use_target_weight=True,
            loss_weight=2.0,
        ),
    ),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
    ),
)


# ============================================================
# 3. Data Pipelines
# ============================================================
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=img_scale,
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False,
    ),
    dict(
        type='RandomResize',
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=img_scale,
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5,
    ),
    dict(type='GenerateKeypointHeatmap', heatmap_size=(48, 48), sigma=2.0),
    dict(type='PackDetInputs'),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='GenerateKeypointHeatmap', heatmap_size=(48, 48), sigma=2.0),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='GenerateKeypointHeatmap', heatmap_size=(48, 48), sigma=2.0),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]


# ============================================================
# 4. Dataloaders
# ============================================================
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ),
    collate_fn=dict(type='default_collate'),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
    ),
    collate_fn=dict(type='default_collate'),
)

test_dataloader = val_dataloader


# ============================================================
# 5. Evaluators
# ============================================================
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
)

test_evaluator = val_evaluator


# ============================================================
# 6. Training Schedule
# ============================================================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(max_epochs - 10, 1)],
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# ============================================================
# 7. Optimizer
# ============================================================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=10, norm_type=2),
)


# ============================================================
# 8. Learning Rate Scheduler
# ============================================================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=100,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        begin=100,
        end=max_epochs,
        T_max=max_epochs - 100,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]


# ============================================================
# 9. Hooks
# ============================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - 10,
        switch_pipeline=train_pipeline_stage2,
    ),
]


# ============================================================
# 10. Environment & Logging
# ============================================================
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# ============================================================
# 11. Runtime
# ============================================================
launcher = 'none'

# RTMDet + MobileViT + Pose (完全兼容版本)
# 移除所有可能不兼容的参数

custom_imports = dict(imports=['rtmdet_pose_ext'], allow_failed_imports=False)

default_scope = 'mmdet'
data_root = '/home/tbai/Desktop/sensir_coco/'
num_classes = 1
num_keypoints = 7
img_scale = (192, 192)
max_epochs = 30

model = dict(
    type='RTMDetWithPose',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
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
    ),
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=num_classes,
        in_channels=96,
        feat_channels=96,
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
        # 移除 deconv_out_channels 和 use_deconv 参数
        loss_keypoint=dict(
            type='KeypointMSELoss',
            use_target_weight=True,
            loss_weight=2.0,
        ),
    ),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
    ),
    test_cfg=dict(
        nms_pre=1000,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='GenerateKeypointHeatmap', heatmap_size=(48, 48), sigma=2.0),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='GenerateKeypointHeatmap', heatmap_size=(48, 48), sigma=2.0),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=('human',)),
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=('human',)),
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
)

test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=1e-6, begin=0, end=max_epochs, T_max=max_epochs, by_epoch=True, convert_to_iter_based=True),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='nccl'))
visualizer = dict(type='DetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')], name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
launcher = 'none'

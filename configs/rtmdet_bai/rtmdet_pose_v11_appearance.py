custom_imports = dict(imports=['rtmdet_pose_ext'], allow_failed_imports=False)
default_scope = 'mmdet'

data_root = '/home/tbai/Desktop/sensir_coco_v10/'  # Reuse V10 data (bone features)
num_classes = 1
img_scale = (384, 384)
max_epochs = 50

# ========= 10 Action Classes =========
# 0: Standing still (1A-1)       -> not-falling
# 1: Walking (1A-2)              -> not-falling
# 2: Sitting down (1A-3)         -> not-falling
# 3: Standing up (1A-4)          -> not-falling
# 4: Lying down (1A-5)           -> not-falling
# 5: Getting up (1A-6)           -> not-falling
# 6: Falling walking (1A-7)      -> falling
# 7: Falling standing (1A-8)     -> falling
# 8: Falling sitting (1A-9)      -> falling
# 9: Falling standing up (1A-10) -> falling

model = dict(
    type='RTMDetWithPose',
    data_preprocessor=dict(type='DetDataPreprocessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True),
    backbone=dict(type='TimmMobileViT', model_name='mobilevit_s', out_indices=(2, 3, 4)),
    neck=dict(type='CSPNeXtPAFPN', in_channels=[96, 128, 640], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(
        type='RTMDetHead', num_classes=num_classes, in_channels=96, feat_channels=96,
        anchor_generator=dict(type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    pose_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=48, sampling_ratio=0, aligned=True),
        out_channels=96,
        featmap_strides=[8, 16, 32]),
    pose_det_cfg=dict(
        nms_pre=4000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=200),
    pose_topk=50,
    pose_use_gt_box=True,
    pose_head=dict(
        type='HeatmapHeadV6',
        num_keypoints=7,
        in_channels=96,
        feat_channels=128,
        simdr_scale=2,
        sigma_1d=1.5,
        simdr_loss_weight=5.0,
        vis_loss_weight=1.0,
        occluded_weight=0.3,
        match_iou_thr=0.1,
        log_stats=True,
        log_interval=20),
    # ========= Action Classification Head (V11: bone + RoI appearance) =========
    # V11: Bone features (36-dim) + GAP appearance features (96-dim) = 132-dim input
    # appearance_dim=96: RoI GAP features from pose_roi_extractor (96 channels)
    # Training: appearance features replicated across T frames (only current frame RoI)
    # Temporal variation comes from bone sequence, appearance provides context
    action_head=dict(
        type='ActionTemporalHead',
        num_keypoints=6,            # 6 bones (not keypoints)
        kpt_dim=6,                  # dx, dy, angle, length, d_angle, d_length
        embed_dim=64,
        hidden_dim=64,
        num_gru_layers=1,
        num_classes=10,
        loss_weight=1.0,
        class_weight=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
                      8.0, 8.0, 8.0, 8.0],
        dropout=0.1,
        temporal_residual=True,
        skip_gru_t1=False,
        bone_mode=True,
        appearance_dim=96,          # V11: RoI GAP appearance feature dimension
    ),
    freeze_det_pose=True,   # Freeze backbone/neck/bbox_head/pose_head, only train action_head
    train_cfg=dict(assigner=dict(type='DynamicSoftLabelAssigner', topk=13), allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(nms_pre=1000, score_thr=0.3, nms=dict(type='nms', iou_threshold=0.45), max_per_img=100))

# ========= ByteTracker (inference only, used by RTMDetPoseTracker) =========
tracker = dict(
    type='ByteTracker',
    motion=dict(type='KalmanFilter'),
    obj_score_thrs=dict(high=0.6, low=0.1),
    init_track_thr=0.7,
    weight_iou_with_det_scores=True,
    match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
    num_tentatives=3,
    num_frames_retain=30,
)

# ========= Pipeline =========
train_pipeline = [
    dict(type='LoadImageFromFile', _scope_='mmdet'),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True, _scope_='mmdet'),
    dict(type='RandomRotateWithPose', angle_range=(-20, 20), prob=0.5, _scope_='mmdet'),
    dict(type='Resize', scale=img_scale, keep_ratio=False, _scope_='mmdet'),
    dict(type='RandomFlip', prob=0.5, _scope_='mmdet'),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18,
         _scope_='mmdet'),
    dict(type='PackDetInputsWithPose', _scope_='mmdet'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', _scope_='mmdet'),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True, _scope_='mmdet'),
    dict(type='Resize', scale=img_scale, keep_ratio=False, _scope_='mmdet'),
    dict(type='CopyImgIdToId', _scope_='mmdet'),
    dict(type='PackDetInputsWithPose', meta_keys=('id','img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'), _scope_='mmdet'),
]

# ========= Dataset =========
K = 7
custom_metainfo = dict(
    classes=('human',),
    num_keypoints=K,
    sigmas=[0.25] * 7,
)

train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoPoseDataset', data_root=data_root, metainfo=custom_metainfo,
                 ann_file='annotations/instances_train.json', data_prefix=dict(img='images/'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=1), pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoPoseDataset', data_root=data_root, metainfo=custom_metainfo,
                 ann_file='annotations/instances_val.json', data_prefix=dict(img='images/'), test_mode=True, pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_val.json', metric='bbox'),
    dict(
        type='CocoMetric',
        _scope_='mmpose',
        ann_file=data_root + 'annotations/instances_val.json',
        iou_type='keypoints',
        score_mode='bbox_keypoint',
        keypoint_score_thr=0.2,
        nms_mode='oks_nms',
        nms_thr=0.9,
        pred_converter=dict(id='id', num_keypoints=7, mapping=[(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)]),
        gt_converter=dict(id='id', num_keypoints=7, mapping=[(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)]),
    ),
    dict(
        type='ActionMetric',
        ann_file=data_root + 'annotations/instances_val.json',
        iou_thr=0.5,
        num_classes=10,
    ),
]
test_evaluator = val_evaluator

# ========= Training schedule =========
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001), clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=1e-6, begin=0, end=max_epochs, T_max=max_epochs, by_epoch=True, convert_to_iter_based=True)]

default_hooks = dict(timer=dict(type='IterTimerHook'), logger=dict(type='LoggerHook', interval=20),
                     param_scheduler=dict(type='ParamSchedulerHook'),
                     checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3),
                     sampler_seed=dict(type='DistSamplerSeedHook'), visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='nccl'))
visualizer = dict(type='DetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')], name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
# Load V6 best checkpoint: backbone, neck, bbox_head, pose_head well-trained
# (bbox=0.866, kpt=0.802). action_head is fully new (random init).
load_from = 'work_dirs/rtmdet_pose_v6/best_coco_bbox_mAP_epoch_75.pth'
resume = False

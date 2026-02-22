custom_imports = dict(imports=['rtmdet_pose_ext'], allow_failed_imports=False)
default_scope = 'mmdet'

data_root = '/home/tbai/Desktop/sensir_coco_v12/'  # V9 format data (kpt_dim=5, non-bone)
num_classes = 1
img_scale = (384, 384)
max_epochs = 20

# ========= V14c: Focal Loss + Hard Negative Weighting =========
# Two-head architecture: 4-class action (standing/walking/sitting/lying)
#   + binary fall detection with focal loss and hard negative weighting
# Uses kpt_dim=5 keypoint features (x, y, vis, dx, dy) like V12
# Label mapping handled internally by ActionV14Head:
#   10-class → 4-class: [0→standing, 1→walking, 2→sitting, 3→standing,
#                         4→lying, 5→standing, 6-9→lying]
#   10-class → binary:  [0-5→0, 6-9→1]
#
# V14c changes (vs V14b):
#   - max_epochs: 15 → 20 (focal loss converges slower)
#   - fall_loss_weight: 1.5 → 2.0 (focal loss has lower magnitude)
#   - focal_gamma: 2.0 (focus on hard-to-classify examples)
#   - hard_neg_classes: [4, 5] (lying_down, getting_up)
#   - hard_neg_weight: 3.0 (3x weight for hard negatives)
#   - kpt_weight: [1.5,1.0,0.5,0.5,1.5,1.2,1.2] (head/hips high, hands low)
#   - class_weight: [1.0, 1.5, 1.5, 1.0] (upweight minority: walking/sitting)

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
    # ========= Action Classification Head (V14c: focal loss + hard neg weighting) =========
    action_head=dict(
        type='ActionV14Head',
        num_keypoints=7,
        kpt_dim=5,                          # x, y, vis, dx, dy
        num_action_classes=4,               # standing, walking, sitting, lying
        loss_weight=1.0,                    # 4-class action CE weight
        fall_loss_weight=2.0,               # binary fall loss weight (was 1.5 in V14b)
        pos_weight=2.0,                     # falling class weight for BCE
        focal_gamma=2.0,                    # NEW: focal loss gamma
        hard_neg_classes=[4, 5],            # NEW: lying_down, getting_up
        hard_neg_weight=3.0,                # NEW: 3x weight for hard negatives
        kpt_weight=[1.5, 1.0, 0.5, 0.5, 1.5, 1.2, 1.2],  # NEW: head/hips high, hands low
        dropout=0.2,                        # more regularization
        class_weight=[1.0, 1.5, 1.5, 1.0], # NEW: upweight walking/sitting (minority)
        bone_mode=False,
        skip_gru_t1=False,
        temporal_residual=True,
    ),
    freeze_det_pose=True,
    train_cfg=dict(assigner=dict(type='DynamicSoftLabelAssigner', topk=13), allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(nms_pre=1000, score_thr=0.3, nms=dict(type='nms', iou_threshold=0.45), max_per_img=100))

# ========= ByteTracker (inference only) =========
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

optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.001), clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=1e-6, begin=0, end=max_epochs, T_max=max_epochs, by_epoch=True, convert_to_iter_based=True)]

default_hooks = dict(timer=dict(type='IterTimerHook'), logger=dict(type='LoggerHook', interval=20),
                     param_scheduler=dict(type='ParamSchedulerHook'),
                     checkpoint=dict(type='CheckpointHook', interval=1, save_best='action/fall_f1', rule='greater', max_keep_ckpts=5),
                     sampler_seed=dict(type='DistSamplerSeedHook'), visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='nccl'))
visualizer = dict(type='DetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')], name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
# Load V6 best checkpoint: backbone, neck, bbox_head, pose_head well-trained
load_from = 'work_dirs/rtmdet_pose_v6/best_coco_bbox_mAP_epoch_75.pth'
resume = False

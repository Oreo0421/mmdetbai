custom_imports = dict(imports=['rtmdet_pose_ext'], allow_failed_imports=False)
default_scope = 'mmdet'

data_root = '/home/tbai/Desktop/sensir_coco/'
num_classes = 1
img_scale = (192, 192)
max_epochs = 30

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
    pose_head=dict(
        type='HeatmapHead', num_keypoints=7, in_channels=96, feat_channels=128,
        upsample_factor=2,  # <-- 新增：deconv 上采样 24→48
        loss_keypoint=dict(type='KeypointMSELoss', use_target_weight=True, loss_weight=2.0)),
    train_cfg=dict(assigner=dict(type='DynamicSoftLabelAssigner', topk=13), allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(nms_pre=1000, score_thr=0.05, nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

# ========= Pipeline =========
# 热图尺寸 = 48x48，匹配 HeatmapHead 的 deconv 输出
train_pipeline = [
    dict(type='LoadImageFromFile', _scope_='mmdet'),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True, _scope_='mmdet'),
    dict(type='Resize', scale=img_scale, keep_ratio=False, _scope_='mmdet'),
    dict(type='RandomFlip', prob=0.5, _scope_='mmdet'),
    dict(type='GeneratePoseHeatmap', heatmap_size=(48, 48), sigma=2.0, _scope_='mmdet'),
    dict(type='PackDetInputsWithPose', _scope_='mmdet'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', _scope_='mmdet'),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True, _scope_='mmdet'),
    dict(type='Resize', scale=img_scale, keep_ratio=False, _scope_='mmdet'),
    dict(type='GeneratePoseHeatmap', heatmap_size=(48, 48), sigma=2.0, _scope_='mmdet'),
    dict(type='CopyImgIdToId', _scope_='mmdet'),
    dict(type='PackDetInputsWithPose', meta_keys=('id','img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'), _scope_='mmdet'),
]

# ========= Dataset =========
K = 7
custom_metainfo = dict(
    classes=('human',),
    num_keypoints=K,
    sigmas=[0.1] * 7,  # <-- 从 0.05 放宽到 0.1，更合理的 OKS 容差
)

train_dataloader = dict(
    batch_size=8, num_workers=4, persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoPoseDataset', data_root=data_root, metainfo=custom_metainfo,
                 ann_file='annotations/instances_train.json', data_prefix=dict(img=''),
                 filter_cfg=dict(filter_empty_gt=True, min_size=1), pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoPoseDataset', data_root=data_root, metainfo=custom_metainfo,
                 ann_file='annotations/instances_val.json', data_prefix=dict(img=''), test_mode=True, pipeline=test_pipeline))

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
]
test_evaluator = val_evaluator

# ========= Training schedule =========
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001), clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=1e-6, begin=0, end=max_epochs, T_max=max_epochs, by_epoch=True, convert_to_iter_based=True)]

default_hooks = dict(timer=dict(type='IterTimerHook'), logger=dict(type='LoggerHook', interval=20),
                     param_scheduler=dict(type='ParamSchedulerHook'),
                     checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3),
                     sampler_seed=dict(type='DistSamplerSeedHook'), visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='nccl'))
visualizer = dict(type='DetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')], name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

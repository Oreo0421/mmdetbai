# ============================================================
# RTMDet + MobileViT (timm) on SensIR COCO
# Author: Taijing Bai
# Task: Human Detection (1 class)
# Dataset: SensIR (stitched_png -> COCO)
# ============================================================

_base_ = 'configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'

# ============================================================
# 1. Dataset
# ============================================================

data_root = '/home/tbai/Desktop/sensir_coco/'

metainfo = dict(classes=('human',))
num_classes = 1

train_ann = 'annotations/instances_train.json'
val_ann   = 'annotations/instances_val.json'
test_ann  = 'annotations/instances_test.json'


# ============================================================
# 2. Model
# ============================================================

model = dict(
    backbone=dict(
        _delete_=True,
        type='TimmMobileViT',
        model_name='mobilevit_s',
        out_indices=(2, 3, 4),
        # ⚠️ 不写 pretrained=True
        # 你的 TimmMobileViT 当前 __init__ 不支持这个参数
    ),
    neck=dict(
        _delete_=True,
        type='CSPNeXtPAFPN',
        in_channels=[96, 128, 640],   # ⚠️ 必须与你的 MobileViT 输出一致
        out_channels=96,
        num_csp_blocks=1,
    ),
    bbox_head=dict(
        _delete_=True,
        type='RTMDetHead',
        num_classes=1,
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
)


# ============================================================
# 3. Image size
# ============================================================

# 原始 40x114 → 直接拉伸成正方形，热图检测更稳定
img_scale = (192, 192)


# ============================================================
# 4. Pipelines
# ============================================================

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
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

# ⚠️ 关键修复点：
# JSON 里的 file_name 已经是 "images/train/xxx.png"
# 所以 data_prefix.img 必须是 ''（空字符串）

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann,
        data_prefix=dict(img=''),   # ✅ 必须留空
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
        data_prefix=dict(img=''),   # ✅ 必须留空
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
        data_prefix=dict(img=''),   # ✅ 必须留空
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
    metric='bbox',
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}{test_ann}',
    metric='bbox',
)


# ============================================================
# 7. Training schedule
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
# 8. Optimizer & LR
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


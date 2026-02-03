# ============================================================
# RTMDet Human Detector - WORKING CONFIG (MobileViT backbone)
# Dataset: baseline_split_AAG
# Images: 40x114 (resized/padded to 192x192), single-channel read as 3-channel
# Classes: human (1 class)
# ============================================================

_base_ = 'configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'

# -----------------------------
# 1) Dataset paths & classes
# -----------------------------
data_root = '/home/tbai/Desktop/baseline_split_AAG/'

metainfo = dict(classes=('human',))
num_classes = 1

train_ann = 'annotations/person_keypoints_train.json'
val_ann   = 'annotations/person_keypoints_val.json'
test_ann  = 'annotations/person_keypoints_test.json'

# -----------------------------
# 2) Model (MobileViT -> PAFPN -> RTMDetHead)
#    IMPORTANT: use _delete_=True to avoid leftover params
# -----------------------------
model = dict(
    backbone=dict(
        _delete_=True,
        type='TimmMobileViT',
        model_name='mobilevit_s',
        out_indices=(2, 3, 4),
        init_cfg=None,  # 先跑通；跑通后再加预训练
    ),
    neck=dict(
        _delete_=True,
        type='CSPNeXtPAFPN',
        in_channels=[96, 128, 640],   # ✅ 你检查出来的真实输出通道
        out_channels=96,              # ✅ 统一输出 96
        num_csp_blocks=1,             # ✅ rtmdet_s 常用
    ),
    bbox_head=dict(
        _delete_=True,
    type='RTMDetHead',
    num_classes=1,
    in_channels=96,
    feat_channels=96,
    stacked_convs=2,
    with_objectness=False,

    # ✅ 3 个 level，对应 P3/P4/P5
    anchor_generator=dict(
        type='MlvlPointGenerator',
        offset=0,
        strides=[8, 16, 32],
    ),

    # ✅ 关键修复：用 point-based 的 bbox coder
    bbox_coder=dict(
        type='DistancePointBBoxCoder'
    ),

    # 分类 loss（你前面已经验证这套是对的）
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
    )
)

# -----------------------------
# 3) Image scale
# -----------------------------
img_scale = (192, 192)

# -----------------------------
# 4) Pipelines
# -----------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

# ✅ val/test 不要 LoadAnnotations（评估用 ann_file 读 GT，避免几何不一致问题）
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

# -----------------------------
# 5) Dataloaders
# -----------------------------
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann,
        data_prefix=dict(img='images/train/'),
        filter_cfg=None,
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
        data_prefix=dict(img='images/val/'),
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
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

# -----------------------------
# 6) Evaluators
# -----------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann,
    metric='bbox',
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + test_ann,
    metric='bbox',
)

# -----------------------------
# 7) Hooks
# -----------------------------
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
)


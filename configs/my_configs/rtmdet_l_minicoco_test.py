# configs/my_configs/rtmdet_l_minicoco_test.py

_base_ = ['../rtmdet/rtmdet_l_8xb32-300e_coco.py']

# 你的 mini COCO 根目录（你截图是 Desktop/coco_mini700）
data_root = '/home/tbai/Desktop/coco_mini700/'

# COCO 80类：不要改成 1！
metainfo = dict(classes=(
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
    'clock','vase','scissors','teddy bear','hair drier','toothbrush'
),)

# -------- dataloader：只改路径即可 --------
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017/'),
        metainfo=metainfo,
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/val2017/'),
        metainfo=metainfo,
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='images/test2017/'),
        metainfo=metainfo,
    )
)

# -------- evaluator：也要跟着改 ann_file --------
val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json'
)

test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_test2017.json'
)

# （可选）mini数据集很小，省点显存/速度
val_dataloader.update(dict(batch_size=1, num_workers=2))
test_dataloader.update(dict(batch_size=1, num_workers=2))


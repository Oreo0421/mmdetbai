# 🚀 解决 MMEngine 配置文件错误

## 问题原因

您遇到的错误:
```
ConfigParsingError: The configuration file type in the inheritance chain must match...
```

是因为新版 MMEngine 改变了配置文件继承语法。旧的 `_base_ = '...'` 语法已经废弃。

---

## ✅ 解决方案：使用独立配置文件

我为您创建了 **完全独立的配置文件**，不依赖任何 `_base_`，可以直接使用。

---

## 📦 使用步骤

### 第 1 步：准备文件

```bash
# 1. 创建目录
mkdir -p /home/tbai/rtmdet_pose_ext

# 2. 将以下文件放入该目录:
#    - __init__.py
#    - rtmdet_with_pose.py
#    - heatmap_head.py
#    - keypoint_mse_loss.py
#    - keypoint_transforms_fixed.py (重命名为 keypoint_transforms.py)

cd /home/tbai/rtmdet_pose_ext
mv keypoint_transforms_fixed.py keypoint_transforms.py
```

### 第 2 步：使用独立配置文件

使用 `rtmdet_pose_standalone.py` 配置文件，它：
- ✅ 不依赖任何 `_base_` 文件
- ✅ 包含所有必要的配置
- ✅ 兼容新版 MMEngine

**⚠️ 重要**: 修改配置文件第 7 行的路径:

```python
sys.path.insert(0, '/home/tbai/rtmdet_pose_ext')  # 改为您的实际路径
```

### 第 3 步：开始训练

```bash
cd /path/to/mmdetection

# 训练
python tools/train.py /home/tbai/rtmdet_pose_standalone.py

# 或多卡训练
bash tools/dist_train.sh /home/tbai/rtmdet_pose_standalone.py 4
```

---

## 🔍 验证安装

在训练前测试一下:

```bash
cd /path/to/mmdetection

python -c "
import sys
sys.path.insert(0, '/home/tbai/rtmdet_pose_ext')
import rtmdet_pose_ext

from mmengine.config import Config
cfg = Config.fromfile('/home/tbai/rtmdet_pose_standalone.py')
print('✓ Config loaded successfully!')
print(f'Model type: {cfg.model.type}')
print(f'Backbone: {cfg.model.backbone.type}')
print(f'Pose head: {cfg.model.pose_head.type}')
"
```

如果看到:
```
✓ RTMDet Pose Extension loaded successfully!
✓ Config loaded successfully!
Model type: RTMDetWithPose
Backbone: TimmMobileViT
Pose head: HeatmapHead
```

说明配置正确！

---

## 📝 配置文件说明

`rtmdet_pose_standalone.py` 包含:

1. **数据增强**: CachedMosaic + MixUp (前20 epochs) → 简单增强 (后10 epochs)
2. **优化器**: AdamW (lr=2e-4, weight_decay=1e-4)
3. **学习率**: WarmUp (100 iter) + CosineAnnealing
4. **训练技巧**: 
   - EMA (指数移动平均)
   - Pipeline 切换 (stage2)
   - 梯度裁剪

---

## ⚙️ 可调参数

### 如果想简化训练 (移除高级增强):

修改 `train_pipeline`:

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='GenerateKeypointHeatmap', heatmap_size=(48, 48), sigma=2.0),
    dict(type='PackDetInputs'),
]
```

并删除 `custom_hooks` 中的 `PipelineSwitchHook`。

### 如果 GPU 内存不足:

```python
train_dataloader = dict(
    batch_size=4,  # 从 8 改为 4
    num_workers=2,
)
```

### 如果想调整姿态损失权重:

```python
pose_head=dict(
    ...,
    loss_keypoint=dict(
        type='KeypointMSELoss',
        loss_weight=3.0,  # 增大以更关注姿态
    ),
)
```

---

## 🐛 常见问题

### Q1: 仍然报 "not in registry" 错误

**A**: 检查路径是否正确:

```bash
ls -la /home/tbai/rtmdet_pose_ext/
# 应该看到所有 .py 文件
```

然后确认配置文件开头:

```python
sys.path.insert(0, '/home/tbai/rtmdet_pose_ext')  # 路径正确吗?
import rtmdet_pose_ext  # 这行会打印成功信息
```

### Q2: 找不到 TimmMobileViT

**A**: 您的自定义 MobileViT backbone 需要确保已注册。如果没有，临时使用 RTMDet 自带的 CSPNeXt:

```python
backbone=dict(
    type='CSPNeXt',
    arch='P5',
    expand_ratio=0.5,
    deepen_factor=0.33,
    widen_factor=0.5,
    channel_attention=True,
    norm_cfg=dict(type='BN'),
    act_cfg=dict(type='SiLU', inplace=True),
),
neck=dict(
    type='CSPNeXtPAFPN',
    in_channels=[128, 256, 512],  # CSPNeXt 输出通道
    out_channels=96,
    ...
),
```

### Q3: loss_keypoint 一直是 0

**调试脚本**:

```python
# test_keypoints.py
import sys
sys.path.insert(0, '/home/tbai/rtmdet_pose_ext')
import rtmdet_pose_ext

from mmengine.config import Config
from mmdet.registry import DATASETS

cfg = Config.fromfile('/home/tbai/rtmdet_pose_standalone.py')
dataset = DATASETS.build(cfg.train_dataloader.dataset)

sample = dataset[0]
gt = sample['data_samples'].gt_instances

print("=== Debug Info ===")
print(f"Bboxes: {gt.bboxes.shape if hasattr(gt, 'bboxes') else 'None'}")
print(f"Has keypoints: {hasattr(gt, 'keypoints')}")

if hasattr(gt, 'keypoints'):
    print(f"Keypoints shape: {gt.keypoints.shape}")
    print(f"Keypoints:\n{gt.keypoints}")

print(f"Has heatmap: {hasattr(gt, 'keypoints_heatmap')}")
if hasattr(gt, 'keypoints_heatmap'):
    print(f"Heatmap shape: {gt.keypoints_heatmap.shape}")
    print(f"Heatmap range: [{gt.keypoints_heatmap.min():.4f}, {gt.keypoints_heatmap.max():.4f}]")
```

运行:
```bash
python test_keypoints.py
```

---

## 📊 预期训练日志

```
Epoch [1][20/522]  lr: 2.0000e-04, time: 0.432
loss_cls: 0.7234
loss_bbox: 0.5123
loss_keypoint: 0.0345  ← 关键! 不应该是 0
loss: 1.2702
```

---

## 💾 文件清单

您需要的文件:

```
/home/tbai/
├── rtmdet_pose_ext/          # 自定义模块目录
│   ├── __init__.py           # 模块注册
│   ├── rtmdet_with_pose.py   # 检测器
│   ├── heatmap_head.py       # 姿态头
│   ├── keypoint_mse_loss.py  # 损失函数
│   └── keypoint_transforms.py # 数据处理
│
└── rtmdet_pose_standalone.py # 配置文件
```

---

## 🎯 快速检查清单

训练前确认:

- [ ] 所有 `.py` 文件在 `/home/tbai/rtmdet_pose_ext/`
- [ ] `rtmdet_pose_standalone.py` 中的路径已修改
- [ ] 运行验证脚本成功
- [ ] 数据路径正确 (`/home/tbai/Desktop/sensir_coco/`)
- [ ] JSON 文件存在且包含 keypoints

全部 ✅ 后开始训练！

---

需要更多帮助? 请提供:
1. 完整错误信息
2. `test_keypoints.py` 的输出
3. MMDetection 版本: `python -c "import mmdet; print(mmdet.__version__)"`

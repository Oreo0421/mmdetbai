#!/usr/bin/env python3
"""诊断关键点数据加载问题"""

import sys
import json
sys.path.insert(0, '/home/tbai/mmdetection/mmdetection')

print("=== 步骤 1: 检查原始数据 ===")
data_root = '/home/tbai/Desktop/sensir_coco/'
ann_file = data_root + 'annotations/instances_train.json'

with open(ann_file, 'r') as f:
    data = json.load(f)

print(f"图像数量: {len(data['images'])}")
print(f"标注数量: {len(data['annotations'])}")

# 检查第一个标注
first_ann = data['annotations'][0]
print(f"\n第一个标注:")
print(f"  image_id: {first_ann.get('image_id')}")
print(f"  bbox: {first_ann.get('bbox')}")
print(f"  有 keypoints: {'keypoints' in first_ann}")

if 'keypoints' in first_ann:
    kpts = first_ann['keypoints']
    print(f"  keypoints 长度: {len(kpts)} (应该是 21 = 7*3)")
    print(f"  keypoints 前6个: {kpts[:6]}")
else:
    print("  ❌ 没有 keypoints 字段!")

print("\n=== 步骤 2: 测试数据加载 ===")

# 清除缓存
for m in list(sys.modules.keys()):
    if 'rtmdet_pose_ext' in m:
        del sys.modules[m]

from mmengine.config import Config
from mmdet.registry import DATASETS

cfg = Config.fromfile('/home/tbai/mmdetection/mmdetection/rtmdet_pose_fixed_pipeline.py')

print(f"\nPipeline transforms:")
for i, t in enumerate(cfg.train_pipeline):
    print(f"  {i}: {t['type']}")

print("\n构建 dataset...")
dataset = DATASETS.build(cfg.train_dataloader.dataset)

print(f"Dataset 大小: {len(dataset)}")

print("\n获取第一个样本...")
sample = dataset[0]

print("\n样本内容:")
print(f"  inputs shape: {sample['inputs'].shape}")

gt = sample['data_samples'].gt_instances

print(f"\nGT instances 属性:")
for attr in dir(gt):
    if not attr.startswith('_'):
        try:
            val = getattr(gt, attr)
            if not callable(val):
                print(f"  {attr}: {type(val)} {getattr(val, 'shape', '')}")
        except:
            pass

print(f"\n关键检查:")
print(f"  Has bboxes: {hasattr(gt, 'bboxes')}")
print(f"  Has keypoints: {hasattr(gt, 'keypoints')}")
print(f"  Has keypoints_heatmap: {hasattr(gt, 'keypoints_heatmap')}")
print(f"  Has keypoint_weights: {hasattr(gt, 'keypoint_weights')}")

if hasattr(gt, 'keypoints'):
    print(f"  Keypoints shape: {gt.keypoints.shape}")
    print(f"  Keypoints:\n{gt.keypoints}")

if hasattr(gt, 'keypoints_heatmap'):
    print(f"  Heatmap shape: {gt.keypoints_heatmap.shape}")
    print(f"  Heatmap max: {gt.keypoints_heatmap.max():.4f}")
    print(f"  Heatmap min: {gt.keypoints_heatmap.min():.4f}")
else:
    print("  ❌ 没有生成热图!")

if hasattr(gt, 'keypoint_weights'):
    print(f"  Weights: {gt.keypoint_weights}")
else:
    print("  ❌ 没有生成权重!")

print("\n=== 诊断完成 ===")

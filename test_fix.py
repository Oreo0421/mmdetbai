#!/usr/bin/env python
"""测试 GeneratePoseHeatmap 和 PackDetInputsWithPose 是否正常工作"""

import numpy as np
import torch

# 模拟 results 数据（来自 pipeline 中间状态）
def create_mock_results():
    """创建模拟的 pipeline 中间结果"""
    return {
        'img_path': '/path/to/image.jpg',
        'img_id': 1,
        'height': 480,
        'width': 640,
        'img_shape': (480, 640, 3),
        'ori_shape': (480, 640, 3),
        'img': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        'instances': [
            {
                'ignore_flag': False,
                'bbox': [100, 50, 200, 300],  # x1, y1, x2, y2
                'bbox_label': 0,
                # 7 个关键点: [x1, y1, v1, x2, y2, v2, ...]
                'keypoints': [
                    150, 60, 2,    # 0: 头顶
                    150, 80, 2,    # 1: 颈部
                    130, 100, 2,   # 2: 左肩
                    170, 100, 2,   # 3: 右肩
                    150, 150, 2,   # 4: 躯干中心
                    135, 200, 2,   # 5: 左髋
                    165, 200, 2,   # 6: 右髋
                ]
            }
        ],
        'gt_bboxes': np.array([[100, 50, 200, 300]], dtype=np.float32),
        'gt_bboxes_labels': np.array([0], dtype=np.int64),
        'gt_ignore_flags': np.array([False], dtype=bool),
    }


def test_generate_pose_heatmap():
    """测试 GeneratePoseHeatmap transform"""
    print("\n" + "=" * 60)
    print("测试 GeneratePoseHeatmap")
    print("=" * 60)
    
    from keypoint_transforms import GeneratePoseHeatmap
    
    # 创建 transform
    transform = GeneratePoseHeatmap(heatmap_size=(48, 48), sigma=2.0)
    print(f"Transform: {transform}")
    
    # 创建模拟数据
    results = create_mock_results()
    print(f"\n输入 results keys: {list(results.keys())}")
    print(f"instances[0] keys: {list(results['instances'][0].keys())}")
    print(f"keypoints: {results['instances'][0]['keypoints']}")
    
    # 执行 transform
    results = transform(results)
    
    # 检查结果
    print(f"\n输出 results keys: {list(results.keys())}")
    
    assert 'gt_keypoints' in results, "❌ 缺少 gt_keypoints"
    assert 'gt_keypoints_heatmap' in results, "❌ 缺少 gt_keypoints_heatmap"
    
    kpts = results['gt_keypoints']
    heatmap = results['gt_keypoints_heatmap']
    
    print(f"\n✓ gt_keypoints shape: {kpts.shape}")  # 期望 (7, 3)
    print(f"✓ gt_keypoints_heatmap shape: {heatmap.shape}")  # 期望 (7, 48, 48)
    print(f"✓ heatmap max values per channel: {[f'{heatmap[i].max():.4f}' for i in range(7)]}")
    print(f"✓ 所有热图都有值: {all(heatmap[i].max() > 0 for i in range(7))}")
    
    return results


def test_pack_det_inputs_with_pose(results):
    """测试 PackDetInputsWithPose"""
    print("\n" + "=" * 60)
    print("测试 PackDetInputsWithPose")
    print("=" * 60)
    
    from custom_pack import PackDetInputsWithPose
    
    # 创建 transform
    transform = PackDetInputsWithPose()
    print(f"Transform: {transform}")
    
    # 执行 transform
    packed = transform(results)
    
    print(f"\n输出 packed keys: {list(packed.keys())}")
    
    data_samples = packed['data_samples']
    print(f"data_samples type: {type(data_samples)}")
    
    gt_instances = data_samples.gt_instances
    print(f"gt_instances type: {type(gt_instances)}")
    
    # 检查关键点
    has_kpts = hasattr(gt_instances, 'keypoints')
    has_heatmap = hasattr(gt_instances, 'keypoints_heatmap')
    
    print(f"\n✓ gt_instances.keypoints 存在: {has_kpts}")
    print(f"✓ gt_instances.keypoints_heatmap 存在: {has_heatmap}")
    
    if has_kpts:
        kpts = gt_instances.keypoints
        print(f"  - keypoints type: {type(kpts)}")
        print(f"  - keypoints shape: {kpts.shape}")
        print(f"  - keypoints dtype: {kpts.dtype}")
    
    if has_heatmap:
        heatmap = gt_instances.keypoints_heatmap
        print(f"  - keypoints_heatmap type: {type(heatmap)}")
        print(f"  - keypoints_heatmap shape: {heatmap.shape}")
        print(f"  - keypoints_heatmap max: {heatmap.max():.4f}")
    
    return packed


def main():
    print("\n" + "=" * 60)
    print("RTMDet Pose Extension - 修复测试")
    print("=" * 60)
    
    # 测试 GeneratePoseHeatmap
    results = test_generate_pose_heatmap()
    
    # 测试 PackDetInputsWithPose
    packed = test_pack_det_inputs_with_pose(results)
    
    # 最终验证
    print("\n" + "=" * 60)
    print("最终验证")
    print("=" * 60)
    
    gt = packed['data_samples'].gt_instances
    
    success = (
        hasattr(gt, 'keypoints') and 
        hasattr(gt, 'keypoints_heatmap') and
        gt.keypoints_heatmap.max() > 0
    )
    
    if success:
        print("\n✅ 所有测试通过！修复成功！")
        print("\n现在你可以:")
        print("1. 将这些文件复制到 rtmdet_pose_ext/ 目录")
        print("2. 更新配置使用 PackDetInputsWithPose")
        print("3. 运行训练")
    else:
        print("\n❌ 测试失败，请检查代码")
    
    return success


if __name__ == '__main__':
    main()

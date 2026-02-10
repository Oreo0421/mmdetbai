#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTMDet-Pose 推理脚本 - 修正版
修复：使用 mmdet 初始化，自定义推理 pipeline（无需标注）
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import numpy as np
import glob
import torch
import argparse

from mmdet.apis import init_detector
from mmengine.dataset import Compose


def parse_args():
    parser = argparse.ArgumentParser(description='RTMDet-Pose 推理脚本 - 7关键点')
    parser.add_argument('--config',
                       default='/home/tbai/mmdetection/mmdetection/work_dirs/rtmdet_pose_clean/rtmdet_pose_clean.py',
                       help='配置文件路径')
    parser.add_argument('--checkpoint',
                       default='//home/tbai/mmdetection/mmdetection/work_dirs/rtmdet_pose_0206/best_coco_bbox_mAP_epoch_29.pth',
                       help='权重文件路径')
    parser.add_argument('--input',
                       default='/home/tbai/Desktop/SenIRDatasetProcessed/1A-2 Walking/AAG/stitched_png',
                       help='输入图片文件夹路径')
    parser.add_argument('--output',
                       default='/home/tbai/Desktop/inference_results',
                       help='输出文件夹路径')
    parser.add_argument('--det-thr',
                       type=float,
                       default=0.3,
                       help='检测框置信度阈值')
    parser.add_argument('--kpt-thr',
                       type=float,
                       default=0.3,
                       help='关键点置信度阈值')
    parser.add_argument('--show-kpt-idx',
                       action='store_true',
                       help='是否显示关键点编号')
    parser.add_argument('--thickness',
                       type=int,
                       default=2,
                       help='线条粗细')
    parser.add_argument('--radius',
                       type=int,
                       default=2,
                       help='关键点半径')
    return parser.parse_args()


# ---- 推理专用 pipeline（不需要标注和GT热图）----
INFER_PIPELINE = [
    dict(type='LoadImageFromFile', _scope_='mmdet'),
    dict(type='Resize', scale=(192, 192), keep_ratio=False, _scope_='mmdet'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
         _scope_='mmdet'),
]


def inference_single_image(model, img_path, pipeline):
    """对单张图像进行推理"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    # 构造数据字典（模拟 dataset 的输出）
    data_info = {
        'img_path': img_path,
        'img_id': 0,
        'ori_shape': (h, w),
        'height': h,
        'width': w,
    }

    # 经过 pipeline：LoadImage → Resize → Pack
    data = pipeline(data_info)

    # 送入模型
    device = next(model.parameters()).device
    data['inputs'] = [data['inputs'].to(device)]
    data['data_samples'] = [data['data_samples'].to(device)]

    with torch.no_grad():
        results = model.test_step(data)

    return results, img


def visualize_7_keypoints(img, bboxes, bbox_scores, keypoints, kpt_scores, args):
    """可视化检测框和7个关键点"""
    img_vis = img.copy()

    colors = [
        (0, 0, 255),    # 红
        (0, 128, 255),  # 橙
        (0, 255, 255),  # 黄
        (0, 255, 0),    # 绿
        (255, 255, 0),  # 青
        (255, 0, 0),    # 蓝
        (255, 0, 255),  # 紫
    ]

    for inst_idx in range(len(bboxes)):
        bbox = bboxes[inst_idx]
        det_score = bbox_scores[inst_idx]

        # 画检测框
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), args.thickness)
        cv2.putText(img_vis, f'{det_score:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 画关键点
        if inst_idx < len(keypoints):
            kpts = keypoints[inst_idx]
            scrs = kpt_scores[inst_idx]
            for i, (kpt, score) in enumerate(zip(kpts, scrs)):
                if score > args.kpt_thr:
                    x, y = int(kpt[0]), int(kpt[1])
                    color = colors[i % len(colors)]
                    cv2.circle(img_vis, (x, y), args.radius, color, -1)

                    if args.show_kpt_idx:
                        text = f"{i}:{score:.2f}"
                        cv2.putText(img_vis, text, (x + 8, y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(img_vis, text, (x + 8, y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img_vis


def process_image(model, img_path, output_dir, pipeline, args):
    """处理单张图片"""
    try:
        result_tuple = inference_single_image(model, img_path, pipeline)

        if result_tuple is None:
            print("⚠️  读取失败")
            return False

        results, img = result_tuple

        if len(results) == 0:
            print("⚠️  未检测到")
            return False

        result = results[0]
        pred = result.pred_instances

        # 过滤低置信度检测
        if hasattr(pred, 'scores'):
            keep = pred.scores > args.det_thr
            pred = pred[keep]

        if len(pred) == 0:
            print("⚠️  无有效检测")
            return False

        bboxes = pred.bboxes.cpu().numpy()
        bbox_scores = pred.scores.cpu().numpy()

        keypoints = pred.keypoints.cpu().numpy() if hasattr(pred, 'keypoints') else np.array([])
        kpt_scores = pred.keypoint_scores.cpu().numpy() if hasattr(pred, 'keypoint_scores') else np.array([])

        # 可视化
        img_vis = visualize_7_keypoints(img, bboxes, bbox_scores, keypoints, kpt_scores, args)

        # 保存图片
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img_vis)

        # 保存坐标txt
        txt_path = os.path.join(output_dir, 'keypoints_' + os.path.splitext(filename)[0] + '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"图像: {filename}\n")
            f.write(f"检测到 {len(bboxes)} 个目标\n")
            f.write("=" * 60 + "\n")
            for inst_idx in range(len(bboxes)):
                f.write(f"\n--- 目标 {inst_idx} (det_score={bbox_scores[inst_idx]:.3f}) ---\n")
                bbox = bboxes[inst_idx]
                f.write(f"  bbox: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})\n")
                if inst_idx < len(keypoints):
                    for i in range(min(len(keypoints[inst_idx]), 7)):
                        x, y = keypoints[inst_idx][i]
                        score = kpt_scores[inst_idx][i]
                        f.write(f"  kpt[{i}]: ({x:.2f}, {y:.2f}), score={score:.3f}\n")

        print("✓")
        return True

    except Exception as e:
        print(f"✗ {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    print("=" * 70)
    print("RTMDet-Pose 推理脚本 - 7关键点")
    print("=" * 70)
    print(f"配置文件: {args.config}")
    print(f"权重文件: {args.checkpoint}")
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")

    if torch.cuda.is_available():
        print(f"使用设备: GPU ({torch.cuda.get_device_name(0)})")
        device = 'cuda:0'
    else:
        print("使用设备: CPU")
        device = 'cpu'

    print(f"检测阈值: {args.det_thr}")
    print(f"关键点阈值: {args.kpt_thr}")
    print("=" * 70)

    if not os.path.isdir(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)

    # ---- 加载模型（mmdet） ----
    print("\n正在加载模型...")
    try:
        model = init_detector(args.config, args.checkpoint, device=device)
        model.eval()
        print("✓ 模型加载成功!")
    except Exception as e:
        print(f"✗ 模型加载失败: {str(e)}")
        return

    # ---- 构建推理 pipeline ----
    pipeline = Compose(INFER_PIPELINE)

    # ---- 获取图片列表 ----
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(args.input, f'*{ext.upper()}')))
    image_files = sorted(list(set(image_files)))

    if len(image_files) == 0:
        print(f"\n错误: 未找到图片")
        return

    print(f"\n找到 {len(image_files)} 张图片")
    print("开始推理...\n")

    success_count = 0
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {os.path.basename(img_path)} ", end="")
        if process_image(model, img_path, args.output, pipeline, args):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"推理完成!")
    print(f"成功: {success_count}/{len(image_files)} ({success_count * 100 // len(image_files)}%)")
    print(f"失败: {len(image_files) - success_count}")
    print(f"结果保存在: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()

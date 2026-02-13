#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTMDet-Pose V6 (SimDR) 推理脚本
使用 SimDR 1D classification 解码关键点
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
    parser = argparse.ArgumentParser(description='RTMDet-Pose V6 SimDR 推理脚本')
    parser.add_argument('--config',
                       default='/home/tbai/mmdetection/mmdetection/rtmdet_pose_v6.py',
                       help='配置文件路径')
    parser.add_argument('--checkpoint',
                       default='/home/tbai/mmdetection/mmdetection/work_dirs/rtmdet_pose_v6/best_coco_bbox_mAP_epoch_4.pth',
                       help='权重文件路径')
    parser.add_argument('--input',
                       default='/home/tbai/Desktop/SenIRDatasetProcessed/1A-2 Walking/AAG/stitched_png',
                       help='输入图片文件夹路径')
    parser.add_argument('--output',
                       default='/home/tbai/Desktop/inference_results_v6',
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
    parser.add_argument('--skeleton',
                       type=str,
                       nargs='*',
                       default=None,
                       help='骨架连接，格式: 0-1 1-2 1-3 ... 不传则不画骨架')
    parser.add_argument('--thickness',
                       type=int,
                       default=2,
                       help='线条/骨架粗细')
    parser.add_argument('--radius',
                       type=int,
                       default=3,
                       help='关键点显示半径')
    args = parser.parse_args()

    # 解析 skeleton: "0-1" -> (0, 1)
    if args.skeleton is not None:
        parsed = []
        for s in args.skeleton:
            a, b = s.split('-')
            parsed.append((int(a), int(b)))
        args.skeleton = parsed

    return args


# ---- 推理专用 pipeline（不需要标注和GT）----
INFER_PIPELINE = [
    dict(type='LoadImageFromFile', _scope_='mmdet'),
    dict(type='Resize', scale=(384, 384), keep_ratio=False, _scope_='mmdet'),
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

    data_info = {
        'img_path': img_path,
        'img_id': 0,
        'ori_shape': (h, w),
        'height': h,
        'width': w,
    }

    data = pipeline(data_info)

    device = next(model.parameters()).device
    data['inputs'] = [data['inputs'].to(device)]
    data['data_samples'] = [data['data_samples'].to(device)]

    with torch.no_grad():
        results = model.test_step(data)

    return results, img


KPT_COLORS = [
    (0, 0, 255),    # 0: 红
    (0, 128, 255),  # 1: 橙
    (0, 255, 255),  # 2: 黄
    (0, 255, 0),    # 3: 绿
    (255, 255, 0),  # 4: 青
    (255, 0, 0),    # 5: 蓝
    (255, 0, 255),  # 6: 紫
]

LIMB_COLOR = (0, 255, 128)


def visualize_pose(img, bboxes, bbox_scores, keypoints, kpt_scores, args):
    """可视化检测框、关键点和骨架"""
    img_vis = img.copy()

    for inst_idx in range(len(bboxes)):
        bbox = bboxes[inst_idx]
        det_score = bbox_scores[inst_idx]

        # 画检测框
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), args.thickness)
        cv2.putText(img_vis, f'{det_score:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if inst_idx >= len(keypoints):
            continue

        kpts = keypoints[inst_idx]
        scrs = kpt_scores[inst_idx]

        # 画骨架连接线（仅当用户指定 --skeleton 时）
        if args.skeleton:
            for (a, b) in args.skeleton:
                if a < len(kpts) and b < len(kpts):
                    if scrs[a] > args.kpt_thr and scrs[b] > args.kpt_thr:
                        pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                        pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                        cv2.line(img_vis, pt1, pt2, LIMB_COLOR, args.thickness)

        # 画关键点
        for i, (kpt, score) in enumerate(zip(kpts, scrs)):
            if score > args.kpt_thr:
                x, y = int(kpt[0]), int(kpt[1])
                color = KPT_COLORS[i % len(KPT_COLORS)]
                cv2.circle(img_vis, (x, y), args.radius + 1, (255, 255, 255), -1)
                cv2.circle(img_vis, (x, y), args.radius, color, -1)

                if args.show_kpt_idx:
                    text = f"{i}:{score:.2f}"
                    cv2.putText(img_vis, text, (x + 8, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                    cv2.putText(img_vis, text, (x + 8, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img_vis


def process_image(model, img_path, output_dir, pipeline, args):
    """处理单张图片"""
    try:
        result_tuple = inference_single_image(model, img_path, pipeline)

        if result_tuple is None:
            print("  读取失败")
            return False

        results, img = result_tuple

        if len(results) == 0:
            print("  未检测到")
            return False

        result = results[0]
        pred = result.pred_instances

        # 过滤低置信度检测
        if hasattr(pred, 'scores'):
            keep = pred.scores > args.det_thr
            pred = pred[keep]

        if len(pred) == 0:
            print("  无有效检测")
            return False

        bboxes = pred.bboxes.cpu().numpy()
        bbox_scores = pred.scores.cpu().numpy()

        keypoints = pred.keypoints.cpu().numpy() if hasattr(pred, 'keypoints') else np.array([])
        kpt_scores = pred.keypoint_scores.cpu().numpy() if hasattr(pred, 'keypoint_scores') else np.array([])

        # 可视化
        img_vis = visualize_pose(img, bboxes, bbox_scores, keypoints, kpt_scores, args)

        # 保存图片
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img_vis)

        # 保存坐标txt
        txt_path = os.path.join(output_dir, 'keypoints_' + os.path.splitext(filename)[0] + '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Image: {filename}\n")
            f.write(f"Model: SimDR V6\n")
            f.write(f"Detections: {len(bboxes)}\n")
            f.write("=" * 60 + "\n")
            for inst_idx in range(len(bboxes)):
                f.write(f"\n--- Instance {inst_idx} (det_score={bbox_scores[inst_idx]:.3f}) ---\n")
                bbox = bboxes[inst_idx]
                f.write(f"  bbox: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})\n")
                if inst_idx < len(keypoints):
                    for i in range(min(len(keypoints[inst_idx]), 7)):
                        x, y = keypoints[inst_idx][i]
                        score = kpt_scores[inst_idx][i]
                        vis_str = "visible" if score > args.kpt_thr else "low_conf"
                        f.write(f"  kpt[{i}]: ({x:.2f}, {y:.2f}), score={score:.3f} [{vis_str}]\n")

        print(f"  {len(bboxes)} dets, {(kpt_scores > args.kpt_thr).sum()} kpts visible")
        return True

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    print("=" * 70)
    print("RTMDet-Pose V6 (SimDR) Inference")
    print("=" * 70)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")

    if torch.cuda.is_available():
        print(f"Device:     GPU ({torch.cuda.get_device_name(0)})")
        device = 'cuda:0'
    else:
        print("Device:     CPU")
        device = 'cpu'

    print(f"Det thr:    {args.det_thr}")
    print(f"Kpt thr:    {args.kpt_thr}")
    print(f"Radius:     {args.radius}")
    print(f"Skeleton:   {args.skeleton if args.skeleton else 'None (no bones drawn)'}")
    print("=" * 70)

    if not os.path.isdir(args.input):
        print(f"Error: input dir not found: {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)

    # ---- 加载模型 ----
    print("\nLoading model...")
    try:
        model = init_detector(args.config, args.checkpoint, device=device)
        model.eval()
        print("Model loaded!")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
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
        print(f"\nNo images found in {args.input}")
        return

    print(f"\nFound {len(image_files)} images")
    print("Starting inference...\n")

    success_count = 0
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {os.path.basename(img_path)}", end="")
        if process_image(model, img_path, args.output, pipeline, args):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"Done! {success_count}/{len(image_files)} succeeded")
    print(f"Results saved to: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()

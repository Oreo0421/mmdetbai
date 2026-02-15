# -*- coding: utf-8 -*-
"""
ActionMetric & FallingMetric: Evaluation metrics for action classification.

ActionMetric: 10-class action classification + binary falling (grouped).
FallingMetric: Binary falling detection (legacy, kept for backward compat).
"""
from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import load
from mmengine.logging import MMLogger

from mmdet.registry import METRICS


ACTION_NAMES = [
    'Standing still',
    'Walking',
    'Sitting down',
    'Standing up',
    'Lying down',
    'Getting up',
    'Falling walking',
    'Falling standing',
    'Falling sitting',
    'Falling standing up',
]
FALLING_CLASS_IDS = {6, 7, 8, 9}


@METRICS.register_module(force=True)
class ActionMetric(BaseMetric):
    """10-class action classification metric with binary falling grouping.

    Matches predicted bboxes to GT via IoU, collects (pred_class, gt_class)
    pairs, and computes:
      - 10-class confusion matrix, per-class precision/recall/F1, accuracy
      - Binary falling metrics (classes 6-9 grouped) with precision/recall/F1

    Args:
        ann_file (str): Path to COCO annotation file.
        iou_thr (float): IoU threshold for matching. Default: 0.5.
        num_classes (int): Number of action classes. Default: 10.
        collect_device (str): Device for collecting results. Default: 'cpu'.
        prefix (str): Metric prefix. Default: 'action'.
    """

    def __init__(
        self,
        ann_file: str,
        iou_thr: float = 0.5,
        num_classes: int = 10,
        collect_device: str = 'cpu',
        prefix: str = 'action',
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = iou_thr
        self.num_classes = num_classes

        self.ann_file = ann_file
        self._gt_data = {}  # img_id -> list of {bbox, action_class}
        self._load_gt()

    def _load_gt(self):
        """Load GT action_class labels from COCO annotation file."""
        data = load(self.ann_file)
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            attributes = ann.get('attributes', {})
            if not isinstance(attributes, dict):
                attributes = {}

            # Prefer action_class, fall back to falling
            if 'action_class' in attributes:
                action_class = int(attributes['action_class'])
            else:
                falling = int(attributes.get('falling', 0))
                action_class = 6 if falling else 0  # rough mapping

            if img_id not in self._gt_data:
                self._gt_data[img_id] = []
            self._gt_data[img_id].append({
                'bbox': ann['bbox'],  # [x, y, w, h] COCO format
                'action_class': action_class,
            })

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch."""
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            img_id = data_sample['img_id']

            if 'action_scores' not in pred:
                continue

            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            action_scores = pred['action_scores'].cpu().numpy()

            # Determine predicted class
            if action_scores.ndim == 2 and action_scores.shape[-1] > 1:
                # Multi-class: (M, C) probabilities
                pred_classes = action_scores.argmax(axis=-1)
                fall_probs = action_scores[:, 6:].sum(axis=-1)
            else:
                # Binary: (M,) falling probability
                action_scores = action_scores.flatten()
                pred_classes = (action_scores >= 0.5).astype(int) * 6
                fall_probs = action_scores

            gt_list = self._gt_data.get(img_id, [])
            if len(gt_list) == 0 or len(pred_bboxes) == 0:
                continue

            gt_bboxes = np.array([
                [g['bbox'][0], g['bbox'][1],
                 g['bbox'][0] + g['bbox'][2], g['bbox'][1] + g['bbox'][3]]
                for g in gt_list
            ])
            gt_classes = np.array([g['action_class'] for g in gt_list])

            ious = self._compute_iou(pred_bboxes, gt_bboxes)

            gt_matched = set()
            for pred_idx in np.argsort(-pred_scores):
                if ious.shape[1] == 0:
                    break
                best_gt = ious[pred_idx].argmax()
                if (ious[pred_idx, best_gt] >= self.iou_thr
                        and best_gt not in gt_matched):
                    gt_matched.add(best_gt)
                    self.results.append({
                        'pred_class': int(pred_classes[pred_idx]),
                        'gt_class': int(gt_classes[best_gt]),
                        'fall_prob': float(fall_probs[pred_idx]),
                        'det_score': float(pred_scores[pred_idx]),
                    })

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute action classification metrics."""
        logger = MMLogger.get_current_instance()

        if len(results) == 0:
            logger.warning('ActionMetric: No matched predictions found.')
            return OrderedDict(accuracy=0.0, fall_f1=0.0)

        pred_classes = np.array([r['pred_class'] for r in results])
        gt_classes = np.array([r['gt_class'] for r in results])
        fall_probs = np.array([r['fall_prob'] for r in results])

        N = self.num_classes

        # ---- 10-class confusion matrix ----
        cm = np.zeros((N, N), dtype=int)
        for gt, pred in zip(gt_classes, pred_classes):
            if 0 <= gt < N and 0 <= pred < N:
                cm[gt, pred] += 1

        # Overall accuracy
        total = len(results)
        correct = int((pred_classes == gt_classes).sum())
        accuracy = correct / max(total, 1)

        # Per-class metrics
        header = f'\n{"":>20s} {"Prec":>6s} {"Recall":>6s} {"F1":>6s} {"Support":>8s}'
        lines = [header]
        per_class_f1 = []
        for c in range(N):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            support = int(cm[c, :].sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            per_class_f1.append(f1)
            name = ACTION_NAMES[c] if c < len(ACTION_NAMES) else f'class_{c}'
            fall_marker = ' *' if c in FALLING_CLASS_IDS else ''
            lines.append(
                f'{name:>20s} {prec:6.3f} {rec:6.3f} {f1:6.3f} {support:8d}{fall_marker}')

        macro_f1 = np.mean(per_class_f1)

        # ---- Binary falling (grouped classes 6-9) ----
        gt_falling = np.array([1 if c in FALLING_CLASS_IDS else 0 for c in gt_classes])
        pred_falling = np.array([1 if c in FALLING_CLASS_IDS else 0 for c in pred_classes])

        fall_tp = int(((pred_falling == 1) & (gt_falling == 1)).sum())
        fall_fp = int(((pred_falling == 1) & (gt_falling == 0)).sum())
        fall_fn = int(((pred_falling == 0) & (gt_falling == 1)).sum())
        fall_tn = int(((pred_falling == 0) & (gt_falling == 0)).sum())

        fall_prec = fall_tp / max(fall_tp + fall_fp, 1)
        fall_rec = fall_tp / max(fall_tp + fall_fn, 1)
        fall_f1 = 2 * fall_prec * fall_rec / max(fall_prec + fall_rec, 1e-8)
        fall_acc = (fall_tp + fall_tn) / max(total, 1)

        # AP for falling detection
        fall_ap = self._compute_ap(fall_probs, gt_falling)

        lines.append(f'\n--- 10-class Summary ---')
        lines.append(f'  Accuracy:  {accuracy:.4f}')
        lines.append(f'  Macro-F1:  {macro_f1:.4f}')
        lines.append(f'\n--- Binary Falling (classes 6-9 grouped) ---')
        lines.append(f'  TP={fall_tp} FP={fall_fp} TN={fall_tn} FN={fall_fn}')
        lines.append(f'  Accuracy:  {fall_acc:.4f}')
        lines.append(f'  Precision: {fall_prec:.4f}')
        lines.append(f'  Recall:    {fall_rec:.4f}')
        lines.append(f'  F1:        {fall_f1:.4f}')
        lines.append(f'  AP:        {fall_ap:.4f}')

        logger.info('\n--- Action Classification Metrics ---' + '\n'.join(lines)
                     + '\n' + '-' * 40)

        metrics = OrderedDict(
            accuracy=round(accuracy, 4),
            macro_f1=round(macro_f1, 4),
            fall_accuracy=round(fall_acc, 4),
            fall_precision=round(fall_prec, 4),
            fall_recall=round(fall_rec, 4),
            fall_f1=round(fall_f1, 4),
            fall_ap=round(fall_ap, 4),
        )
        # Per-class F1 for monitoring
        for c in range(N):
            name = ACTION_NAMES[c] if c < len(ACTION_NAMES) else f'class_{c}'
            name = name.lower().replace(' ', '_')
            metrics[f'f1_{name}'] = round(per_class_f1[c], 4)

        return metrics

    @staticmethod
    def _compute_ap(scores, labels):
        """Compute Average Precision (AP) for binary classification."""
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]
        num_pos = int(sorted_labels.sum())
        if num_pos == 0:
            return 0.0
        tp_cumsum = np.cumsum(sorted_labels)
        fp_cumsum = np.cumsum(1 - sorted_labels)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / num_pos
        precisions = np.concatenate([[1.0], precisions])
        recalls = np.concatenate([[0.0], recalls])
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        recall_diff = np.diff(recalls)
        return float(np.sum(recall_diff * precisions[1:]))

    @staticmethod
    def _compute_iou(bboxes1, bboxes2):
        """Compute IoU between two sets of bboxes (xyxy format)."""
        x1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0:1].T)
        y1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1:2].T)
        x2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2:3].T)
        y2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3:4].T)
        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        return inter / np.maximum(union, 1e-8)


@METRICS.register_module(force=True)
class FallingMetric(BaseMetric):
    """Binary falling detection metric (legacy).

    Kept for backward compatibility. For new configs, use ActionMetric.
    """

    def __init__(
        self,
        ann_file: str,
        iou_thr: float = 0.5,
        score_thr: float = 0.5,
        collect_device: str = 'cpu',
        prefix: str = 'falling',
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = iou_thr
        self.score_thr = score_thr

        self.ann_file = ann_file
        self._gt_falling = {}
        self._load_gt_falling()

    def _load_gt_falling(self):
        """Load GT falling labels from COCO annotation file."""
        data = load(self.ann_file)
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            attributes = ann.get('attributes', {})
            if isinstance(attributes, dict):
                if 'action_class' in attributes:
                    falling = int(int(attributes['action_class']) >= 6)
                else:
                    falling = int(attributes.get('falling', 0))
            else:
                falling = 0

            if img_id not in self._gt_falling:
                self._gt_falling[img_id] = []
            self._gt_falling[img_id].append({
                'bbox': ann['bbox'],
                'falling': falling,
            })

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch."""
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            img_id = data_sample['img_id']

            if 'action_scores' not in pred:
                continue

            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            action_scores = pred['action_scores'].cpu().numpy()

            # Get falling probability
            if action_scores.ndim == 2 and action_scores.shape[-1] > 1:
                fall_probs = action_scores[:, 6:].sum(axis=-1)
            else:
                fall_probs = action_scores.flatten()

            gt_list = self._gt_falling.get(img_id, [])
            if len(gt_list) == 0 or len(pred_bboxes) == 0:
                continue

            gt_bboxes = np.array([
                [g['bbox'][0], g['bbox'][1],
                 g['bbox'][0] + g['bbox'][2], g['bbox'][1] + g['bbox'][3]]
                for g in gt_list
            ])
            gt_falling = np.array([g['falling'] for g in gt_list])

            ious = ActionMetric._compute_iou(pred_bboxes, gt_bboxes)

            gt_matched = set()
            for pred_idx in np.argsort(-pred_scores):
                if ious.shape[1] == 0:
                    break
                best_gt = ious[pred_idx].argmax()
                if (ious[pred_idx, best_gt] >= self.iou_thr
                        and best_gt not in gt_matched):
                    gt_matched.add(best_gt)
                    self.results.append({
                        'action_score': float(fall_probs[pred_idx]),
                        'gt_falling': int(gt_falling[best_gt]),
                        'det_score': float(pred_scores[pred_idx]),
                    })

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute falling detection metrics."""
        logger = MMLogger.get_current_instance()

        if len(results) == 0:
            logger.warning('FallingMetric: No matched predictions found.')
            return OrderedDict(
                accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, ap=0.0)

        action_scores = np.array([r['action_score'] for r in results])
        gt_labels = np.array([r['gt_falling'] for r in results])

        pred_labels = (action_scores >= self.score_thr).astype(int)

        tp = int(((pred_labels == 1) & (gt_labels == 1)).sum())
        fp = int(((pred_labels == 1) & (gt_labels == 0)).sum())
        tn = int(((pred_labels == 0) & (gt_labels == 0)).sum())
        fn = int(((pred_labels == 0) & (gt_labels == 1)).sum())

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        ap = ActionMetric._compute_ap(action_scores, gt_labels)

        logger.info(
            f'\n--- Falling Detection Metrics ---\n'
            f'  Samples: {total} (pos={int(gt_labels.sum())}, '
            f'neg={int((gt_labels == 0).sum())})\n'
            f'  TP={tp} FP={fp} TN={tn} FN={fn}\n'
            f'  Accuracy:  {accuracy:.4f}\n'
            f'  Precision: {precision:.4f}\n'
            f'  Recall:    {recall:.4f}\n'
            f'  F1:        {f1:.4f}\n'
            f'  AP:        {ap:.4f}\n'
            f'-------------------------------')

        return OrderedDict(
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            ap=round(ap, 4),
        )

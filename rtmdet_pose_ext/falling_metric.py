# -*- coding: utf-8 -*-
"""
FallingMetric: Evaluation metric for falling detection (action classification).

Computes binary classification metrics by matching predicted bboxes to GT bboxes
via IoU, then comparing predicted action_scores with GT falling labels.

Outputs: Accuracy, Precision, Recall, F1, AP (Average Precision).
"""
from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import load
from mmengine.logging import MMLogger

from mmdet.registry import METRICS


@METRICS.register_module(force=True)
class FallingMetric(BaseMetric):
    """Binary falling detection metric.

    Matches predicted bboxes to GT via IoU, collects (action_score, gt_falling)
    pairs, and computes Accuracy, Precision, Recall, F1, AP.

    Args:
        ann_file (str): Path to COCO annotation file (to load GT falling labels).
        iou_thr (float): IoU threshold for matching pred to GT. Default: 0.5.
        score_thr (float): Action score threshold for binary decision. Default: 0.5.
        collect_device (str): Device for collecting results. Default: 'cpu'.
        prefix (str): Metric prefix. Default: 'falling'.
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

        # Load GT falling labels from annotation file
        self.ann_file = ann_file
        self._gt_falling = {}  # img_id -> list of falling labels
        self._load_gt_falling()

    def _load_gt_falling(self):
        """Load GT falling labels from COCO annotation file."""
        data = load(self.ann_file)
        # Build image_id -> [falling_label, ...] mapping
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            attributes = ann.get('attributes', {})
            if isinstance(attributes, dict):
                falling = int(attributes.get('falling', 0))
            else:
                falling = 0

            if img_id not in self._gt_falling:
                self._gt_falling[img_id] = []
            self._gt_falling[img_id].append({
                'bbox': ann['bbox'],  # [x, y, w, h] COCO format
                'falling': falling,
            })

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch: collect predicted action_scores and GT falling."""
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            img_id = data_sample['img_id']

            # Skip if no action_scores in prediction
            if 'action_scores' not in pred:
                continue

            pred_bboxes = pred['bboxes'].cpu().numpy()   # (M, 4) xyxy
            pred_scores = pred['scores'].cpu().numpy()    # (M,)
            action_scores = pred['action_scores'].cpu().numpy()  # (M,)

            # Get GT for this image
            gt_list = self._gt_falling.get(img_id, [])
            if len(gt_list) == 0:
                continue

            # Convert GT bboxes from COCO [x,y,w,h] to [x1,y1,x2,y2]
            gt_bboxes = np.array([
                [g['bbox'][0], g['bbox'][1],
                 g['bbox'][0] + g['bbox'][2], g['bbox'][1] + g['bbox'][3]]
                for g in gt_list
            ])
            gt_falling = np.array([g['falling'] for g in gt_list])

            # Match predicted bboxes to GT by IoU
            if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
                continue

            ious = self._compute_iou(pred_bboxes, gt_bboxes)  # (M, G)

            gt_matched = set()
            for pred_idx in np.argsort(-pred_scores):  # highest det score first
                if ious.shape[1] == 0:
                    break
                best_gt = ious[pred_idx].argmax()
                if ious[pred_idx, best_gt] >= self.iou_thr and best_gt not in gt_matched:
                    gt_matched.add(best_gt)
                    self.results.append({
                        'action_score': float(action_scores[pred_idx]),
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

        # Binary predictions at threshold
        pred_labels = (action_scores >= self.score_thr).astype(int)

        # Confusion matrix
        tp = int(((pred_labels == 1) & (gt_labels == 1)).sum())
        fp = int(((pred_labels == 1) & (gt_labels == 0)).sum())
        tn = int(((pred_labels == 0) & (gt_labels == 0)).sum())
        fn = int(((pred_labels == 0) & (gt_labels == 1)).sum())

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Average Precision (AP)
        ap = self._compute_ap(action_scores, gt_labels)

        # Log results
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

    @staticmethod
    def _compute_ap(scores, labels):
        """Compute Average Precision (AP) for binary classification."""
        # Sort by descending score
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]

        num_pos = int(sorted_labels.sum())
        if num_pos == 0:
            return 0.0

        tp_cumsum = np.cumsum(sorted_labels)
        fp_cumsum = np.cumsum(1 - sorted_labels)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / num_pos

        # Prepend (recall=0, precision=1) for AP calculation
        precisions = np.concatenate([[1.0], precisions])
        recalls = np.concatenate([[0.0], recalls])

        # Make precision monotonically decreasing (right to left)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # AP = area under precision-recall curve
        recall_diff = np.diff(recalls)
        ap = float(np.sum(recall_diff * precisions[1:]))
        return ap

    @staticmethod
    def _compute_iou(bboxes1, bboxes2):
        """Compute IoU between two sets of bboxes (xyxy format).

        Args:
            bboxes1: (M, 4) predictions.
            bboxes2: (G, 4) ground truth.

        Returns:
            ious: (M, G) IoU matrix.
        """
        x1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0:1].T)
        y1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1:2].T)
        x2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2:3].T)
        y2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3:4].T)

        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        union = area1[:, None] + area2[None, :] - inter
        return inter / np.maximum(union, 1e-8)

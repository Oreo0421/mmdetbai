import cv2
import numpy as np
import torch
import json

from mmengine.registry import init_default_scope
from mmengine.config import Config
from mmengine.dataset import Compose


# =====================================================
# 0) PATHS (EDIT HERE)
# =====================================================
DET_CONFIG = '/home/tbai/mmdetection/mmdetection/tmdet_human_mobilevit_s.py'
DET_CKPT   = '/home/tbai/mmdetection/mmdetection/work_dirs/tmdet_mobilevit_run1/epoch_5.pth'

POSE_CONFIG = '/home/tbai/mmpose/work_dirs/mobilevit_pose_ep10/timm_mobilevit_pipeline.py'
POSE_CKPT   = '/home/tbai/mmpose/work_dirs/mobilevit_pose_ep10/epoch_40.pth'

IMG_PATH = '/mnt/dst_datasets3/project/SensIR/SensIR/SenIRDatasetProcessed/1A-2 Walking/AAG/undistorted_stitched_png/frame_0040.png'
DEVICE = 'cuda:0'

DET_SCORE_THR = 0.3
DET_CAT_ID = 0
BBOX_EXPAND_SCALE = 2.0

OUT_IMG  = 'det2pose_result.png'
OUT_JSON = 'det2pose_result.json'


# =====================================================
# utils
# =====================================================
def to_numpy(x):
    return x.cpu().numpy() if hasattr(x, 'cpu') else x


def to_3ch(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


def clip_bbox_xyxy_float(b, w, h):
    x1, y1, x2, y2 = map(float, b)
    x1 = max(0.0, min(x1, w - 1.0))
    y1 = max(0.0, min(y1, h - 1.0))
    x2 = max(0.0, min(x2, w - 1.0))
    y2 = max(0.0, min(y2, h - 1.0))
    if x2 <= x1: x2 = min(w - 1.0, x1 + 1.0)
    if y2 <= y1: y2 = min(h - 1.0, y1 + 1.0)
    return [x1, y1, x2, y2]


def expand_bbox_float(b, w, h, scale=2.0):
    x1, y1, x2, y2 = map(float, b)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1), (y2 - y1)
    bw2, bh2 = bw * scale / 2.0, bh * scale / 2.0
    return clip_bbox_xyxy_float(
        [cx - bw2, cy - bh2, cx + bw2, cy + bh2], w, h
    )


def draw_rect(img, bbox_float, color_bgr, thickness=2):
    x1, y1, x2, y2 = bbox_float
    x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), [x1, y1, x2, y2])
    cv2.rectangle(img, (x1i, y1i), (x2i, y2i), color_bgr, thickness)


# =====================================================
# main
# =====================================================
def main():

    img_raw = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(IMG_PATH)
    h, w = img_raw.shape[:2]

    # ---------------- Detector ----------------
    init_default_scope('mmdet')
    from mmdet.apis import init_detector

    det_model = init_detector(DET_CONFIG, DET_CKPT, device=DEVICE)

    det_cfg = Config.fromfile(DET_CONFIG)
    test_pipeline = Compose(det_cfg.test_pipeline)

    data = dict(img_path=IMG_PATH, img_id=0)
    data = test_pipeline(data)

    if isinstance(data['inputs'], torch.Tensor) and data['inputs'].dim() == 3:
        data['inputs'] = data['inputs'].unsqueeze(0)
    if not isinstance(data['data_samples'], (list, tuple)):
        data['data_samples'] = [data['data_samples']]

    with torch.no_grad():
        det_out = det_model.test_step(data)[0]

    pred = det_out.pred_instances
    bboxes = to_numpy(pred.bboxes)
    scores = to_numpy(pred.scores)
    labels = to_numpy(pred.labels)

    keep = (scores >= DET_SCORE_THR) & (labels == DET_CAT_ID)
    bboxes, scores = bboxes[keep], scores[keep]
    if len(bboxes) == 0:
        raise RuntimeError('No human detected.')

    idx = int(np.argmax(scores))
    bbox_score = float(scores[idx])
    bbox_raw = clip_bbox_xyxy_float(bboxes[idx], w, h)
    bbox_pose = expand_bbox_float(bbox_raw, w, h, scale=BBOX_EXPAND_SCALE)

    print('[Detector] raw bbox:', bbox_raw, 'score:', bbox_score)
    print('[PoseInput] bbox:', bbox_pose, f'(scale={BBOX_EXPAND_SCALE})')

    # ---------------- Pose ----------------
    init_default_scope('mmpose')
    from mmpose.apis import init_model as init_pose_model
    from mmpose.apis import inference_topdown
    from mmpose.structures import merge_data_samples

    pose_model = init_pose_model(POSE_CONFIG, POSE_CKPT, device=DEVICE)

    bboxes_np = np.array([bbox_pose], dtype=np.float32)
    pose_results = inference_topdown(pose_model, IMG_PATH, bboxes_np)
    pose_results = merge_data_samples(pose_results)

    kpts = to_numpy(pose_results.pred_instances.keypoints[0])  # (K,2)

    kpt_scores = None
    if hasattr(pose_results.pred_instances, 'keypoint_scores'):
        kpt_scores = to_numpy(pose_results.pred_instances.keypoint_scores[0])

    # ✅ 输出置信度（但不画在图上）
    print('[Pose] keypoints (x,y):')
    print(kpts)
    if kpt_scores is not None:
        print('[Pose] keypoint_scores:')
        print(kpt_scores)
    else:
        print('[Pose] keypoint_scores: NOT FOUND')

    # ---------------- Visualization (ONLY bbox + points) ----------------
    img_vis = to_3ch(img_raw.copy())

    draw_rect(img_vis, bbox_raw, (0, 255, 255), 2)   # Yellow: det bbox
    draw_rect(img_vis, bbox_pose, (0, 0, 255), 2)    # Red: pose bbox

    for (x, y) in kpts:
        xi, yi = int(round(float(x))), int(round(float(y)))
        cv2.circle(img_vis, (xi, yi), 3, (0, 255, 0), -1)

    cv2.imwrite(OUT_IMG, img_vis)
    print('✅ Saved image:', OUT_IMG)

    # ---------------- Save JSON (可选但推荐) ----------------
    out = {
        'img_path': IMG_PATH,
        'det_bbox_xyxy': bbox_raw,
        'det_bbox_score': bbox_score,
        'pose_bbox_xyxy': bbox_pose,
        'keypoints': kpts.tolist(),
        'keypoint_scores': None if kpt_scores is None else kpt_scores.tolist(),
        'bbox_expand_scale': BBOX_EXPAND_SCALE
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print('✅ Saved json:', OUT_JSON)


if __name__ == '__main__':
    main()


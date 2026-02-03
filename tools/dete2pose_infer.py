import cv2
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_model
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples


# =========================
# 1. 路径配置（你只需要改这里）
# =========================
DET_CONFIG = '/home/tbai/mmdetection/mmdetection/rtmdet_human_myAAG.py'
DET_CKPT   = 'work_dirs/rtmdet_human_myAAG/best_coco_bbox_mAP_epoch_30.pth'

POSE_CONFIG = '/home/tbai/mmpose/timm_mobilevit_pipeline.py'
POSE_CKPT   = '/home/tbai/mmpose/work_dirs/mobilevit_pose_ep10/epoch_40.pth'

IMG_PATH = '/mnt/dst_datasets3/project/SensIR/SensIR/SenIRDatasetProcessed/1A-2 Walking/AAG/undistorted_stitched_png/frame_0030.png'
DEVICE = 'cuda:0'


# =========================
# 2. 初始化模型
# =========================
det_model = init_detector(DET_CONFIG, DET_CKPT, device=DEVICE)
pose_model = init_pose_model(POSE_CONFIG, POSE_CKPT, device=DEVICE)


# =========================
# 3. Detector 推理
# =========================
det_result = inference_detector(det_model, IMG_PATH)

pred = det_result.pred_instances
bboxes = pred.bboxes.cpu().numpy()      # (N, 4) xyxy
scores = pred.scores.cpu().numpy()
labels = pred.labels.cpu().numpy()

# 👉 单类 human detector：label == 0
keep = (scores > 0.3) & (labels == 0)
bboxes = bboxes[keep]
scores = scores[keep]

if len(bboxes) == 0:
    raise RuntimeError('No human detected')

# 取 score 最高的一个人
idx = np.argmax(scores)
bbox = bboxes[idx].tolist()
bbox_score = float(scores[idx])

print('[Detector] bbox:', bbox, 'score:', bbox_score)


# =========================
# 4. 用 detector bbox 喂给 MMPose
# =========================
person_results = [{
    'bbox': bbox,
    'bbox_score': bbox_score
}]

pose_results = inference_topdown(
    pose_model,
    IMG_PATH,
    person_results
)

pose_results = merge_data_samples(pose_results)


# =========================
# 5. 可视化（bbox + keypoints）
# =========================
img = cv2.imread(IMG_PATH)
if img.ndim == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

pose_model.visualizer.set_image(img)
pose_model.visualizer.add_datasample(
    'result',
    img,
    pose_results,
    draw_gt=False,
    draw_bbox=True,
    draw_heatmap=False,
    kpt_thr=0.3
)

vis_img = pose_model.visualizer.get_image()
cv2.imwrite('det2pose_result.png', vis_img)

print('✅ Saved det2pose_result.png')


# =========================
# 6. 打印关键点（数值验证）
# =========================
kpts = pose_results.pred_instances.keypoints[0].cpu().numpy()
print('[Pose] keypoints:')
print(kpts)


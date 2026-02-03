import os
import json
import cv2
import numpy as np

from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector


# =========================
# 0) 你的路径（已按你给的改好）
# =========================
DET_CONFIG = "/home/tbai/mmdetection/mmdetection/work_dirs/rtmdet_mobilevit_s_sensir_test/rtmdet_mobilevit_s_sensir.py"
DET_CKPT   = "/home/tbai/mmdetection/mmdetection/work_dirs/rtmdet_mobilevit_s_sensir_test/best_coco_bbox_mAP_epoch_30.pth"

INPUT_PATH = "/home/tbai/Desktop/SenIRDatasetProcessed/1A-2 Walking/AAG/undistorted_stitched_no_overlap_png/"
OUT_DIR    = "det_infer_out"
OUT_JSON   = "det_results.json"

DEVICE    = "cuda:1"   # 你现在用的
SCORE_THR = 0.6        # 你设的阈值


# =========================
# 1) 初始化
# =========================
os.makedirs(OUT_DIR, exist_ok=True)
init_default_scope("mmdet")

model = init_detector(
    DET_CONFIG,
    DET_CKPT,
    device=DEVICE
)

# =========================
# 2) 处理输入（单张 or 文件夹）
# =========================
if os.path.isfile(INPUT_PATH):
    img_list = [INPUT_PATH]
else:
    img_list = [
        os.path.join(INPUT_PATH, f)
        for f in os.listdir(INPUT_PATH)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
img_list = sorted(img_list)[:50]  
all_results = []

# =========================
# 3) 推理
# =========================
for img_path in img_list:
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    result = inference_detector(model, img)
    pred = result.pred_instances

    bboxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()

    vis_img = img.copy()
    detections = []

    for bbox, score, label in zip(bboxes, scores, labels):
        if score < SCORE_THR:
            continue
        if label != 0:  # person 类
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # ✅ 只画框，不写任何文字
        cv2.rectangle(
            vis_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        detections.append({
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(score)
        })

    # 保存可视化结果
    out_img_path = os.path.join(OUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_img_path, vis_img)

    all_results.append({
        "image": os.path.basename(img_path),
        "width": w,
        "height": h,
        "detections": detections
    })

    print(f"Saved image: {out_img_path}")

# =========================
# 4) 保存 JSON
# =========================
with open(OUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Saved JSON results to {OUT_JSON}")
print("Inference finished.")


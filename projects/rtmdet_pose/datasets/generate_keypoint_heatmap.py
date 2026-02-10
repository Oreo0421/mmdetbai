import math
import numpy as np
import cv2
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

def _draw_gaussian(heatmap, center, sigma):
    """在单个 heatmap 上画 2D gaussian（最小实现）"""
    x, y = center
    h, w = heatmap.shape

    tmp_size = int(3 * sigma)
    x1, y1 = int(x - tmp_size), int(y - tmp_size)
    x2, y2 = int(x + tmp_size + 1), int(y + tmp_size + 1)

    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return heatmap  # 完全在外面

    # gaussian patch
    size = 2 * tmp_size + 1
    xs = np.arange(0, size, 1, np.float32)
    ys = xs[:, None]
    x0 = y0 = tmp_size
    g = np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma * sigma))

    # overlap ranges
    g_x1 = max(0, -x1)
    g_y1 = max(0, -y1)
    g_x2 = min(size, w - x1)
    g_y2 = min(size, h - y1)

    h_x1 = max(0, x1)
    h_y1 = max(0, y1)
    h_x2 = min(w, x2)
    h_y2 = min(h, y2)

    heatmap[h_y1:h_y2, h_x1:h_x2] = np.maximum(
        heatmap[h_y1:h_y2, h_x1:h_x2],
        g[g_y1:g_y2, g_x1:g_x2]
    )
    return heatmap


@TRANSFORMS.register_module()
class GenerateKeypointHeatmap:
    """把 gt_keypoints 生成 heatmap targets.
    - 输入：results['gt_keypoints'] [N,K,2], results['gt_keypoints_visible'] [N,K]
    - 输出：results['gt_kpt_heatmaps'] [K,Hh,Wh], results['gt_kpt_weights'] [K]
    默认对多人：对每个 keypoint channel 做 max 合并（one-stage 常见做法）
    """

    def __init__(self, num_keypoints, heatmap_size=(48, 48), sigma=2.0, use_max=True):
        self.num_keypoints = int(num_keypoints)
        self.heatmap_size = tuple(heatmap_size)  # (W,H) or (H,W)? 这里统一 (W,H) 也行，只要你一致
        self.sigma = float(sigma)
        self.use_max = bool(use_max)

    def transform(self, results: dict) -> dict:
        if 'gt_keypoints' not in results:
            return results

        gt_kpts = results['gt_keypoints']              # [N,K,2]
        gt_vis  = results.get('gt_keypoints_visible')  # [N,K]
        if gt_vis is None:
            gt_vis = np.ones((gt_kpts.shape[0], gt_kpts.shape[1]), dtype=np.float32)

        img_h, img_w = results['img_shape'][:2]
        hm_w, hm_h = self.heatmap_size

        # scale from image -> heatmap
        sx = hm_w / float(img_w)
        sy = hm_h / float(img_h)

        heatmaps = np.zeros((self.num_keypoints, hm_h, hm_w), dtype=np.float32)
        weights  = np.zeros((self.num_keypoints,), dtype=np.float32)

        # 合并多人：每个关节通道取 max
        for n in range(gt_kpts.shape[0]):
            for k in range(min(self.num_keypoints, gt_kpts.shape[1])):
                if gt_vis[n, k] <= 0:
                    continue
                x, y = gt_kpts[n, k]
                x_hm = x * sx
                y_hm = y * sy
                if x_hm < 0 or y_hm < 0 or x_hm >= hm_w or y_hm >= hm_h:
                    continue
                weights[k] = 1.0
                _draw_gaussian(heatmaps[k], (x_hm, y_hm), self.sigma)

        results['gt_kpt_heatmaps'] = heatmaps     # [K,Hh,Wh]
        results['gt_kpt_weights']  = weights      # [K]
        results['kpt_heatmap_size'] = (hm_h, hm_w)
        return results


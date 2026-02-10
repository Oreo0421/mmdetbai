from typing import Dict, Tuple
import numpy as np
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module(force=True)
class GeneratePoseHeatmap(BaseTransform):
    """为关键点生成高斯热图。
    
    关键修复：在生成热图之前，将关键点坐标从原始图像空间
    变换到当前（经过 Resize / RandomFlip 后的）图像空间。
    """

    def __init__(self, heatmap_size: Tuple[int, int] = (48, 48), sigma: float = 2.0,
                 num_keypoints: int = 7):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.num_keypoints = num_keypoints

    def transform(self, results: Dict) -> Dict:
        img_shape = results.get('img_shape')           # 当前（resize后）图像尺寸 (h, w)
        h, w = img_shape[:2]
        hm_h, hm_w = self.heatmap_size

        # ---- 获取 scale_factor 和 flip 状态 ----
        # mmdet Resize 存储: scale_factor = (w_scale, h_scale)
        scale_factor = results.get('scale_factor', (1.0, 1.0))
        if isinstance(scale_factor, np.ndarray):
            scale_factor = tuple(scale_factor.tolist())
        w_scale, h_scale = float(scale_factor[0]), float(scale_factor[1])

        flipped = results.get('flip', False)
        flip_direction = results.get('flip_direction', 'horizontal')

        K = self.num_keypoints
        heatmap = np.zeros((K, hm_h, hm_w), dtype=np.float32)
        keypoints_tensor = np.zeros((K, 3), dtype=np.float32)

        if 'instances' not in results or len(results['instances']) == 0:
            results['gt_keypoints'] = keypoints_tensor
            results['gt_keypoints_heatmap'] = heatmap
            return results

        # 取第一个实例的关键点（单目标场景）
        inst = results['instances'][0]
        if 'keypoints' not in inst:
            results['gt_keypoints'] = keypoints_tensor
            results['gt_keypoints_heatmap'] = heatmap
            return results

        kpts = inst['keypoints']

        for i in range(K):
            idx = i * 3
            if idx + 2 >= len(kpts):
                break
            x_orig, y_orig, v = float(kpts[idx]), float(kpts[idx + 1]), float(kpts[idx + 2])

            if v <= 0:
                keypoints_tensor[i] = [0, 0, 0]
                continue

            # ---- 1) 从原始坐标映射到 resize 后的坐标 ----
            x = x_orig * w_scale
            y = y_orig * h_scale

            # ---- 2) 如果发生了翻转，镜像 x 坐标 ----
            if flipped and flip_direction == 'horizontal':
                x = w - 1.0 - x

            # 保存变换后的关键点坐标（在 resize 后的图像空间中）
            keypoints_tensor[i] = [x, y, v]

            # ---- 3) 映射到热图空间并生成高斯 ----
            hm_x = x * hm_w / w
            hm_y = y * hm_h / h

            hm_x_int = int(round(hm_x))
            hm_y_int = int(round(hm_y))

            if 0 <= hm_x_int < hm_w and 0 <= hm_y_int < hm_h:
                heatmap[i] = self._generate_gaussian(hm_h, hm_w, hm_x_int, hm_y_int)

        results['gt_keypoints'] = keypoints_tensor
        results['gt_keypoints_heatmap'] = heatmap
        return results

    def _generate_gaussian(self, h: int, w: int, cx: int, cy: int) -> np.ndarray:
        x = np.arange(0, w, 1, np.float32)
        y = np.arange(0, h, 1, np.float32)[:, np.newaxis]
        return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * self.sigma ** 2))


@TRANSFORMS.register_module(force=True)
class CopyImgIdToId(BaseTransform):
    """Copy results['img_id'] to results['id'] for mmpose CocoMetric."""

    def __init__(self, src_key: str = 'img_id', dst_key: str = 'id'):
        super().__init__()
        self.src_key = src_key
        self.dst_key = dst_key

    def transform(self, results: Dict) -> Dict:
        if self.src_key in results and self.dst_key not in results:
            results[self.dst_key] = results[self.src_key]
        return results

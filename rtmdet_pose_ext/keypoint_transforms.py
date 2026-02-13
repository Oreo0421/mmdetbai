from typing import Dict, Tuple
import cv2
import numpy as np
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module(force=True)
class RandomRotateWithPose(BaseTransform):
    """Random rotation with canvas expansion for elongated images.

    Place AFTER LoadAnnotations and BEFORE Resize in the pipeline.

    Key design: expands the canvas to fit ALL rotated content (no cropping),
    then Resize(384,384) normalizes the size. This is critical for elongated
    images like 40x114 IR thermal frames where same-canvas rotation would
    crop significant content.

    Data flow:
      1. Compute expanded canvas size to fit rotated image
      2. Rotate image with cv2.warpAffine onto expanded canvas
      3. Rotate gt_bboxes (4-corner -> new AABB, clipped to expanded canvas)
      4. Rotate keypoints in instances (now in expanded canvas coords)
      5. Resize then maps from expanded canvas -> 384x384, setting scale_factor
      6. Head applies scale_factor to keypoints as usual -> correct!

    Args:
        angle_range (tuple[float, float]): Min/max rotation angle in degrees.
        prob (float): Probability of applying rotation.
        border_value (int): Pixel fill value for padded areas.
    """

    def __init__(self, angle_range=(-20, 20), prob=0.5, border_value=0):
        super().__init__()
        self.angle_range = angle_range
        self.prob = prob
        self.border_value = border_value

    def transform(self, results: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return results

        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        if abs(angle) < 0.5:
            return results

        img = results['img']
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # --- Compute expanded canvas to fit all rotated content ---
        cos_a = abs(np.cos(np.radians(angle)))
        sin_a = abs(np.sin(np.radians(angle)))
        new_w = int(np.ceil(w * cos_a + h * sin_a))
        new_h = int(np.ceil(w * sin_a + h * cos_a))

        # Rotation matrix around original center
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        # Shift to center of expanded canvas
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0

        # --- 1. Rotate image onto expanded canvas ---
        border = (self.border_value,) * 3 if img.ndim == 3 else (self.border_value,)
        results['img'] = cv2.warpAffine(
            img, M, (new_w, new_h), borderValue=border)
        results['img_shape'] = (new_h, new_w)

        # --- 2. Rotate gt_bboxes ---
        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            if isinstance(bboxes, np.ndarray) and len(bboxes) > 0:
                results['gt_bboxes'] = self._rotate_bboxes(
                    bboxes, M, new_w, new_h)

        # --- 3. Rotate keypoints in instances ---
        if 'instances' in results:
            for inst in results['instances']:
                if 'keypoints' not in inst:
                    continue
                kpts = np.array(inst['keypoints'], dtype=np.float32)
                if kpts.ndim == 1:
                    kpts = kpts.reshape(-1, 3)

                for k in range(len(kpts)):
                    x, y, v = kpts[k]
                    if v <= 0:
                        continue
                    # Apply rotation (including canvas shift)
                    new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
                    new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
                    # Should always be in bounds with expanded canvas,
                    # but clip for safety
                    if 0 <= new_x < new_w and 0 <= new_y < new_h:
                        kpts[k, 0] = new_x
                        kpts[k, 1] = new_y
                    else:
                        kpts[k] = [0, 0, 0]

                inst['keypoints'] = kpts.flatten().tolist()

        return results

    def _rotate_bboxes(self, bboxes, M, img_w, img_h):
        """Rotate bboxes by transforming 4 corners -> new AABB, clipped."""
        N = len(bboxes)
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        # 4 corners per bbox: (N, 4, 2)
        corners = np.stack([
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ], axis=1)

        # Homogeneous coords (N, 4, 3)
        ones = np.ones((N, 4, 1), dtype=np.float32)
        corners_h = np.concatenate([corners, ones], axis=2)

        # Apply rotation: M (2,3) @ corners_h^T -> (N, 4, 2)
        rotated = np.einsum('ij,nkj->nki', M, corners_h)

        # New axis-aligned bounding box, clipped to expanded canvas
        new_x1 = np.clip(rotated[:, :, 0].min(axis=1), 0, img_w - 1)
        new_y1 = np.clip(rotated[:, :, 1].min(axis=1), 0, img_h - 1)
        new_x2 = np.clip(rotated[:, :, 0].max(axis=1), 0, img_w - 1)
        new_y2 = np.clip(rotated[:, :, 1].max(axis=1), 0, img_h - 1)

        return np.stack([new_x1, new_y1, new_x2, new_y2], axis=1).astype(np.float32)


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

        # ---- 4) 翻转时交换左右对称关键点 ----
        if flipped and flip_direction == 'horizontal':
            for left, right in [(2, 3), (5, 6)]:
                if left < K and right < K:
                    keypoints_tensor[[left, right]] = keypoints_tensor[[right, left]].copy()
                    heatmap[[left, right]] = heatmap[[right, left]].copy()

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

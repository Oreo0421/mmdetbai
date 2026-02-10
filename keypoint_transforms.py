from typing import Dict, Tuple
import numpy as np
import torch
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class GeneratePoseHeatmap(BaseTransform):
    """为 7 个关键点生成高斯热图"""
    
    def __init__(self, heatmap_size: Tuple[int, int] = (48, 48), sigma: float = 2.0):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.num_keypoints = 7
    
    def transform(self, results: Dict) -> Dict:
        """生成关键点热图"""
        
        img_shape = results.get('img_shape', (results['height'], results['width']))
        h, w = img_shape[:2]
        hm_h, hm_w = self.heatmap_size
        
        # 初始化空热图
        heatmap = np.zeros((self.num_keypoints, hm_h, hm_w), dtype=np.float32)
        keypoints_tensor = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        
        # 从 instances 提取 keypoints
        if 'instances' in results and len(results['instances']) > 0:
            inst = results['instances'][0]  # 取第一个实例
            if 'keypoints' in inst:
                kpts = inst['keypoints']
                # 解析 [x1, y1, v1, x2, y2, v2, ...] 格式
                for i in range(self.num_keypoints):
                    x = kpts[i * 3]
                    y = kpts[i * 3 + 1]
                    v = kpts[i * 3 + 2]
                    
                    keypoints_tensor[i] = [x, y, v]
                    
                    if v > 0:  # 可见点
                        # 转换到热图坐标
                        hm_x = int(x * hm_w / w)
                        hm_y = int(y * hm_h / h)
                        
                        if 0 <= hm_x < hm_w and 0 <= hm_y < hm_h:
                            heatmap[i] = self._generate_gaussian(hm_h, hm_w, hm_x, hm_y)
        
        # 关键：存储到 results 中，让 PackDetInputs 能够访问
        results['gt_keypoints'] = keypoints_tensor  # (7, 3)
        results['gt_keypoints_heatmap'] = heatmap   # (7, 48, 48)
        
        # 同时更新 instances（可选，保持一致性）
        if 'instances' in results:
            for inst in results['instances']:
                inst['keypoints_heatmap'] = heatmap
        
        return results
    
    def _generate_gaussian(self, h: int, w: int, cx: int, cy: int) -> np.ndarray:
        """生成以 (cx, cy) 为中心的高斯热图"""
        x = np.arange(0, w, 1, np.float32)
        y = np.arange(0, h, 1, np.float32)
        y = y[:, np.newaxis]
        
        gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * self.sigma ** 2))
        return gaussian

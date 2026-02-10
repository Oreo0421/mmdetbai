# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple
import numpy as np
import torch

from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


class GenerateKeypointHeatmap(BaseTransform):
    """Generate Gaussian heatmaps for keypoint annotations.
    
    This transform converts keypoint coordinates (x, y, visibility) into
    Gaussian heatmaps for training pose estimation models.
    
    Args:
        heatmap_size (tuple): Output heatmap size (H, W).
            Default: (48, 48) - stride=4 for 192x192 input
        sigma (float): Gaussian kernel standard deviation.
            Larger sigma = wider heatmap peaks (easier to learn)
        use_different_sigma (bool): Use different sigma for each keypoint
    """
    
    def __init__(
        self,
        heatmap_size: Tuple[int, int] = (48, 48),
        sigma: float = 2.0,
        use_different_sigma: bool = False,
    ):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.use_different_sigma = use_different_sigma
    
    def transform(self, results: Dict) -> Dict:
        """Transform function to generate heatmaps.
        
        Args:
            results (dict): Result dict from loading pipeline
        
        Returns:
            dict: Updated results with 'keypoints_heatmap' and 'keypoint_weights'
        """
        # 首先从 ann_info 中提取关键点 (COCO 格式)
        if 'ann_info' in results:
            ann_info = results['ann_info']
            if 'keypoints' in ann_info and len(ann_info['keypoints']) > 0:
                # COCO 格式: [x1, y1, v1, x2, y2, v2, ...]
                keypoints_flat = ann_info['keypoints']
                keypoints = np.array(keypoints_flat).reshape(-1, 3)
                
                # 添加到 gt_instances
                if 'gt_instances' in results:
                    results['gt_instances'].keypoints = keypoints
        
        # 现在处理 gt_instances
        if 'gt_instances' not in results:
            return results
        
        gt_instances = results['gt_instances']
        
        # Check if keypoints exist
        if not hasattr(gt_instances, 'keypoints') or len(gt_instances.keypoints) == 0:
            return results
        
        # Get image shape and heatmap size
        img_h, img_w = results['img_shape'][:2]
        heatmap_h, heatmap_w = self.heatmap_size
        
        # Get all keypoints (assuming single person detection)
        # Shape: (num_instances, num_keypoints, 3) where 3 = [x, y, visibility]
        keypoints = np.array(gt_instances.keypoints)
        
        if keypoints.ndim == 2:  # Single instance
            keypoints = keypoints[np.newaxis, ...]
        
        num_instances, num_keypoints, _ = keypoints.shape
        
        # Initialize heatmaps and weights for all instances
        all_heatmaps = []
        all_weights = []
        
        for inst_idx in range(num_instances):
            kpts = keypoints[inst_idx]  # (num_keypoints, 3)
            
            # Generate heatmaps for this instance
            heatmaps = np.zeros((num_keypoints, heatmap_h, heatmap_w), dtype=np.float32)
            weights = np.zeros(num_keypoints, dtype=np.float32)
            
            for kpt_idx in range(num_keypoints):
                x, y, v = kpts[kpt_idx]
                
                # Visibility: 0=not labeled, 1=labeled but occluded, 2=labeled and visible
                if v == 0:
                    # Not labeled - zero weight
                    weights[kpt_idx] = 0.0
                    continue
                elif v == 1:
                    # Occluded - reduced weight
                    weights[kpt_idx] = 0.5
                else:  # v == 2
                    # Visible - full weight
                    weights[kpt_idx] = 1.0
                
                # Scale keypoint to heatmap resolution
                x_hm = x * heatmap_w / img_w
                y_hm = y * heatmap_h / img_h
                
                # Generate Gaussian heatmap
                heatmaps[kpt_idx] = self._generate_gaussian_heatmap(
                    heatmap_w, heatmap_h, x_hm, y_hm, self.sigma
                )
            
            all_heatmaps.append(heatmaps)
            all_weights.append(weights)
        
        # Stack heatmaps if multiple instances (for now, assume single instance)
        # In your case, you have 1 human per image
        heatmaps = all_heatmaps[0] if len(all_heatmaps) == 1 else np.stack(all_heatmaps).sum(axis=0)
        weights = all_weights[0] if len(all_weights) == 1 else np.stack(all_weights).max(axis=0)
        
        # Add to gt_instances
        gt_instances.keypoints_heatmap = torch.from_numpy(heatmaps).float()
        gt_instances.keypoint_weights = torch.from_numpy(weights).float()
        
        return results
    
    def _generate_gaussian_heatmap(
        self,
        width: int,
        height: int,
        center_x: float,
        center_y: float,
        sigma: float,
    ) -> np.ndarray:
        """Generate a 2D Gaussian heatmap.
        
        Args:
            width (int): Heatmap width
            height (int): Heatmap height
            center_x (float): Gaussian center x coordinate
            center_y (float): Gaussian center y coordinate
            sigma (float): Gaussian standard deviation
        
        Returns:
            np.ndarray: Gaussian heatmap of shape (height, width)
        """
        # Create coordinate grids
        x = np.arange(0, width, 1, dtype=np.float32)
        y = np.arange(0, height, 1, dtype=np.float32)
        y = y[:, np.newaxis]
        
        # Calculate 2D Gaussian
        # G(x, y) = exp(-((x - mu_x)^2 + (y - mu_y)^2) / (2 * sigma^2))
        heatmap = np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2)
        )
        
        return heatmap
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(heatmap_size={self.heatmap_size}, '
        repr_str += f'sigma={self.sigma})'
        return repr_str


class LoadKeypointsFromCOCO(BaseTransform):
    """Load keypoint annotations from COCO format.
    
    This transform extracts keypoint data from COCO annotations
    and adds them to gt_instances.
    
    COCO keypoint format:
        [x1, y1, v1, x2, y2, v2, ..., xn, yn, vn]
    where:
        - (xi, yi): keypoint coordinates
        - vi: visibility flag (0=not labeled, 1=occluded, 2=visible)
    """
    
    def __init__(self, num_keypoints: int = 7):
        super().__init__()
        self.num_keypoints = num_keypoints
    
    def transform(self, results: Dict) -> Dict:
        """Load keypoints from results dict.
        
        Args:
            results (dict): Result dict from COCO loading
        
        Returns:
            dict: Updated results with keypoints in gt_instances
        """
        if 'ann_info' not in results:
            return results
        
        ann_info = results['ann_info']
        
        # Check if keypoints exist in annotation
        if 'keypoints' not in ann_info or len(ann_info['keypoints']) == 0:
            return results
        
        # Parse keypoints from COCO format
        keypoints_flat = ann_info['keypoints']
        
        # Reshape to (num_keypoints, 3)
        keypoints = np.array(keypoints_flat).reshape(-1, 3)
        
        # Add to gt_instances
        if 'gt_instances' in results:
            results['gt_instances'].keypoints = keypoints
        
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_keypoints={self.num_keypoints})'

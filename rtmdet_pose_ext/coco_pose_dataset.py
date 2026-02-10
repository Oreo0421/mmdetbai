from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module(force=True)
class CocoPoseDataset(CocoDataset):
    """COCO dataset with keypoint support"""
    
    METAINFO = {
        'classes': ('person',),
        'palette': [(220, 20, 60)],
    }
    
    def parse_data_info(self, raw_data_info: dict) -> dict:
        """重写以添加 keypoints 到 instances"""
        data_info = super().parse_data_info(raw_data_info)
        
        # raw_data_info 包含原始 COCO 标注
        raw_ann_info = raw_data_info.get('raw_ann_info', [])
        if isinstance(raw_ann_info, dict):
            raw_ann_info = [raw_ann_info]
        
        # 添加 keypoints 到 instances
        if 'instances' in data_info:
            for i, inst in enumerate(data_info['instances']):
                if i < len(raw_ann_info):
                    ann = raw_ann_info[i]
                    if 'keypoints' in ann:
                        inst['keypoints'] = ann['keypoints']
        
        return data_info

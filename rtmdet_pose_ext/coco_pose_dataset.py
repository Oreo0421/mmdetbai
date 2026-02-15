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
                    # 提取动作类别和 falling 属性
                    attributes = ann.get('attributes', {})
                    if isinstance(attributes, dict):
                        # 优先使用 action_class (0-9)
                        if 'action_class' in attributes:
                            action_class = int(attributes['action_class'])
                            inst['action_class'] = action_class
                            inst['falling'] = int(action_class >= 6)
                        else:
                            # 向后兼容：回退到 falling 属性
                            inst['falling'] = int(attributes.get('falling', 0))
                            inst['action_class'] = -1  # unknown
                    else:
                        inst['falling'] = 0
                        inst['action_class'] = -1

        return data_info

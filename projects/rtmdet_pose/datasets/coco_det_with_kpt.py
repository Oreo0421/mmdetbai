import numpy as np
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class CocoDetWithKeypointDataset(CocoDataset):
    """在 CocoDataset 的基础上，把 ann['keypoints'] 保留到 instances 里。
    兼容单人/多人：每个 instance 都带 keypoints。
    """

    def parse_data_info(self, raw_data_info):
        data_info = super().parse_data_info(raw_data_info)

        # CocoDataset 的 data_info 里通常有 instances(list)，每个 instance 来自 annotation
        instances = data_info.get('instances', [])
        ann_list = raw_data_info.get('ann_info', raw_data_info.get('anns', None))

        # 在不同版本里 raw_data_info 结构可能不完全一致：
        # 最稳做法：从 raw_data_info['raw_ann_info'] 或 raw_data_info['ann_info'] 找
        raw_ann_info = raw_data_info.get('raw_ann_info', None)
        if raw_ann_info is None:
            raw_ann_info = raw_data_info.get('ann_info', None)

        # 如果拿不到 raw ann，就直接返回（只跑 det）
        if raw_ann_info is None:
            return data_info

        # raw_ann_info 是 list，每个 ann 对应一个实例；把 keypoints 塞进 instances
        # 注意：super() 里可能过滤掉 ignore/crowd，一般 instances 和 raw_ann_info 数量可能不一致
        # 这里尽量用 bbox 匹配（最小实现先按顺序对齐；你的数据若无 crowd/ignore 基本可行）
        min_len = min(len(instances), len(raw_ann_info))
        for i in range(min_len):
            ann = raw_ann_info[i]
            if 'keypoints' in ann and ann['keypoints'] is not None:
                kpt = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)  # [K,3]
                instances[i]['keypoints'] = kpt  # 保留给后续 LoadAnnotationsWithKeypoints
        data_info['instances'] = instances
        return data_info


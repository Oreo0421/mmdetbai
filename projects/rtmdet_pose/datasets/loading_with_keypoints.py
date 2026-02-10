import numpy as np
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import LoadAnnotations


@TRANSFORMS.register_module()
class LoadAnnotationsWithKeypoints(LoadAnnotations):
    """在 LoadAnnotations 基础上，额外读取 instances[i]['keypoints'] -> gt_keypoints/gt_keypoints_visible"""

    def transform(self, results: dict) -> dict:
        results = super().transform(results)

        instances = results.get('instances', None)
        if instances is None or len(instances) == 0:
            return results

        kpts = []
        vis  = []

        for inst in instances:
            if 'keypoints' not in inst:
                continue
            kpt = np.asarray(inst['keypoints'], dtype=np.float32)  # [K,3]
            kpts.append(kpt[:, :2])                                # xy
            # COCO: v=0(未标注),1(标注但不可见),2(可见)
            v = kpt[:, 2]
            vis.append((v > 0).astype(np.float32))

        if len(kpts) == 0:
            return results

        results['gt_keypoints'] = np.stack(kpts, axis=0)           # [N,K,2]
        results['gt_keypoints_visible'] = np.stack(vis, axis=0)    # [N,K]
        return results


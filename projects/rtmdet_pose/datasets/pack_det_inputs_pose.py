from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs


@TRANSFORMS.register_module()
class PackDetInputsWithPose(PackDetInputs):
    """在 PackDetInputs 基础上，把 heatmap/weights 也塞进 data_sample 里。"""

    def transform(self, results: dict) -> dict:
        packed = super().transform(results)
        data_sample = packed['data_samples']

        # 存到 data_sample 里，后面你的 pose_head.loss() 自己去取
        if 'gt_kpt_heatmaps' in results:
            data_sample.set_field(results['gt_kpt_heatmaps'], 'gt_kpt_heatmaps')
        if 'gt_kpt_weights' in results:
            data_sample.set_field(results['gt_kpt_weights'], 'gt_kpt_weights')
        if 'kpt_heatmap_size' in results:
            data_sample.set_field(results['kpt_heatmap_size'], 'kpt_heatmap_size')

        return packed


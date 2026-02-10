import numpy as np
import torch

from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs


@TRANSFORMS.register_module(force=True)
class PackDetInputsWithPose(PackDetInputs):
    """PackDetInputs + keep pose GT fields.

    What we do:
      1) Call parent PackDetInputs to pack `inputs` and `data_samples`
      2) If `gt_keypoints` / `gt_keypoints_heatmap` exist in `results`,
         write them onto `data_samples` so the loss can use them.
      3) Mirror `gt_keypoints` into `data_samples.gt_instances.keypoints`
         with length aligned to number of boxes.
    """

    def transform(self, results: dict) -> dict:
        packed = super().transform(results)

        data_sample = packed.get('data_samples', None)
        if data_sample is None:
            return packed

        # ---- restore pose gt fields onto data_sample (from results dict) ----
        if 'gt_keypoints' in results:
            kpts = results['gt_keypoints']
            if isinstance(kpts, np.ndarray):
                kpts = torch.from_numpy(kpts)
            if torch.is_tensor(kpts):
                data_sample.gt_keypoints = kpts.float()

        if 'gt_keypoints_heatmap' in results:
            hm = results['gt_keypoints_heatmap']
            if isinstance(hm, np.ndarray):
                hm = torch.from_numpy(hm)
            if torch.is_tensor(hm):
                data_sample.gt_keypoints_heatmap = hm.float()

        # ---- mirror keypoints into gt_instances for downstream compatibility ----
        if hasattr(data_sample, 'gt_instances') and data_sample.gt_instances is not None:
            gt_instances = data_sample.gt_instances
            N = len(gt_instances)

            if hasattr(data_sample, 'gt_keypoints'):
                kpts = data_sample.gt_keypoints
                # expected shapes:
                #   (K, D) -> single instance
                #   (N, K, D) -> multi instance
                if kpts.dim() == 2:
                    kpts = kpts.unsqueeze(0)  # (1, K, D)

                if N == 0:
                    # keep empty to be consistent
                    K = kpts.size(1)
                    D = kpts.size(2)
                    gt_instances.keypoints = kpts.new_zeros((0, K, D))
                else:
                    # align instance count
                    if kpts.size(0) != N:
                        if kpts.size(0) > N:
                            kpts = kpts[:N]
                        else:
                            kpts = kpts[:1].repeat(N, 1, 1)
                    gt_instances.keypoints = kpts

        return packed

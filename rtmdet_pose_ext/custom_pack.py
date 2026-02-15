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
        else:
            kpts = None
            if 'instances' in results and len(results['instances']) > 0:
                # collect keypoints from instances if available
                all_kpts = []
                for inst in results['instances']:
                    if 'keypoints' not in inst:
                        continue
                    kp = inst['keypoints']
                    kp = np.asarray(kp, dtype=np.float32)
                    if kp.ndim == 1:
                        if kp.size % 3 != 0:
                            continue
                        kp = kp.reshape(-1, 3)
                    all_kpts.append(kp)
                if len(all_kpts) > 0:
                    kpts = np.stack(all_kpts, axis=0)

        if kpts is not None:
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

            # ---- pack action_class and falling labels into gt_instances ----
            if 'instances' in results and len(results['instances']) > 0:
                action_labels = []
                falling_labels = []
                for inst in results['instances']:
                    action_labels.append(int(inst.get('action_class', -1)))
                    falling_labels.append(int(inst.get('falling', 0)))

                action_t = torch.tensor(action_labels, dtype=torch.long)
                falling_t = torch.tensor(falling_labels, dtype=torch.float32)

                if len(action_t) > N:
                    action_t = action_t[:N]
                    falling_t = falling_t[:N]
                elif len(action_t) < N:
                    pad = N - len(action_t)
                    action_t = torch.cat([
                        action_t,
                        torch.full((pad,), -1, dtype=torch.long)
                    ])
                    falling_t = torch.cat([
                        falling_t,
                        torch.zeros(pad, dtype=torch.float32)
                    ])
                gt_instances.action_class = action_t
                gt_instances.falling = falling_t

                # V9: pack temporal kpt_sequence into gt_instances
                kpt_sequences = []
                has_seq = False
                for inst in results['instances']:
                    seq = inst.get('kpt_sequence', None)
                    if seq is not None and len(seq) > 0:
                        kpt_sequences.append(seq)
                        has_seq = True
                    else:
                        kpt_sequences.append(None)

                if has_seq:
                    # Dynamic feature dim: 21 (K*3) or 35 (K*5 with velocity)
                    first_seq = next(
                        s for s in kpt_sequences if s is not None)
                    KD = len(first_seq[0])
                    max_T = max(
                        len(s) for s in kpt_sequences if s is not None)
                    packed_seqs = torch.zeros(
                        N, max_T, KD, dtype=torch.float32)
                    for i, seq in enumerate(kpt_sequences):
                        if seq is not None and i < N:
                            T = len(seq)
                            for t in range(T):
                                packed_seqs[i, t] = torch.tensor(
                                    seq[t], dtype=torch.float32)
                    gt_instances.kpt_sequence = packed_seqs

        return packed

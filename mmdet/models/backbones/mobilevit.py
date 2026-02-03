
# mmdet/models/backbones/mobilevit_cvnets.py

import argparse
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS

from cvnets.models.classification.mobilevit import MobileViT  # Apple CVNets MobileViT


def _make_minimal_cvnets_opts(
    mode: str = "small",
    n_classes: int = 1000,
    global_pool: str = "mean",
    classifier_dropout: float = 0.0,
) -> argparse.Namespace:
    """
    CVNets uses argparse.Namespace where attribute names may contain dots,
    e.g. getattr(opts, "model.classification.mit.mode", "small").
    That's valid because setattr/getattr can use keys like 'a.b' in __dict__.
    """
    opts = argparse.Namespace()

    # ---- MobileViT-specific ----
    setattr(opts, "model.classification.mit.mode", mode)  # xx_small / x_small / small :contentReference[oaicite:2]{index=2}
    setattr(opts, "model.classification.n_classes", n_classes)
    setattr(opts, "model.layer.global_pool", global_pool)
    setattr(opts, "model.classification.classifier_dropout", classifier_dropout)

    # ---- Safe defaults (commonly referenced by CVNets layers) ----
    # If some CVNets layer complains about missing opts, add them here.
    setattr(opts, "model.classification.gradient_checkpointing", False)
    setattr(opts, "model.classification.freeze_batch_norm", False)

    # activation defaults
    setattr(opts, "model.activation.name", "swish")   # CVNets often uses swish/silu; adjust if needed
    setattr(opts, "model.activation.inplace", True)
    setattr(opts, "model.activation.neg_slope", 0.1)

    return opts


@MODELS.register_module()
class MobileViT_CVNet(BaseModule):
    """
    Wrap Apple ml-cvnets MobileViT as an MMDetection backbone.

    Returns multi-scale feature maps suitable for FPN, e.g. (out_l2, out_l3, out_l4, out_l5).
    Apple CVNets provides these via BaseImageEncoder.extract_end_points_all(). :contentReference[oaicite:3]{index=3}
    """

    def __init__(
        self,
        mode: str = "small",  # "xx_small", "x_small", "small"
        out_keys: Tuple[str, ...] = ("out_l2", "out_l3", "out_l4", "out_l5"),
        use_l5_exp: bool = False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.opts = _make_minimal_cvnets_opts(mode=mode)
        self.backbone = MobileViT(self.opts)  # builds conv_1, layer_1..layer_5, conv_1x1_exp, classifier :contentReference[oaicite:4]{index=4}

        self.out_keys = out_keys
        self.use_l5_exp = use_l5_exp

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # CVNets returns a dict with keys: out_l1..out_l5 (and optionally out_l5_exp)
        feats = self.backbone.extract_end_points_all(
            x,
            use_l5=("out_l5" in self.out_keys or "out_l5_exp" in self.out_keys),
            use_l5_exp=self.use_l5_exp,
        )  # :contentReference[oaicite:5]{index=5}

        outs: List[torch.Tensor] = []
        for k in self.out_keys:
            if k not in feats:
                raise KeyError(
                    f"Requested key '{k}' not in CVNets outputs. Available: {list(feats.keys())}"
                )
            outs.append(feats[k])
        return tuple(outs)

    def load_cvnets_checkpoint(self, ckpt_path: str, strict: bool = False) -> None:
        """
        Apple checkpoints sometimes store 'state_dict' or nested dicts.
        This helper tries common patterns.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

        missing, unexpected = self.backbone.load_state_dict(sd, strict=strict)
        if missing or unexpected:
            print("[MobileViT_CVNet] missing keys:", missing)
            print("[MobileViT_CVNet] unexpected keys:", unexpected)


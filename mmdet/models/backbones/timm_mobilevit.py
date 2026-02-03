import timm
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class TimmMobileViT(BaseModule):

    def __init__(
        self,
        model_name='mobilevit_s',
        out_indices=(2, 3, 4),
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)

        self.backbone = timm.create_model(
            model_name,
            pretrained=(init_cfg is not None),
            features_only=True,
            out_indices=out_indices
        )

        self.out_channels = tuple(self.backbone.feature_info.channels())

    def forward(self, x):
        return tuple(self.backbone(x))


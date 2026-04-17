"""DINOv2 ViT-B/14 + MLP baseline (CLS-only, no VLAD).

The CONTEXT.md week-2 baseline for reference. Lightweight: ~86 M backbone
params + tiny MLP head, projects the CLS token to a 256-d unit-norm
descriptor so similarity scores are directly cosines.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from visword.config import Config
from visword.models.dinov2_salad import _freeze_frozen_backbone_blocks
from visword.models.salad_bridge import OfficialDINOv2


class DINOv2CLS(nn.Module):
    """DINOv2 backbone + 2-layer MLP head returning a 256-d L2-normed vector."""

    OUTPUT_DIM = 256

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.backbone = OfficialDINOv2(
            model_name=cfg.backbone.arch,
            num_trainable_blocks=cfg.backbone.num_trainable_blocks,
            return_token=True,
            norm_layer=True,
        )
        _freeze_frozen_backbone_blocks(self.backbone, cfg.backbone.num_trainable_blocks)

        self.head = nn.Sequential(
            nn.Linear(cfg.backbone.feature_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.OUTPUT_DIM),
        )

    @property
    def descriptor_dim(self) -> int:
        return self.OUTPUT_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, cls_token = self.backbone(x)
        return F.normalize(self.head(cls_token), p=2, dim=-1)

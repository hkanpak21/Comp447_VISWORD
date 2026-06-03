"""I-JEPA ViT-H/14 with partial fine-tuning + MLP head.

End-to-end trainable model that unfreezes the last N blocks of the I-JEPA
encoder and projects the mean-pooled patch tokens through a 2-layer MLP
to a 256-d L2-normalised descriptor.

This is the I-JEPA analogue of ``DINOv2CLS`` — mean-pool + MLP, no
SALAD aggregator. The intent is to isolate the effect of backbone
fine-tuning: if partial fine-tuning pushes I-JEPA's R@10 from 0.18
(frozen + linear adapter) toward DINOv2+MLP levels (0.74), we know the
backbone is learning document-relevant features.

Usage::

    model_kind: ijepa_finetune
    backbone:
      arch: ijepa_vith14
      feature_dim: 1280
      num_trainable_blocks: 2   # unfreeze last 2 of 32 blocks
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from visword.config import Config
from visword.models.ijepa_backbone import IJepaBackbone


class IJepaFinetune(nn.Module):
    """Partially fine-tuned I-JEPA + MLP head → 256-d L2-normed descriptor."""

    OUTPUT_DIM = 256

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.backbone = IJepaBackbone(
            num_trainable_blocks=cfg.backbone.num_trainable_blocks,
        )

        d_in = cfg.backbone.feature_dim   # 1280
        self.head = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.GELU(),
            nn.Linear(512, self.OUTPUT_DIM),
        )

    @property
    def descriptor_dim(self) -> int:
        return self.OUTPUT_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _patch_tokens, mean_pool = self.backbone(x)
        return F.normalize(self.head(mean_pool), p=2, dim=-1)

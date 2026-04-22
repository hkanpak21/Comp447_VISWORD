"""CLIP-image backbone + SALAD aggregator (row 23) and CLIP-image + MLP head
(row 22). Same head/aggregator implementations as the DINOv2 family —
backbone swap only. Lets us isolate "is the SALAD>CLS gap intrinsic to
SALAD or backbone-dependent?".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from visword.config import Config
from visword.models.clip_backbone import CLIPImageBackbone, freeze_clip_backbone_blocks
from visword.models.salad_ablations import AblatedSALAD, descriptor_dim_for


class CLIPSALAD(nn.Module):
    """Row 23 — CLIP-ViT-B/16 image backbone + AblatedSALAD aggregator."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = CLIPImageBackbone(num_trainable_blocks=cfg.backbone.num_trainable_blocks)
        freeze_clip_backbone_blocks(self.backbone, cfg.backbone.num_trainable_blocks)
        self.aggregator = AblatedSALAD(
            num_channels=CLIPImageBackbone.NUM_CHANNELS,
            num_clusters=cfg.salad.num_clusters,
            cluster_dim=cfg.salad.cluster_dim,
            token_dim=cfg.salad.token_dim,
            ablation=cfg.salad.ablation,
            sinkhorn_iters=cfg.salad.sinkhorn_iters,
        )

    @property
    def descriptor_dim(self) -> int:
        return descriptor_dim_for(
            self.cfg.salad.ablation,
            self.cfg.salad.num_clusters,
            self.cfg.salad.cluster_dim,
            self.cfg.salad.token_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches, cls_token = self.backbone(x)
        return self.aggregator((patches, cls_token))


class CLIPCLS(nn.Module):
    """Row 22 — CLIP image backbone + 2-layer MLP head returning 256-d L2-normed."""

    OUTPUT_DIM = 256

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = CLIPImageBackbone(num_trainable_blocks=cfg.backbone.num_trainable_blocks)
        freeze_clip_backbone_blocks(self.backbone, cfg.backbone.num_trainable_blocks)
        self.head = nn.Sequential(
            nn.Linear(CLIPImageBackbone.NUM_CHANNELS, 512),
            nn.GELU(),
            nn.Linear(512, self.OUTPUT_DIM),
        )

    @property
    def descriptor_dim(self) -> int:
        return self.OUTPUT_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, cls_token = self.backbone(x)
        return F.normalize(self.head(cls_token), p=2, dim=-1)


__all__ = ["CLIPSALAD", "CLIPCLS"]

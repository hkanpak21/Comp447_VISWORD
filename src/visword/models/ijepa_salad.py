"""I-JEPA ViT-H/14 with SALAD aggregator.

End-to-end trainable model that unfreezes the last N blocks of the I-JEPA
encoder and projects the patch tokens through the SALAD optimal-transport
aggregator.

Usage::

    model_kind: ijepa_salad
    backbone:
      arch: ijepa_vith14
      feature_dim: 1280
      num_trainable_blocks: 4
      pretrained_checkpoint: "runs/..."
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from visword.config import Config
from visword.models.ijepa_backbone import IJepaBackbone
from visword.models.salad_ablations import AblatedSALAD, descriptor_dim_for


class IJepaSALAD(nn.Module):
    """Partially fine-tuned I-JEPA + SALAD aggregator."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.backbone = IJepaBackbone(
            num_trainable_blocks=cfg.backbone.num_trainable_blocks,
            pretrained_checkpoint=getattr(cfg.backbone, "pretrained_checkpoint", None)
        )

        d_in = cfg.backbone.feature_dim   # 1280
        self.aggregator = AblatedSALAD(
            num_channels=d_in,
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
        patch_tokens, mean_pool = self.backbone(x)
        # SALAD expects (B, C, H, W)
        B, N, C = patch_tokens.shape
        # Assuming square patches (e.g. 16x16 for 224x224 input, or 35x35 for 490x490)
        side = int(N ** 0.5)
        assert side * side == N, f"Patch sequence length {N} is not a perfect square."
        
        # Reshape to (B, C, H, W)
        patch_spatial = patch_tokens.transpose(1, 2).view(B, C, side, side)
        
        desc = self.aggregator((patch_spatial, mean_pool))
        return F.normalize(desc, p=2, dim=-1)

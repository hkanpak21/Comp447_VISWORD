"""DINOv2 ViT-B/14 + official SALAD aggregator (PROJECT_SPEC.md §5 + §6).

The official SALAD aggregator already concatenates an MLP-projected CLS token
with the VLAD matrix before the final L2 norm (third_party/salad/.../salad.py
line 137-139). Our week-2 hand-rolled version dropped that branch and
underperformed the CLS-only baseline — see CONTEXT.md session 2. This
wrapper delegates to the vendored code so the CLS branch is back by default.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from visword.config import Config
from visword.models.salad_bridge import OfficialDINOv2, OfficialSALAD


def _freeze_frozen_backbone_blocks(backbone: OfficialDINOv2, num_trainable: int) -> None:
    """Set ``requires_grad=False`` on the pre-``num_trainable`` ViT blocks.

    ``OfficialDINOv2.forward`` already wraps the frozen blocks in
    ``torch.no_grad()`` so they don't contribute gradients, but their
    parameters still have ``requires_grad=True``. Explicitly freezing keeps
    the optimizer from allocating state for them and makes
    ``sum(p.numel() for p in model.parameters() if p.requires_grad)`` honest.

    Everything below the ``blocks`` list (patch_embed, positional embeddings,
    cls_token, mask_token) is also frozen for the same reason.
    """
    model = backbone.model
    # Backbone-level params outside the blocks (embeddings, tokens) are shared
    # with the frozen path and never re-trained in Izquierdo & Civera's recipe.
    for name, param in model.named_parameters(recurse=False):
        param.requires_grad = False
    for name, mod in model.named_children():
        if name == "blocks":
            continue
        for p in mod.parameters():
            p.requires_grad = False

    for blk in model.blocks[:-num_trainable]:
        for p in blk.parameters():
            p.requires_grad = False


class DINOv2SALAD(nn.Module):
    """DINOv2 backbone (last N blocks trainable) + SALAD aggregator.

    Forward input : (B, 3, H, W); H, W divisible by 14.
    Forward output: (B, num_clusters * cluster_dim + token_dim), L2-normalised
        (done inside the SALAD aggregator).
    """

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

        self.aggregator = OfficialSALAD(
            num_channels=cfg.backbone.feature_dim,
            num_clusters=cfg.salad.num_clusters,
            cluster_dim=cfg.salad.cluster_dim,
            token_dim=cfg.salad.token_dim,
        )

    @property
    def descriptor_dim(self) -> int:
        return self.cfg.salad.num_clusters * self.cfg.salad.cluster_dim + self.cfg.salad.token_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches, cls_token = self.backbone(x)
        return self.aggregator((patches, cls_token))

"""CLIP-ViT-B/16 image branch wrapper exposing the same `(patches, cls_token)`
interface as :class:`OfficialDINOv2`, so it drops into our SALAD / MLP heads.

Differences from DINOv2:
  - Patch grid is 14×14 (not 16×16) at 224 input.
  - Hidden dim is 768 (same) — but the post-LN CLS comes BEFORE CLIP's image
    projection (which would map 768 → 512). We deliberately use the
    pre-projection 768-d features so the SALAD/MLP heads keep their
    expected num_channels=768.
  - The dataset transform on disk uses ImageNet mean/std; we inverse-then
    re-apply CLIP mean/std inside the forward so the dataset stays shared
    with DINOv2 runs (no per-model collate needed).

Trainable-block freezing mirrors `_freeze_frozen_backbone_blocks` from
``dinov2_salad.py`` — only the last `num_trainable_blocks` ResAttn blocks
plus `ln_post` receive gradients.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Install DNS shim before importing open_clip (which downloads weights from HF on
# first construction). Compute nodes SERVFAIL huggingface.co.
from visword.hf_dns_shim import install as _install_dns_shim  # noqa: E402
_install_dns_shim()


class CLIPImageBackbone(nn.Module):
    """Wraps `open_clip.create_model('ViT-B-16', pretrained='openai').visual`.

    forward(x) -> (patches, cls_token) where:
      patches    : (B, 768, 14, 14)
      cls_token  : (B, 768)
    """

    NUM_CHANNELS = 768
    PATCH_GRID = 14   # 224 / 16 = 14

    # CLIP normalisation stats (different from ImageNet).
    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    # The dataset transform applies these:
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, num_trainable_blocks: int = 4) -> None:
        super().__init__()
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        self.visual = model.visual
        self.num_trainable = num_trainable_blocks

        # Renormalisation buffers.
        def _t(v): return torch.tensor(v).view(1, 3, 1, 1)
        self.register_buffer("_in_mean", _t(self._IMAGENET_MEAN))
        self.register_buffer("_in_std", _t(self._IMAGENET_STD))
        self.register_buffer("_clip_mean", _t(self._CLIP_MEAN))
        self.register_buffer("_clip_std", _t(self._CLIP_STD))

    def _renorm(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self._in_std + self._in_mean - self._clip_mean) / self._clip_std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._renorm(x)
        v = self.visual

        # Patch embed: (B, 3, 224, 224) -> (B, 768, 14, 14)
        x = v.conv1(x)
        B, C, H, W = x.shape
        # Flatten patches: (B, C, H*W) -> (B, H*W, C)
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        # Prepend learnable CLS, add positional embedding.
        cls = v.class_embedding.to(x.dtype) + torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                  # (B, 1+H*W, C)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)                          # open_clip uses (L, B, D)

        n_blocks = len(v.transformer.resblocks)
        for i, blk in enumerate(v.transformer.resblocks):
            if i < n_blocks - self.num_trainable:
                with torch.no_grad():
                    x = blk(x)
            else:
                x = blk(x)

        x = x.permute(1, 0, 2)                          # (B, 1+H*W, C)
        x = v.ln_post(x)
        cls_token = x[:, 0]                             # (B, C)
        patches = x[:, 1:].permute(0, 2, 1).reshape(B, C, H, W)
        return patches, cls_token


def freeze_clip_backbone_blocks(backbone: CLIPImageBackbone, num_trainable: int) -> None:
    """Mirror of `_freeze_frozen_backbone_blocks`. ln_post stays trainable."""
    v = backbone.visual
    for p in v.conv1.parameters():
        p.requires_grad = False
    v.class_embedding.requires_grad = False
    v.positional_embedding.requires_grad = False
    for p in v.ln_pre.parameters():
        p.requires_grad = False
    n = len(v.transformer.resblocks)
    for i, blk in enumerate(v.transformer.resblocks):
        rg = i >= n - num_trainable
        for p in blk.parameters():
            p.requires_grad = rg
    # ln_post + (any extra projection on visual) — leave trainable
    # so the head can adapt.


__all__ = ["CLIPImageBackbone", "freeze_clip_backbone_blocks"]

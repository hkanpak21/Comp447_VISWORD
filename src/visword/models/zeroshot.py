"""Zero-shot (frozen, no-training) backbones.

Each class exposes the same interface as our trained models
(`descriptor_dim` property + `forward(x) -> (B, D)` L2-normalised), so the
same eval pipeline can score them.

Rows from the method ladder this module supports:
  - Row 4: ZeroShotDINOv2(mode="cls")       → 768-d CLS token
  - Row 5: ZeroShotDINOv2(mode="mean_patch") → 768-d mean-pooled patches
  - Row 6: ZeroShotCLIPImage                 → 512-d CLIP image projection

No parameters are trainable. The classes still respect the current device
(moved via `.to(device)` by the eval driver).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from visword.config import Config
from visword.models.salad_bridge import OfficialDINOv2


class ZeroShotDINOv2(nn.Module):
    """Frozen DINOv2-ViT-B/14. Returns either the CLS token or the
    mean-pooled patch features, L2-normalised.

    `mode="cls"`        → descriptor_dim = cfg.backbone.feature_dim (768)
    `mode="mean_patch"` → descriptor_dim = cfg.backbone.feature_dim (768)
    """

    def __init__(self, cfg: Config, *, mode: str = "cls") -> None:
        super().__init__()
        if mode not in {"cls", "mean_patch"}:
            raise ValueError(f"unknown zero-shot mode {mode!r}; expected 'cls' or 'mean_patch'")
        self.cfg = cfg
        self.mode = mode
        # OfficialDINOv2 uses `blocks[:-num_trainable_blocks]` which, for
        # num_trainable_blocks=0, evaluates to `blocks[:0]` = empty — meaning
        # no blocks would be wrapped in torch.no_grad() and ALL would run
        # outside it. Use num_trainable_blocks=1 so the first N-1 blocks are
        # in the no_grad branch; we additionally set requires_grad=False on
        # every parameter so the "trainable" last block produces no gradients
        # either. Net effect: fully frozen zero-shot backbone.
        self.backbone = OfficialDINOv2(
            model_name=cfg.backbone.arch,
            num_trainable_blocks=1,
            return_token=True,
            norm_layer=True,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

    @property
    def descriptor_dim(self) -> int:
        return self.cfg.backbone.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches, cls_token = self.backbone(x)
        if self.mode == "cls":
            return F.normalize(cls_token, p=2, dim=-1)
        # mean_patch: (B, C, H, W) → mean over spatial dims → (B, C)
        mean = patches.flatten(2).mean(dim=-1)
        return F.normalize(mean, p=2, dim=-1)


class ZeroShotCLIPImage(nn.Module):
    """Row 6 — CLIP-ViT-B/16 image branch, frozen. Returns the 512-d image
    projection, L2-normed."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai')
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def descriptor_dim(self) -> int:
        return 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.normalize(self.model.encode_image(x).float(), p=2, dim=-1)


class ZeroShotImageNetViT(nn.Module):
    """Row 2 — ImageNet-21k + 1k supervised ViT-B/16 (timm), frozen."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        import timm
        self.vit = timm.create_model(
            'vit_base_patch16_224.augreg_in21k_ft_in1k',
            pretrained=True, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False

    @property
    def descriptor_dim(self) -> int:
        return 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.normalize(self.vit(x).float(), p=2, dim=-1)


class DINOv2LinearProbe(nn.Module):
    """Row 7 — frozen DINOv2 backbone + trainable Linear head.

    The single linear layer (768 → 256, L2-normed) is the ONLY trainable
    parameter. Tests how much retrieval signal is already linearly decodable
    from frozen DINOv2 features, isolating it from what our last-4-block
    fine-tuning adds (rows 8/9) and from the MLP non-linearity (row 8+).
    """

    OUTPUT_DIM = 256

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = OfficialDINOv2(
            model_name=cfg.backbone.arch,
            num_trainable_blocks=1,
            return_token=True,
            norm_layer=True,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(cfg.backbone.feature_dim, self.OUTPUT_DIM, bias=True)

    @property
    def descriptor_dim(self) -> int:
        return self.OUTPUT_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, cls_token = self.backbone(x)
        return F.normalize(self.head(cls_token), p=2, dim=-1)


__all__ = [
    "ZeroShotDINOv2", "ZeroShotCLIPImage", "ZeroShotImageNetViT",
    "DINOv2LinearProbe",
]

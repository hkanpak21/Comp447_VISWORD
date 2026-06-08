"""Ticket 02 — shared model-wrapper for the legible-resolution comparison grid.

One factory, ``build_encoder(name, cfg)``, returns a frozen encoder with the
common interface already used across the repo (``descriptor_dim`` property +
``forward((B, 3, 224, 224)) -> (B, D)`` L2-normalised, ImageNet-normalised input),
so the same encode/score pipeline runs every model apples-to-apples.

Reuses the existing ``ZeroShot*`` classes; adds MAE (``ZeroShotMAE``). Multi-vector
models (ColPali) are out of scope here — they route to late-interaction scoring in
ticket 03, not through this single-vector wrapper.
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

from visword.config import Config
from visword.models.zeroshot import (
    ZeroShotCLIPImage,
    ZeroShotDINOv2,
    ZeroShotIJepa,
    ZeroShotImageNetViT,
    ZeroShotMAE,
    ZeroShotPlainViT,
    ZeroShotSigLIP,
    ZeroShotPix2Struct,
    ZeroShotDonut,
    ZeroShotNougat,
)

# name -> builder(cfg). Order = the grid's reporting order.
_ENCODERS: dict[str, Callable[[Config | None], nn.Module]] = {
    "clip": lambda cfg: ZeroShotCLIPImage(cfg),
    "siglip": lambda cfg: ZeroShotSigLIP(cfg),
    "dinov2_cls": lambda cfg: ZeroShotDINOv2(cfg, mode="cls"),
    "dinov2_mean": lambda cfg: ZeroShotDINOv2(cfg, mode="mean_patch"),
    "imagenet_vit": lambda cfg: ZeroShotImageNetViT(cfg),
    "random_vit": lambda cfg: ZeroShotPlainViT(cfg),
    "ijepa": lambda cfg: ZeroShotIJepa(cfg),
    "mae": lambda cfg: ZeroShotMAE(cfg),
    # Doc-pretrained family (ticket 03) — OCR-free document encoders.
    "pix2struct": lambda cfg: ZeroShotPix2Struct(cfg),
    "donut": lambda cfg: ZeroShotDonut(cfg),
    "nougat": lambda cfg: ZeroShotNougat(cfg),
}

#: Canonical grid order; ``ijepa`` is ViT-H/14 (not matched-arch — see zeroshot.py).
ENCODER_NAMES: tuple[str, ...] = tuple(_ENCODERS)


def build_encoder(name: str, cfg: Config | None = None) -> nn.Module:
    """Build a frozen encoder by name. ``cfg`` is required for the DINOv2 variants
    (they read ``cfg.backbone.arch`` / ``cfg.backbone.feature_dim``); the others
    use fixed pretrained checkpoints and ignore it."""
    if name not in _ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; expected one of {ENCODER_NAMES}")
    return _ENCODERS[name](cfg)


__all__ = ["build_encoder", "ENCODER_NAMES"]

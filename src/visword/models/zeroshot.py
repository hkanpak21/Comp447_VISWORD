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

# Install DNS shim BEFORE any lazy timm / open_clip / transformers import
# (compute nodes SERVFAIL huggingface.co on the internal resolver).
from visword.hf_dns_shim import install as _install_dns_shim  # noqa: E402
_install_dns_shim()

from visword.config import Config
from visword.models.salad_bridge import OfficialDINOv2


# Dataset transform applies ImageNet normalisation. Encoders trained with
# different stats need inline renormalisation in their forward pass; this
# helper avoids per-encoder dataset variants.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _renorm_module(target_mean: tuple[float, float, float],
                   target_std: tuple[float, float, float]) -> nn.Module:
    """Inline ImageNet→target renormalisation as registered buffers."""
    class _Renorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            def _t(v): return torch.tensor(v).view(1, 3, 1, 1)
            self.register_buffer("_in_mean", _t(_IMAGENET_MEAN))
            self.register_buffer("_in_std", _t(_IMAGENET_STD))
            self.register_buffer("_tgt_mean", _t(target_mean))
            self.register_buffer("_tgt_std", _t(target_std))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (x * self._in_std + self._in_mean - self._tgt_mean) / self._tgt_std
    return _Renorm()


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
    projection, L2-normed.

    Adds inline ImageNet→CLIP renormalisation (the dataset transform
    applies ImageNet stats; CLIP was pretrained with its own mean/std).
    Older runs of this class did *not* renormalise — those R@k numbers
    are a few points off as a result.
    """

    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai')
        for p in self.model.parameters():
            p.requires_grad = False
        self.renorm = _renorm_module(self._CLIP_MEAN, self._CLIP_STD)

    @property
    def descriptor_dim(self) -> int:
        return 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.renorm(x)
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


class ZeroShotSigLIP(nn.Module):
    """Frozen SigLIP-ViT-B/16 image branch (Zhai 2023; sigmoid pairwise loss).

    The SigLIP ↔ CLIP comparison isolates *loss form* (sigmoid vs softmax
    InfoNCE) under matched architecture, data, and image-text supervision.
    Both image and text dims are 768 for the base model, but
    ``encode_image`` returns the *projection* used in the contrastive loss.
    """

    HF_NAME = "google/siglip-base-patch16-224"

    # SigLIP preprocessor uses [-1, 1] range (mean=0.5, std=0.5).
    _SIGLIP_MEAN = (0.5, 0.5, 0.5)
    _SIGLIP_STD = (0.5, 0.5, 0.5)

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(self.HF_NAME)
        for p in self.model.parameters():
            p.requires_grad = False
        self.renorm = _renorm_module(self._SIGLIP_MEAN, self._SIGLIP_STD)

    @property
    def descriptor_dim(self) -> int:
        return int(self.model.config.vision_config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.renorm(x)
            feats = self.model.get_image_features(pixel_values=x).float()
        return F.normalize(feats, p=2, dim=-1)


class ZeroShotSigLIPText(nn.Module):
    """SigLIP text branch — companion to ZeroShotSigLIP for the Platonic grid."""

    HF_NAME = "google/siglip-base-patch16-224"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(self.HF_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.HF_NAME)
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def descriptor_dim(self) -> int:
        return int(self.model.config.text_config.hidden_size)

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: torch.device | str = "cpu") -> torch.Tensor:
        enc = self.tokenizer(texts, padding="max_length", truncation=True,
                             max_length=64, return_tensors="pt").to(device)
        feats = self.model.get_text_features(**enc).float()
        return F.normalize(feats, p=2, dim=-1)


class ZeroShotPlainViT(nn.Module):
    """Random-init ViT-B/16, frozen — sanity floor for the encoder grid.

    Tests whether *any* reasonable initialisation gives non-trivial retrieval
    on Wikipedia screenshots, distinguishing "pretraining did something" from
    "the architecture alone is enough."
    """

    SEED = 1234

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        import timm
        # Deterministic random init so two zero-shot runs agree.
        prev = torch.random.get_rng_state()
        torch.manual_seed(self.SEED)
        try:
            self.vit = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=0)
        finally:
            torch.random.set_rng_state(prev)
        for p in self.vit.parameters():
            p.requires_grad = False

    @property
    def descriptor_dim(self) -> int:
        return 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.normalize(self.vit(x).float(), p=2, dim=-1)


class ZeroShotIJepa(nn.Module):
    """Frozen I-JEPA encoder (Assran 2023) — image-only predictive
    pretraining, no text supervision. Tests whether predictive objectives
    induce text-aligned representations without the contrastive crutch.

    Loads ``facebook/ijepa_vith14_1k`` from HuggingFace transformers (the
    ViT-H/14 variant; I-JEPA's released models are H-scale, no B variant
    publicly). Note: this is the *only* encoder in the grid that is not
    ViT-B; documented as an asymmetry — it participates in the Platonic
    alignment grid but not in the matched-architecture RQ2 retrieval
    comparison.
    """

    HF_NAME = "facebook/ijepa_vith14_1k"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(self.HF_NAME)
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def descriptor_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # I-JEPA returns BaseModelOutput; pool patch tokens (no CLS).
        with torch.no_grad():
            out = self.model(pixel_values=x)
            # last_hidden_state: (B, num_patches, hidden_size). Mean-pool.
            feats = out.last_hidden_state.mean(dim=1).float()
        return F.normalize(feats, p=2, dim=-1)


__all__ = [
    "ZeroShotDINOv2", "ZeroShotCLIPImage", "ZeroShotImageNetViT",
    "ZeroShotSigLIP", "ZeroShotSigLIPText", "ZeroShotPlainViT", "ZeroShotIJepa",
    "DINOv2LinearProbe",
]

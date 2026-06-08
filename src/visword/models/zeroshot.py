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
            out = self.model(pixel_values=x, interpolate_pos_encoding=True)
            # last_hidden_state: (B, num_patches, hidden_size). Mean-pool.
            feats = out.last_hidden_state.mean(dim=1).float()
        return F.normalize(feats, p=2, dim=-1)


class ZeroShotMAE(nn.Module):
    """Frozen MAE encoder (He 2022) — pixel-reconstruction masked autoencoder,
    image-only, no text supervision. The new document-family-adjacent baseline
    for the legible grid and the base for our own reader (ticket 04); contrasts
    with I-JEPA (feature-prediction) as pixel-reconstruction SSL.

    Loads ``facebook/vit-mae-base`` (ViT-B/16, 768-d, 224x224). MAE's forward
    randomly masks 75% of patches by default — we set ``mask_ratio = 0`` so the
    FULL image is encoded and the pooled embedding is deterministic. MAE has no
    contrastively-trained CLS, so we mean-pool the patch tokens (as for I-JEPA).
    MAE was pretrained with ImageNet stats, matching the dataset transform — no
    inline renormalisation needed.
    """

    HF_NAME = "facebook/vit-mae-base"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        from transformers import ViTMAEModel
        self.model = ViTMAEModel.from_pretrained(self.HF_NAME)
        # Encode ALL patches (no random masking) -> deterministic embedding.
        self.model.config.mask_ratio = 0.0
        if hasattr(self.model, "embeddings"):
            self.model.embeddings.config.mask_ratio = 0.0
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def descriptor_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(pixel_values=x)
            # last_hidden_state: (B, 1 + num_patches, hidden); index 0 is CLS.
            # Mean-pool the patch tokens (permutation-invariant, so robust to the
            # MAE shuffle even though mask_ratio=0 keeps them all).
            feats = out.last_hidden_state[:, 1:, :].mean(dim=1).float()
        return F.normalize(feats, p=2, dim=-1)


class ZeroShotPix2Struct(nn.Module):
    """Frozen Pix2Struct-base vision encoder (Lee 2023, ICML Oral).

    Pix2Struct is an encoder-decoder pretrained to parse masked webpage
    screenshots into simplified HTML. We extract only the **patch encoder**
    (``Pix2StructVisionModel``) and mean-pool its patch hidden states into a
    single vector. This gives a 768-d embedding comparable to ViT-B baselines.

    ``Pix2StructVisionModel.forward`` requires ``flattened_patches`` produced
    by ``Pix2StructImageProcessor`` — it does NOT accept raw ``pixel_values``.
    The forward pass therefore un-normalises the ImageNet-normalised input
    tensor back to [0, 1], converts each image to PIL, runs the processor
    (which performs the model's HiRes patchification), and then calls the
    vision model. All processing is done within ``torch.no_grad()``.
    """

    HF_NAME = "google/pix2struct-base"

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        import numpy as np
        from transformers import Pix2StructVisionModel, Pix2StructImageProcessor
        self.model = Pix2StructVisionModel.from_pretrained(self.HF_NAME)
        for p in self.model.parameters():
            p.requires_grad = False
        self.processor = Pix2StructImageProcessor.from_pretrained(self.HF_NAME)
        self._np = np
        # Buffers for ImageNet denormalisation (tensor → [0,1] → PIL).
        self.register_buffer("_in_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_in_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    @property
    def descriptor_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def _to_pil_list(self, x: torch.Tensor):
        """Denorm ImageNet tensor (B,3,H,W) → list of PIL Images."""
        from PIL import Image as _Image
        x01 = (x * self._in_std + self._in_mean).clamp(0, 1).cpu()
        pils = []
        for img in x01:
            arr = (img.permute(1, 2, 0).numpy() * 255).astype(self._np.uint8)
            pils.append(_Image.fromarray(arr))
        return pils

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        with torch.no_grad():
            pils = self._to_pil_list(x)
            inputs = self.processor(
                images=pils, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = self.model(**inputs)
            feats = out.last_hidden_state.mean(dim=1).float()
        return F.normalize(feats, p=2, dim=-1)


class ZeroShotDonut(nn.Module):
    """Frozen Donut-base vision encoder (Kim 2022, ECCV).

    Donut (Document Understanding Transformer) is an OCR-free
    encoder-decoder for document understanding. We extract only the Swin
    Transformer vision encoder and mean-pool its last feature map into a
    single vector. This gives a comparable dense document representation
    without any OCR or text decoder involvement.

    Donut uses its own mean/std stats; we inline-renormalise from ImageNet.
    Input is resized to 224×224 — Donut's encoder is flexible via SwinTransformer.
    """

    HF_NAME = "naver-clova-ix/donut-base"
    # Donut image normalisation: same as ImageNet (it uses the SwinT defaults).
    _DONUT_MEAN = (0.485, 0.456, 0.406)
    _DONUT_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        from transformers import DonutSwinModel
        self.model = DonutSwinModel.from_pretrained(self.HF_NAME)
        for p in self.model.parameters():
            p.requires_grad = False
        # Donut uses ImageNet stats — no renorm needed; register identity.
        self.renorm = _renorm_module(self._DONUT_MEAN, self._DONUT_STD)

    @property
    def descriptor_dim(self) -> int:
        # SwinTransformer last-stage hidden dim.
        return int(self.model.config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.renorm(x)
            out = self.model(pixel_values=x)
            # last_hidden_state: (B, seq_len, hidden_size). Mean-pool spatial tokens.
            feats = out.last_hidden_state.mean(dim=1).float()
        return F.normalize(feats, p=2, dim=-1)


class ZeroShotNougat(nn.Module):
    """Frozen Nougat-base vision encoder (Blecher 2023).

    Nougat is an OCR-free academic document parser (arXiv PDFs → Markdown).
    It uses a Swin Transformer encoder identical in structure to Donut-base
    (same HF architecture class, different weights). We mean-pool the last
    hidden state of the encoder for a single dense page embedding.
    """

    HF_NAME = "facebook/nougat-base"
    # Nougat uses standard ImageNet normalisation (inherited from SwinT).
    _NOUGAT_MEAN = (0.485, 0.456, 0.406)
    _NOUGAT_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        from transformers import DonutSwinModel
        # Nougat shares the DonutSwinModel architecture.
        self.model = DonutSwinModel.from_pretrained(self.HF_NAME)
        for p in self.model.parameters():
            p.requires_grad = False
        self.renorm = _renorm_module(self._NOUGAT_MEAN, self._NOUGAT_STD)

    @property
    def descriptor_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.renorm(x)
            out = self.model(pixel_values=x)
            feats = out.last_hidden_state.mean(dim=1).float()
        return F.normalize(feats, p=2, dim=-1)


__all__ = [
    "ZeroShotDINOv2", "ZeroShotCLIPImage", "ZeroShotImageNetViT",
    "ZeroShotSigLIP", "ZeroShotSigLIPText", "ZeroShotPlainViT", "ZeroShotIJepa",
    "ZeroShotMAE", "DINOv2LinearProbe",
    "ZeroShotPix2Struct", "ZeroShotDonut", "ZeroShotNougat",
]

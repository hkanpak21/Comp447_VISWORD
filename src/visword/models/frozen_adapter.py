"""Frozen backbone + trainable adapter head for the adapter-capacity ablation.

Supports three backbone families with a unified interface:
  - ``dinov2_vitb14`` (768-d CLS token)
  - ``clip_vitb16``   (768-d pre-projection CLS from the CLIP visual tower)
  - ``ijepa_vith14``  (1280-d mean-pooled patch tokens)

The backbone is *completely frozen* (requires_grad=False on every parameter).
Only the adapter head receives gradients.  This isolates the effect of adapter
capacity (linear / mlp / bottleneck) from backbone fine-tuning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from visword.config import Config
from visword.models.adapters import build_adapter


class FrozenBackboneAdapter(nn.Module):
    """Frozen encoder backbone + trainable adapter → L2-normed descriptor."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        arch = cfg.backbone.arch

        if arch == "dinov2_vitb14":
            self.backbone = self._build_dinov2(cfg)
            d_in = cfg.backbone.feature_dim  # 768
        elif arch == "clip_vitb16":
            self.backbone = self._build_clip()
            d_in = 768  # pre-projection CLS
        elif arch == "ijepa_vith14":
            self.backbone = self._build_ijepa()
            d_in = 1280
        else:
            raise ValueError(f"unsupported backbone arch {arch!r} for frozen_adapter")

        # Freeze everything in the backbone.
        for p in self.backbone.parameters():
            p.requires_grad = False

        d_out = cfg.adapter.output_dim
        self.adapter = build_adapter(cfg.adapter.kind, d_in, d_out)
        self._descriptor_dim = d_out

    # -- backbone builders --------------------------------------------------

    @staticmethod
    def _build_dinov2(cfg: Config) -> nn.Module:
        from visword.models.salad_bridge import OfficialDINOv2
        return OfficialDINOv2(
            model_name=cfg.backbone.arch,
            num_trainable_blocks=1,  # structural requirement; we freeze all
            return_token=True,
            norm_layer=True,
        )

    @staticmethod
    def _build_clip() -> nn.Module:
        from visword.models.clip_backbone import CLIPImageBackbone
        return CLIPImageBackbone(num_trainable_blocks=0)

    @staticmethod
    def _build_ijepa() -> nn.Module:
        from visword.hf_dns_shim import install as _install_dns_shim
        _install_dns_shim()
        from transformers import AutoModel

        class _IJepaWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k")

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.model(pixel_values=x)
                # Mean-pool patch tokens → (B, 1280)
                return out.last_hidden_state.mean(dim=1)

        return _IJepaWrapper()

    # -- interface ----------------------------------------------------------

    @property
    def descriptor_dim(self) -> int:
        return self._descriptor_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arch = self.cfg.backbone.arch

        with torch.no_grad():
            if arch == "dinov2_vitb14":
                _patches, cls_token = self.backbone(x)
                features = cls_token
            elif arch == "clip_vitb16":
                _patches, cls_token = self.backbone(x)
                features = cls_token
            elif arch == "ijepa_vith14":
                features = self.backbone(x)
            else:
                raise RuntimeError(f"unknown arch {arch!r}")

        features = features.float()
        return F.normalize(self.adapter(features), p=2, dim=-1)

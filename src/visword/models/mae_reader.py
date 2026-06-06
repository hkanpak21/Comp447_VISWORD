"""Ticket 04 — our reader: a fine-tunable MAE that learns to read the body text.

Takes pretrained MAE (``facebook/vit-mae-base``) and teaches it to read by predicting
the frozen BERT[CLS] embedding of the page BODY (regression; objective matches Barış's
I-JEPA text-target, so MAE pixel-reconstruction vs I-JEPA feature-prediction is the only
backbone difference). Parameter-efficient: only the last ``num_trainable_blocks`` MAE
encoder layers + a projection head train; everything else is frozen.

The retrieval descriptor is ``forward(x)`` = L2-normalised predicted-BERT[CLS] direction,
so the same ``page_reid_recall`` protocol scores it; the training loss regresses this
same normalised vector to the (normalised) frozen BERT[CLS] of the body. ``mask_ratio``
is forced to 0 so MAE encodes the full image (no random masking) — deterministic eval.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from visword.hf_dns_shim import install as _install_dns_shim

_install_dns_shim()


class MAEBodyReader(nn.Module):
    """Fine-tunable MAE reader (pooled MAE patch tokens -> predicted BERT[CLS]-of-body)."""

    HF_NAME = "facebook/vit-mae-base"

    def __init__(
        self,
        num_trainable_blocks: int = 4,
        bert_dim: int = 768,
        proj_hidden: int | None = None,
        hf_name: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import ViTMAEModel

        self.model = ViTMAEModel.from_pretrained(hf_name or self.HF_NAME)
        self.model.config.mask_ratio = 0.0
        if hasattr(self.model, "embeddings"):
            self.model.embeddings.config.mask_ratio = 0.0

        # Freeze everything, then unfreeze the last N transformer blocks.
        for p in self.model.parameters():
            p.requires_grad = False
        layers = self.model.encoder.layer  # ViTMAE encoder is a ModuleList of blocks
        self.num_trainable_blocks = max(0, min(num_trainable_blocks, len(layers)))
        for blk in layers[len(layers) - self.num_trainable_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True

        hidden = int(self.model.config.hidden_size)
        if proj_hidden:
            self.head = nn.Sequential(
                nn.Linear(hidden, proj_hidden), nn.GELU(), nn.Linear(proj_hidden, bert_dim))
        else:
            self.head = nn.Linear(hidden, bert_dim)
        self._bert_dim = int(bert_dim)

    @property
    def descriptor_dim(self) -> int:
        return self._bert_dim

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        # last_hidden_state: (B, 1 + num_patches, hidden); index 0 is CLS. Mean-pool patches.
        return out.last_hidden_state[:, 1:, :].mean(dim=1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Raw predicted-BERT[CLS] vector (B, bert_dim) — pre-normalisation."""
        return self.head(self._pool(x).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised retrieval descriptor (== normalised prediction target space)."""
        return F.normalize(self.embed(x), p=2, dim=-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


__all__ = ["MAEBodyReader"]

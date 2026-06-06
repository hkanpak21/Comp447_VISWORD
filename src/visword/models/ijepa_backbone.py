"""I-JEPA ViT-H/14 backbone with selective layer unfreezing.

Loads ``facebook/ijepa_vith14_1k`` via HuggingFace ``transformers`` and
supports partial fine-tuning by unfreezing the last *N* transformer blocks
out of 32 total.  The remaining blocks, the patch/position embeddings, and
the final LayerNorm are frozen.

Architecture (ViT-Huge/14):
    model.embeddings      — patch + position embeddings  (1.08 M params)
    model.encoder.layer   — 32 × IJepaLayer              (19.7 M params each)
    model.layernorm       — final LayerNorm               (2 560 params)

Forward returns ``(patch_tokens, mean_pool)`` so downstream modules can
choose CLS-style pooling or attach a SALAD-style aggregator later.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IJepaBackbone(nn.Module):
    """Frozen-except-last-N I-JEPA ViT-H/14 backbone.

    Parameters
    ----------
    num_trainable_blocks : int
        Number of encoder blocks (from the end) to leave trainable.
        ``0`` = fully frozen, ``2`` = last 2 blocks trainable, etc.
        Maximum 32 (all blocks).
    """

    TOTAL_BLOCKS = 32
    HIDDEN_DIM = 1280

    def __init__(self, num_trainable_blocks: int = 2, pretrained_checkpoint: str | None = None) -> None:
        super().__init__()

        # DNS shim must be installed before any HuggingFace import on Valar.
        from visword.hf_dns_shim import install as _install_dns_shim
        _install_dns_shim()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k")

        if pretrained_checkpoint is not None:
            ckpt = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=True)
            if "encoder" in ckpt:
                state_dict = ckpt["encoder"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
            
            # The checkpoint from ijepa_text_target might have 'model.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=False)

        self.num_trainable_blocks = num_trainable_blocks
        self._freeze(num_trainable_blocks)

    def _freeze(self, num_trainable: int) -> None:
        """Freeze everything except the last ``num_trainable`` encoder blocks.

        Mirrors the pattern in ``dinov2_salad.py:_freeze_frozen_backbone_blocks``
        — explicitly set ``requires_grad=False`` so the optimiser doesn't
        allocate momentum/state for frozen params.
        """
        m = self.model

        # 1. Freeze embeddings (patch_embeddings, position_embeddings).
        for p in m.embeddings.parameters():
            p.requires_grad = False

        # 2. Freeze the final LayerNorm.
        for p in m.layernorm.parameters():
            p.requires_grad = False

        # 3. Freeze encoder blocks [0 .. 32-num_trainable).
        n_frozen = self.TOTAL_BLOCKS - num_trainable
        for block in m.encoder.layer[:n_frozen]:
            for p in block.parameters():
                p.requires_grad = False

        # 4. Ensure the trainable blocks are explicitly unfrozen
        #    (they should already be, but be explicit).
        for block in m.encoder.layer[n_frozen:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, 3, H, W) tensor — supports arbitrary input sizes (e.g. 224x224, 490x490)
            by interpolating positional encodings dynamically.

        Returns
        -------
        patch_tokens : (B, N, 1280) — per-patch hidden states from the last
            encoder layer (N = (H/14)*(W/14) for HxW input with patch_size 14).
        mean_pool : (B, 1280) — mean of ``patch_tokens`` over the sequence
            dimension. This is the I-JEPA-native pooling strategy.
        """
        out = self.model(pixel_values=x, interpolate_pos_encoding=True)
        patch_tokens = out.last_hidden_state      # (B, N, 1280)
        mean_pool = patch_tokens.mean(dim=1)      # (B, 1280)
        return patch_tokens, mean_pool

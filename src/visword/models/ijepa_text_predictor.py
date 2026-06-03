# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import torch
import torch.nn as nn
from visword.models.ijepa_predictor import (
    Block,
    trunc_normal_,
    get_2d_sincos_pos_embed,
    apply_masks,
)

class VisionTransformerTextPredictor(nn.Module):
    """Vision Transformer Predictor that maps visual context patches to text token representations."""

    def __init__(
        self,
        num_patches,
        max_text_tokens=64,
        embed_dim=1280,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        target_dim=768,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        
        # 1D learnable position embeddings for text tokens
        self.predictor_text_pos_embed = nn.Parameter(
            torch.zeros(1, max_text_tokens, predictor_embed_dim)
        )
        
        # 2D fixed positional embedding for visual patches
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=False
        )
        predictor_pos_embed = get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1],
            int(num_patches**.5),
            cls_token=False
        )
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        # Projects to target text representation dimension (768 for BERT)
        self.predictor_proj = nn.Linear(predictor_embed_dim, target_dim, bias=True)

        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.predictor_text_pos_embed, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, target_len: int):
        """
        :param x: Visual context representations from context encoder (B * nenc, N_ctxt, embed_dim)
        :param masks_x: Mask indices of visual context (B, N_ctxt) or list of list (collated masks)
        :param target_len: Sequence length of target text tokens (T)
        """
        # Determine batch size from x and masks_x
        # x is (B * nenc, N_ctxt, embed_dim)
        if isinstance(masks_x, list):
            nenc = len(masks_x)
            masks_x_flat = masks_x
        else:
            # If it's a tensor of shape (B, N_ctxt) or similar
            nenc = 1
            masks_x_flat = [masks_x]

        B = len(x) // nenc

        # 1. Map visual context from encoder-dim to predictor-dim
        x = self.predictor_embed(x)  # (B * nenc, N_ctxt, predictor_embed_dim)

        # 2. Add 2D positional embedding to visual context tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)  # (B, num_patches, predictor_embed_dim)
        x_pos_embed_ctxt = apply_masks(x_pos_embed, masks_x_flat)  # (B * nenc, N_ctxt, predictor_embed_dim)
        x = x + x_pos_embed_ctxt

        _, N_ctxt, D = x.shape

        # 3. Create target query tokens with 1D positional embedding
        # Target tokens are: mask_token + 1D pos_embed
        pos_embs = self.predictor_text_pos_embed[:, :target_len, :]  # (1, T, predictor_embed_dim)
        pos_embs = pos_embs.repeat(B * nenc, 1, 1)  # (B * nenc, T, predictor_embed_dim)

        pred_tokens = self.mask_token.repeat(B * nenc, target_len, 1)
        pred_tokens = pred_tokens + pos_embs  # (B * nenc, T, predictor_embed_dim)

        # 4. Concat visual context and target query tokens
        x = torch.cat([x, pred_tokens], dim=1)  # (B * nenc, N_ctxt + T, predictor_embed_dim)

        # 5. Forward through transformer blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # 6. Extract predictions corresponding to the target query tokens
        x = x[:, N_ctxt:]  # (B * nenc, T, predictor_embed_dim)
        x = self.predictor_proj(x)  # (B * nenc, T, target_dim)

        return x

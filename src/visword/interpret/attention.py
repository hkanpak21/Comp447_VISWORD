"""Last-block CLS→patch attention heatmaps (PROJECT_SPEC.md §8).

DINOv2's Attention module uses ``torch.nn.functional.scaled_dot_product_attention``
(SDPA), which returns the attention output but drops the intermediate
attention matrix. To recover CLS→patch attention for a crop we:

1. Register a forward pre-hook on the last ViT block's ``attn`` module to
   capture its input tensor ``x`` of shape ``(B, N, C)`` (N = 1 + num_patches).
2. Apply the attention's ``qkv`` linear ourselves to get Q / K / V.
3. Reshape into ``(B, H, N, D)`` and compute ``softmax(Q K^T / sqrt(D))``.
4. Take the first row (CLS as query), average over heads, and reshape to
   the ``(side, side)`` patch grid for overlay.

Output artefact: ``<run_dir>/interpret/attention_sample<i>.png`` overlays
the heatmap on the decoded crop. A JSON sidecar records mean/max attn.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class AttentionCapture:
    """CLS→patch attention weights for a single sample."""
    weights: np.ndarray                # shape (side, side), in [0, 1]; sums to ~1
    side: int
    num_heads: int


def compute_cls_to_patch_attention(
    backbone_model: torch.nn.Module,
    images: torch.Tensor,
) -> list[AttentionCapture]:
    """Run ``backbone_model`` on ``images`` and recover last-block CLS attention.

    Args:
        backbone_model: an ``OfficialDINOv2`` wrapper (exposes ``.model.blocks``).
        images: ``(B, 3, H, W)`` image tensor, H and W divisible by 14.

    Returns a list of ``AttentionCapture`` objects, one per image.
    """
    backbone_model.eval()
    last_block = backbone_model.model.blocks[-1]
    attn = last_block.attn

    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(_module, inputs):
        # inputs[0] is the ``x`` about to enter the attention module; pre-norm
        # already applied inside the block so this is post-LN.
        captured["x"] = inputs[0].detach()

    handle = attn.register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            _ = backbone_model(images)
    finally:
        handle.remove()

    x = captured["x"]                                # (B, N, C)
    B, N, C = x.shape
    num_heads = attn.num_heads
    head_dim = C // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Re-apply qkv to the same input the attn module saw.
    with torch.no_grad():
        qkv = attn.qkv(x)                            # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)                      # (B, H, N, D)
        attn_logits = (q @ k.transpose(-2, -1)) * scale        # (B, H, N, N)
        attn_probs = F.softmax(attn_logits, dim=-1)             # (B, H, N, N)

    side = int(round(math.sqrt(N - 1)))
    if side * side != (N - 1):
        raise RuntimeError(
            f"patch count {N - 1} is not a perfect square — patch grid ill-defined"
        )

    cls_attn = attn_probs[:, :, 0, 1:].mean(dim=1)   # (B, N-1), mean over heads
    out: list[AttentionCapture] = []
    for i in range(B):
        w = cls_attn[i].reshape(side, side).cpu().numpy()
        out.append(AttentionCapture(weights=w, side=side, num_heads=num_heads))
    return out


def render_overlay(
    pil_crop: Image.Image,
    capture: AttentionCapture,
    out_path: Path,
) -> Path:
    """Overlay the attention heatmap on the crop, save as PNG + JSON sidecar."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.imshow(pil_crop)
    side = capture.side
    heatmap = capture.weights
    # Upsample to image resolution by nearest-neighbour so the patch grid is visible.
    upsampled = np.kron(heatmap, np.ones((pil_crop.height // side, pil_crop.width // side)))
    # Normalise for display only — the raw weights already sum to ~1.
    vmax = float(upsampled.max()) if upsampled.max() > 0 else 1.0
    ax.imshow(upsampled, cmap="magma", alpha=0.45, vmin=0, vmax=vmax)
    ax.set_axis_off()
    ax.set_title(f"last-block CLS attn  ({capture.num_heads} heads)", fontsize=9)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "side": capture.side,
        "num_heads": capture.num_heads,
        "mean": float(capture.weights.mean()),
        "max": float(capture.weights.max()),
        "entropy": float(-(capture.weights * np.log(capture.weights.clip(min=1e-12))).sum()),
    }, indent=2))
    return out_path


__all__ = [
    "AttentionCapture",
    "compute_cls_to_patch_attention",
    "render_overlay",
]

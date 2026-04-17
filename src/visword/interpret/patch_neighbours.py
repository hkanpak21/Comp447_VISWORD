"""Per-patch nearest-neighbour analysis on anchor / positive / negative crops.

For each anchor patch token, find:
  * the nearest patch in the positive image (ideally it's the "same" region)
  * the farthest patch in the negative image (should be structurally dissimilar)

Render a 3-panel PNG showing the anchor crop with its query patch outlined,
plus the matched positive and farthest negative patches outlined in their
respective crops. Useful for asking: does the model's patch-level geometry
look like what a human would pick?
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


@dataclass
class PatchMatch:
    anchor_patch: int          # flat index into (H*W) patch grid
    positive_patch: int        # nearest positive patch
    negative_patch: int        # farthest negative patch
    positive_cos: float
    negative_cos: float


def _flatten_patches(feature_map: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W)`` -> ``(B, H*W, C)``, L2-normed along C."""
    B, C, H, W = feature_map.shape
    z = feature_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
    return F.normalize(z, p=2, dim=-1)


def _patch_box(side: int, flat_idx: int, image_size: int) -> tuple[int, int, int, int]:
    patch_side = image_size // side
    row, col = divmod(flat_idx, side)
    x0, y0 = col * patch_side, row * patch_side
    return (x0, y0, x0 + patch_side, y0 + patch_side)


def find_patch_matches(
    backbone_model: torch.nn.Module,
    anchor_img: torch.Tensor,
    positive_img: torch.Tensor,
    negative_img: torch.Tensor,
    *,
    k_examples: int = 4,
) -> list[PatchMatch]:
    """Run the backbone on a 3-image batch; pair anchor patches with pos/neg extremes.

    Returns up to ``k_examples`` matches sampled from the top-k anchor
    patches with the most confident positive correspondences (high cosine).
    """
    backbone_model.eval()
    batch = torch.stack([anchor_img, positive_img, negative_img]).to(
        next(backbone_model.parameters()).device
    )
    with torch.no_grad():
        out = backbone_model(batch)
        # OfficialDINOv2 returns either `f` or `(f, t)` — handle both.
        fmap = out[0] if isinstance(out, tuple) else out       # (3, C, H, W)
    patches = _flatten_patches(fmap)                            # (3, N, C)
    anc, pos, neg = patches[0], patches[1], patches[2]

    sim_pos = anc @ pos.T                                       # (N, N)
    sim_neg = anc @ neg.T
    best_pos_cos, best_pos_idx = sim_pos.max(dim=1)
    worst_neg_cos, worst_neg_idx = sim_neg.min(dim=1)

    # Rank anchor patches by positive-match confidence; pick top-k.
    topk = torch.argsort(best_pos_cos, descending=True)[:k_examples].tolist()
    return [
        PatchMatch(
            anchor_patch=int(i),
            positive_patch=int(best_pos_idx[i]),
            negative_patch=int(worst_neg_idx[i]),
            positive_cos=float(best_pos_cos[i]),
            negative_cos=float(worst_neg_cos[i]),
        )
        for i in topk
    ]


def render_matches(
    anchor_pil: Image.Image,
    positive_pil: Image.Image,
    negative_pil: Image.Image,
    matches: list[PatchMatch],
    *,
    side: int,
    image_size: int,
    out_dir: Path,
) -> list[Path]:
    """Draw each match into its own PNG under ``out_dir``."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette_red = (220, 40, 40, 230)
    palette_green = (40, 180, 60, 230)
    palette_blue = (30, 90, 180, 230)

    outputs: list[Path] = []
    for i, m in enumerate(matches):
        anc = anchor_pil.convert("RGBA").copy()
        pos = positive_pil.convert("RGBA").copy()
        neg = negative_pil.convert("RGBA").copy()
        ImageDraw.Draw(anc).rectangle(
            _patch_box(side, m.anchor_patch, image_size), outline=palette_blue, width=3)
        ImageDraw.Draw(pos).rectangle(
            _patch_box(side, m.positive_patch, image_size), outline=palette_green, width=3)
        ImageDraw.Draw(neg).rectangle(
            _patch_box(side, m.negative_patch, image_size), outline=palette_red, width=3)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)
        for ax, img, title in zip(axes, (anc, pos, neg),
                                  ("anchor", f"positive  cos={m.positive_cos:.3f}",
                                   f"negative  cos={m.negative_cos:.3f}")):
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.set_axis_off()

        out_path = out_dir / f"patch_neighbours_sample{i}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


__all__ = [
    "PatchMatch",
    "find_patch_matches",
    "render_matches",
]

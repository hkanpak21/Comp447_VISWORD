"""Phase-2 with the title region of every anchor + pool image blanked.

Used to test the H-OCR hypothesis (RQ2): CLIP's high Phase-2 R@1 may be
driven by reading the rendered page title via its glyph-recognition
ability. Blanking the top 15 % of each image (where titles render in
wiki-ss screenshots) and re-running Phase-2 measures the contribution
of that text region.

Predictions:
  * H-OCR true: CLIP's R@1 drops substantially after blanking; DINOv2 /
    image-only encoders drop only modestly.
  * H-OCR false: all encoders drop by the same amount (the blanked
    region carried roughly equal information for everyone).

This module reuses :mod:`visword.eval_phase2` for everything except the
image-loading step, which is replaced by a wrapper that blanks the top
``blank_top_frac`` fraction of pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import yaml
from PIL import Image, ImageDraw

from visword.config import Config
from visword.data.light_dataset import default_transform
from visword.eval_phase2 import (
    _compute_triplet_recall,
    load_val_triplets,
)


def _blank_top(img: Image.Image, frac: float) -> Image.Image:
    """Return a copy of ``img`` with the top ``frac`` fraction of rows
    painted white. ``frac=0.0`` is a no-op."""
    if frac <= 0.0:
        return img
    out = img.copy()
    w, h = out.size
    cutoff = int(round(h * frac))
    if cutoff <= 0:
        return out
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, w, cutoff), fill=(255, 255, 255))
    return out


@torch.no_grad()
def _encode_images_blanked(
    model: torch.nn.Module,
    image_paths: Iterable[Path],
    *,
    transform,
    target_size: int,
    device: torch.device,
    blank_top_frac: float,
    batch_size: int = 16,
) -> torch.Tensor:
    paths = list(image_paths)
    tensors: list[torch.Tensor] = []
    for path in paths:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
            im = _blank_top(im, blank_top_frac)
            tensors.append(transform(im))
    if not tensors:
        return torch.empty(0, 0)
    out: list[torch.Tensor] = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        out.append(model(batch).cpu())
    return torch.cat(out, dim=0)


def phase2_recall_blanked(
    model: torch.nn.Module,
    anchors_cache_dir: Path,
    *,
    target_size: int,
    k_values: list[int],
    blank_top_frac: float = 0.15,
    max_triplets: int | None = None,
    device: torch.device | str | None = None,
) -> dict:
    """Phase-2 with title-region blanking. Mirrors ``phase2_recall`` but
    applies ``_blank_top(frac)`` to every image before encoding."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    triplets = load_val_triplets(anchors_cache_dir)
    if max_triplets is not None:
        triplets = triplets[:max_triplets]

    images_root = Path(anchors_cache_dir) / "images"
    transform = default_transform()

    per_k_hits: dict[str, list[float]] = {str(k): [] for k in k_values}
    same_sims: list[float] = []
    diff_sims: list[float] = []
    skipped = 0

    for triplet in triplets:
        anchor_path = images_root / triplet["anchor"]
        positives = [images_root / p for p in triplet.get("positives", [])
                     if (images_root / p).exists()]
        negatives = [images_root / n for n in triplet.get("negatives", [])
                     if (images_root / n).exists()]
        if not anchor_path.exists() or not positives or not negatives:
            skipped += 1
            continue

        anchor_embed = _encode_images_blanked(
            model, [anchor_path],
            transform=transform, target_size=target_size,
            device=device, blank_top_frac=blank_top_frac)
        pool_embeds = _encode_images_blanked(
            model, positives + negatives,
            transform=transform, target_size=target_size,
            device=device, blank_top_frac=blank_top_frac)
        recall_per, (pos_mean, neg_mean) = _compute_triplet_recall(
            anchor_embed, pool_embeds, len(positives), k_values)
        for k_str, hit in recall_per.items():
            per_k_hits[k_str].append(hit)
        same_sims.append(pos_mean)
        diff_sims.append(neg_mean)

    n_eligible = len(per_k_hits.get(str(k_values[0]), []))
    return {
        "blank_top_frac": blank_top_frac,
        "num_anchors_total": len(triplets),
        "num_anchors_evaluated": n_eligible,
        "num_skipped_missing_files": skipped,
        "recall": {k: (sum(per_k_hits[k]) / max(1, len(per_k_hits[k])))
                   for k in per_k_hits},
        "sanity": {
            "pos_sim_mean": (sum(same_sims) / max(1, len(same_sims))) if same_sims else float("nan"),
            "neg_sim_mean": (sum(diff_sims) / max(1, len(diff_sims))) if diff_sims else float("nan"),
            "gap": ((sum(same_sims) / len(same_sims)) - (sum(diff_sims) / len(diff_sims)))
                   if same_sims and diff_sims else 0.0,
        },
    }


from visword.eval_phase1 import _build_model_from_cfg, _load_checkpoint, _load_cfg


def run_titleblanked_cli(run_dir: Path, *, blank_top_frac: float,
                         checkpoint: str = "best_phase1.pt",
                         max_triplets: int | None = None) -> dict:
    run_dir = run_dir.resolve()
    cfg = _load_cfg(run_dir)
    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model_from_cfg(cfg)
    blob = _load_checkpoint(model, ckpt_path, device)

    payload = phase2_recall_blanked(
        model, Path(cfg.data.anchors_cache_dir),
        target_size=cfg.cropper.target_size,
        k_values=cfg.eval.k_values,
        blank_top_frac=blank_top_frac,
        max_triplets=max_triplets,
        device=device,
    )
    out_path = run_dir / f"phase2_titleblanked_{int(blank_top_frac * 100):02d}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default="best_phase1.pt")
    p.add_argument("--blank-top-frac", type=float, default=0.15,
                   help="fraction of pixel-rows from the top to paint white")
    p.add_argument("--max-triplets", type=int, default=None)
    args = p.parse_args(argv)
    payload = run_titleblanked_cli(
        args.run_dir, blank_top_frac=args.blank_top_frac,
        checkpoint=args.checkpoint, max_triplets=args.max_triplets)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

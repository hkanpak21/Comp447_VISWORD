"""Phase-2 recall: scroll-based anchor/positives/negatives retrieval (§7).

Operates on the prefetched anchors cache::

    data/wiki_ss_anchors/
    ├── images/<title>_<scroll_idx>.jpg
    ├── metadata.jsonl
    ├── triplets_train.jsonl
    └── triplets_val.jsonl

Each val triplet is ``{"anchor": "<img>", "positives": [...], "negatives": [...]}``.

Protocol (per triplet):
  * Encode the anchor and every image in ``positives ∪ negatives``.
  * Cosine-rank the pool against the anchor (the anchor itself is not in
    the pool).
  * "Recalled at K" iff at least one positive is in the top K.
  * Recall@K = mean over triplets.

Also emits the sanity block (same-page/diff-page sim means, gap,
monotonicity) for parity with Phase-1 output and to catch broken models
loudly (CONTEXT.md session 2 lesson).

**Decoupling.** Does not import ``visword.train`` (TESTS.md D3).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import yaml
from PIL import Image

from visword.config import Config
from visword.data.light_dataset import default_transform
from visword.eval_phase1 import _build_model_from_cfg, _load_checkpoint, _load_cfg


# ---------------------------------------------------------------------------
# Triplet loading
# ---------------------------------------------------------------------------

def load_val_triplets(anchors_cache_dir: Path) -> list[dict]:
    path = Path(anchors_cache_dir) / "triplets_val.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no triplets_val.jsonl at {path}. "
            f"Run scripts/prefetch_data.py --dataset wiki-ss-anchors."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Encoding / recall
# ---------------------------------------------------------------------------

def _encode_images(
    model: torch.nn.Module,
    image_paths: Iterable[Path],
    *,
    transform,
    target_size: int,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Load, resize to ``(target_size, target_size)``, stack, encode in batches."""
    paths = list(image_paths)
    tensors: list[torch.Tensor] = []
    for path in paths:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
            tensors.append(transform(im))

    if not tensors:
        return torch.empty(0, 0)

    out: list[torch.Tensor] = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        with torch.no_grad():
            out.append(model(batch).cpu())
    return torch.cat(out, dim=0)


def _compute_triplet_recall(
    anchor_embed: torch.Tensor,       # (1, D)
    pool_embeds: torch.Tensor,        # (P, D)
    num_positives: int,               # first P[:num_positives] are positives
    k_values: list[int],
) -> tuple[dict[str, float], tuple[float, float]]:
    """Return per-triplet recall@K and (same_sim_mean, diff_sim_mean) for sanity."""
    sim = (anchor_embed @ pool_embeds.T).squeeze(0)      # (P,)
    order = torch.argsort(sim, descending=True)
    pos_idx_set = set(range(num_positives))

    recall: dict[str, float] = {}
    for k in k_values:
        topk = order[:k].tolist()
        recall[str(k)] = 1.0 if any(i in pos_idx_set for i in topk) else 0.0

    pos_sim_mean = float(sim[:num_positives].mean()) if num_positives > 0 else float("nan")
    neg_sim_mean = float(sim[num_positives:].mean()) if sim.shape[0] > num_positives else float("nan")
    return recall, (pos_sim_mean, neg_sim_mean)


def phase2_recall(
    model: torch.nn.Module,
    anchors_cache_dir: Path,
    *,
    target_size: int,
    k_values: list[int],
    max_triplets: int | None = None,
    device: torch.device | str | None = None,
) -> dict:
    """Run the §7 Phase-2 protocol over ``triplets_val.jsonl``."""
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
        # Require the anchor and at least one positive + one negative on disk
        # (a triplet missing one member is no longer a well-formed retrieval).
        if not anchor_path.exists() or not positives or not negatives:
            skipped += 1
            continue

        pool_paths = positives + negatives
        anchor_embed = _encode_images(
            model, [anchor_path],
            transform=transform, target_size=target_size, device=device,
        )
        pool_embeds = _encode_images(
            model, pool_paths,
            transform=transform, target_size=target_size, device=device,
        )
        recall, (p_sim, n_sim) = _compute_triplet_recall(
            anchor_embed, pool_embeds, len(positives), k_values,
        )
        for k, hit in recall.items():
            per_k_hits[k].append(hit)
        if p_sim == p_sim:   # not nan
            same_sims.append(p_sim)
        if n_sim == n_sim:
            diff_sims.append(n_sim)

    recall_mean = {k: (sum(v) / len(v) if v else 0.0) for k, v in per_k_hits.items()}
    same_mean = sum(same_sims) / len(same_sims) if same_sims else float("nan")
    diff_mean = sum(diff_sims) / len(diff_sims) if diff_sims else float("nan")
    monotonic = all(recall_mean[str(a)] <= recall_mean[str(b)] + 1e-6
                    for a, b in zip(sorted(k_values), sorted(k_values)[1:]))

    total_pool = sum(len(t.get("positives", [])) + len(t.get("negatives", [])) for t in triplets)
    return {
        "num_triplets": len(per_k_hits[str(k_values[0])]),
        "num_triplets_skipped": skipped,
        "num_anchors": len(triplets),
        "num_pool_images": total_pool,
        "recall": recall_mean,
        "sanity": {
            "same_page_sim_mean": same_mean,
            "diff_page_sim_mean": diff_mean,
            "gap": same_mean - diff_mean if (same_mean == same_mean and diff_mean == diff_mean) else 0.0,
            "monotonic": bool(monotonic),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_phase2_cli(
    run_dir: Path,
    *,
    checkpoint: str = "best_phase1.pt",
    max_triplets: int | None = None,
) -> dict:
    run_dir = run_dir.resolve()
    cfg = _load_cfg(run_dir)
    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}. Run training first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model_from_cfg(cfg)
    blob = _load_checkpoint(model, ckpt_path, device)

    result = phase2_recall(
        model,
        Path(cfg.data.anchors_cache_dir),
        target_size=cfg.cropper.target_size,
        k_values=cfg.eval.k_values,
        max_triplets=max_triplets,
        device=device,
    )
    payload = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": blob.get("step"),
        **result,
    }
    (run_dir / "phase2_recall.json").write_text(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default="best_phase1.pt",
                   help="filename under <run-dir>/checkpoints/")
    p.add_argument("--max-triplets", type=int, default=None,
                   help="cap for smoke tests; default = all val triplets")
    args = p.parse_args(argv)
    payload = run_phase2_cli(
        args.run_dir, checkpoint=args.checkpoint, max_triplets=args.max_triplets,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

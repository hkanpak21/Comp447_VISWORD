"""Phase-1 recall: same-page non-overlapping-crop retrieval (PROJECT_SPEC.md §7).

Two entry points:

* ``phase1_recall(model, dataset, k_values)`` — library API called
  mid-training by ``visword.train`` (Phase C, eval_every cadence).
* ``main()`` — CLI that loads a checkpoint from a run dir, reconstructs
  the same eval split, runs the full protocol, and writes
  ``phase1_recall.json`` into that run dir per the §7 schema.

**Decoupling.** This module must not import ``visword.train`` (TESTS.md
D3). Keep the run-dir / dataset / model plumbing local.

Protocol:
  * For each page in the eval set, generate all non-overlapping crops
    (``min_text_ratio=0`` so every tile counts).
  * Encode every crop into the model's descriptor space.
  * For each crop as a query, rank every *other* crop by cosine
    similarity; recall@K is the fraction of queries with at least one
    same-page crop in the top K (only queries that have a same-page
    partner are counted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from visword.config import Config
from visword.data import manifest as M
from visword.data.cropper import NonOverlappingCropper
from visword.data.light_dataset import LightWikiScreenshotDataset


@torch.no_grad()
def _encode_all_crops(
    model: torch.nn.Module,
    dataset: LightWikiScreenshotDataset,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every crop in ``dataset`` (using ``iter_all_crops``).

    Returns:
        embeddings: ``(N_crops, D)`` L2-normed (the model already normalises).
        page_ids:   ``(N_crops,)`` page index (local to ``dataset.rows``).
    """
    model.eval()
    embs: list[torch.Tensor] = []
    page_ids: list[int] = []

    for local_idx, crops in dataset.iter_all_crops():
        if not crops:
            continue
        x = torch.stack(crops).to(device)
        z = model(x)
        embs.append(z.detach().cpu())
        page_ids.extend([local_idx] * x.shape[0])

    if not embs:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    return torch.cat(embs, dim=0), torch.as_tensor(page_ids, dtype=torch.long)


def compute_recall_at_k(
    embeddings: torch.Tensor,
    page_ids: torch.Tensor,
    k_values: list[int],
) -> dict[str, float | dict | int]:
    """Page-level recall@K for same-page retrieval.

    A query is "correct at K" iff any of its top-K non-self neighbours
    share its page id.
    """
    if embeddings.numel() == 0:
        return {
            "num_crops": 0,
            "num_pages": 0,
            "recall": {str(k): 0.0 for k in k_values},
            "sanity": {
                "same_page_sim_mean": 0.0,
                "diff_page_sim_mean": 0.0,
                "gap": 0.0,
                "monotonic": True,
            },
        }

    sim = embeddings @ embeddings.T
    n = sim.shape[0]
    sim.fill_diagonal_(float("-inf"))  # never retrieve self

    same_mask = page_ids.unsqueeze(0) == page_ids.unsqueeze(1)
    diff_mask = ~same_mask & ~torch.eye(n, dtype=torch.bool)

    # Sanity stats
    same_vals = sim[same_mask & ~torch.eye(n, dtype=torch.bool)]
    diff_vals = sim[diff_mask]
    same_pos = same_vals[torch.isfinite(same_vals)]
    same_mean = float(same_pos.mean()) if same_pos.numel() > 0 else float("nan")
    diff_mean = float(diff_vals.mean()) if diff_vals.numel() > 0 else float("nan")

    recall: dict[str, float] = {}
    for k in sorted(k_values):
        # Clamp k when the pool is smaller than k (common in debug eval splits).
        # In that regime recall@k effectively equals recall@(n-1).
        k_eff = min(k, max(1, n - 1))
        topk_idx = sim.topk(k_eff, dim=1).indices             # (N, k_eff)
        topk_same = same_mask.gather(1, topk_idx)
        # Only count queries that have at least one non-self same-page neighbour.
        eligible = (same_mask & ~torch.eye(n, dtype=torch.bool)).any(dim=1)
        if eligible.any():
            recall[str(k)] = float(topk_same[eligible].any(dim=1).float().mean())
        else:
            recall[str(k)] = 0.0

    monotonic = all(recall[str(a)] <= recall[str(b)] + 1e-6
                    for a, b in zip(sorted(k_values), sorted(k_values)[1:]))

    return {
        "num_crops": n,
        "num_pages": int(page_ids.unique().numel()),
        "recall": recall,
        "sanity": {
            "same_page_sim_mean": same_mean,
            "diff_page_sim_mean": diff_mean,
            "gap": same_mean - diff_mean if (same_mean == same_mean and diff_mean == diff_mean) else 0.0,
            "monotonic": bool(monotonic),
        },
    }


def phase1_recall(
    model: torch.nn.Module,
    dataset: LightWikiScreenshotDataset,
    *,
    k_values: list[int],
    device: torch.device | str | None = None,
) -> dict:
    """One-shot Phase-1 recall (used by ``visword.train`` for eval_every)."""
    if device is None:
        device = next(model.parameters()).device
    embs, page_ids = _encode_all_crops(model, dataset, device)
    return compute_recall_at_k(embs, page_ids, k_values)


# ---------------------------------------------------------------------------
# CLI (Phase D)
# ---------------------------------------------------------------------------


def _load_cfg(run_dir: Path) -> Config:
    cfg_path = run_dir / "config.resolved.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path} — did training finish?")
    return Config.model_validate(yaml.safe_load(cfg_path.read_text()))


def _rebuild_eval_dataset(cfg: Config) -> LightWikiScreenshotDataset:
    """Rebuild the exact same eval split train.py used.

    ``train.py`` picks ``num_train_samples`` + ``num_eval_samples`` rows from a
    ``numpy.Generator(seed)``'s permutation; we replay that here. Must stay
    in sync with ``visword.train._split_indices``.
    """
    cache_dir = Path(cfg.data.wiki_ss_cache_dir)
    manifest = M.read_manifest(cache_dir)
    num_rows = manifest["num_rows"]
    n_train, n_eval = cfg.data.num_train_samples, cfg.data.num_eval_samples
    if n_train + n_eval > num_rows:
        raise SystemExit(
            f"cache has {num_rows} rows but config wants "
            f"{n_train}+{n_eval} — re-prefetch or adjust config"
        )
    perm = np.random.default_rng(cfg.train.seed).permutation(num_rows)
    eval_idx = perm[n_train : n_train + n_eval].tolist()

    eval_cropper = NonOverlappingCropper(
        crop_size=cfg.cropper.crop_size,
        overlap=cfg.cropper.overlap,
        min_text_ratio=0.0,                           # keep every tile for eval
        target_size=cfg.cropper.target_size,
    )
    return LightWikiScreenshotDataset(
        cache_dir,
        indices=eval_idx,
        cropper=eval_cropper,
        k_per_page=2,
        seed=cfg.train.seed + 1,
    )


def _build_model_from_cfg(cfg: Config) -> torch.nn.Module:
    """Local model factory — avoids importing visword.train (TESTS.md D3)."""
    if cfg.model_kind == "salad":
        from visword.models.dinov2_salad import DINOv2SALAD
        return DINOv2SALAD(cfg)
    if cfg.model_kind == "linear_probe":
        from visword.models.zeroshot import DINOv2LinearProbe
        return DINOv2LinearProbe(cfg)
    if cfg.model_kind == "clip_salad":
        from visword.models.clip_salad import CLIPSALAD
        return CLIPSALAD(cfg)
    if cfg.model_kind == "clip_cls":
        from visword.models.clip_salad import CLIPCLS
        return CLIPCLS(cfg)
    from visword.models.dinov2_cls import DINOv2CLS
    return DINOv2CLS(cfg)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> dict:
    """Load model state_dict from a train.py-style checkpoint."""
    blob = torch.load(ckpt_path, map_location=device)
    state = blob.get("model_state_dict", blob)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return blob


def run_phase1_cli(
    run_dir: Path, *, checkpoint: str = "best_phase1.pt",
) -> dict:
    """Execute the §7 Phase-1 recall protocol, write phase1_recall.json, return it."""
    run_dir = run_dir.resolve()
    cfg = _load_cfg(run_dir)
    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}. Run training first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_ds = _rebuild_eval_dataset(cfg)
    model = _build_model_from_cfg(cfg)
    blob = _load_checkpoint(model, ckpt_path, device)

    result = phase1_recall(model, eval_ds, k_values=cfg.eval.k_values, device=device)
    payload = {
        "checkpoint": str(ckpt_path.relative_to(run_dir.parent.parent)
                          if run_dir.parent.parent in ckpt_path.parents
                          else ckpt_path),
        "checkpoint_step": blob.get("step"),
        "num_pages_evaluated": result["num_pages"],
        "num_crops": result["num_crops"],
        "recall": result["recall"],
        "sanity": result["sanity"],
    }
    out_path = run_dir / "phase1_recall.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default="best_phase1.pt",
                   help="filename under <run-dir>/checkpoints/")
    args = p.parse_args(argv)
    payload = run_phase1_cli(args.run_dir, checkpoint=args.checkpoint)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

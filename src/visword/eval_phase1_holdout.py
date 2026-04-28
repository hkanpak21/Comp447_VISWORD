"""Protocol A — leak-free crop → page retrieval.

Replaces the original Phase-1 protocol (crop-vs-crop nearest neighbour
on an N×N matrix where same-page positives are trivially adjacent) with
a *page-level* retrieval where:

  * Gallery = one descriptor per eval page = L2-normed mean of that
    page's crop embeddings.
  * For the query crop's own source page, the gallery vector is
    recomputed *excluding the query crop itself* (leave-one-crop-out
    aggregation).  This forbids the trivial "find another column of the
    same Wikipedia page" matches that inflated the old metric.
  * Queries restricted to crops with ``min_text_ratio >= 0.05`` to
    match the training distribution (blank/margin crops were never
    trained on; the model has no obligation to map them anywhere
    sensible).

Random-baseline R@1 on a P-page gallery is 1/P (≈ 0.0005 at P=2000),
i.e. a real retrieval problem.

Two entry points (mirrors ``eval_phase1.py``):

* ``protocol_a_recall(model, dataset, k_values)`` — library API.
* ``main()`` — CLI that loads a checkpoint from a run dir and writes
  ``phase1_holdout.json`` into that run dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def compute_protocol_a_recall(
    embeddings: torch.Tensor,         # (N, D), L2-normed
    page_ids: torch.Tensor,           # (N,)
    k_values: list[int],
) -> dict[str, float | dict | int]:
    """Page-level retrieval with leave-one-out same-page gallery aggregation.

    For each query crop c on page p:
      * Compute gallery_q = L2-norm( sum of all crops on page q )  for q != p
      * Compute gallery_p_loo = L2-norm( sum of crops on page p, minus c )
      * Cosine-rank gallery vectors against c; recall@k = correct page in top-k.

    Queries on pages with only one crop are excluded (LOO undefined).
    """
    N = int(embeddings.shape[0])
    if N == 0:
        return {
            "num_crops": 0,
            "num_pages": 0,
            "num_queries_eligible": 0,
            "recall": {str(k): 0.0 for k in k_values},
            "sanity": {
                "same_page_sim_mean": 0.0,
                "diff_page_sim_mean": 0.0,
                "gap": 0.0,
            },
        }

    D = int(embeddings.shape[1])
    # Compact page-id space: remap to contiguous 0..P-1 because ``iter_all_crops``
    # may emit non-contiguous local indices if some pages produce zero crops.
    unique_pages, compact = torch.unique(page_ids, sorted=True, return_inverse=True)
    P = int(unique_pages.numel())

    sums = torch.zeros(P, D, dtype=embeddings.dtype)
    counts = torch.zeros(P, dtype=torch.long)
    sums.index_add_(0, compact, embeddings)
    counts.index_add_(0, compact, torch.ones_like(compact))

    # Default gallery: L2-norm of the per-page sum.
    gallery = F.normalize(sums, dim=1, eps=1e-12)              # (P, D)

    # Base similarity matrix: every crop vs every default gallery entry.
    sim = embeddings @ gallery.T                                # (N, P)

    # LOO replacement for the query's own source page.
    eligible = counts[compact] >= 2                             # (N,)
    if eligible.any():
        elig_idx = torch.nonzero(eligible, as_tuple=True)[0]
        for i in elig_idx.tolist():
            pid = int(compact[i].item())
            loo_unnorm = sums[pid] - embeddings[i]
            loo = F.normalize(loo_unnorm, dim=0, eps=1e-12)
            sim[i, pid] = float(embeddings[i] @ loo)

    # Sanity: same-vs-diff page mean similarity (under the LOO regime).
    rows = torch.arange(N)
    same_sim = sim[rows, compact]                               # (N,)
    # diff = mean over all P columns, excluding the same-page column
    sim_sum = sim.sum(dim=1)
    diff_sim = (sim_sum - same_sim) / max(1, P - 1)
    same_mean = float(same_sim[eligible].mean()) if eligible.any() else float("nan")
    diff_mean = float(diff_sim[eligible].mean()) if eligible.any() else float("nan")

    # Recall at K: correct page in top-K columns.
    recall: dict[str, float] = {}
    for k in sorted(k_values):
        k_eff = min(k, P)
        topk_idx = sim.topk(k_eff, dim=1).indices               # (N, k_eff)
        correct = (topk_idx == compact.unsqueeze(1)).any(dim=1)
        if eligible.any():
            recall[str(k)] = float(correct[eligible].float().mean())
        else:
            recall[str(k)] = 0.0

    return {
        "num_crops": N,
        "num_pages": P,
        "num_queries_eligible": int(eligible.sum().item()),
        "recall": recall,
        "sanity": {
            "same_page_sim_mean": same_mean,
            "diff_page_sim_mean": diff_mean,
            "gap": (same_mean - diff_mean) if same_mean == same_mean and diff_mean == diff_mean else 0.0,
        },
    }


def protocol_a_recall(
    model: torch.nn.Module,
    dataset: LightWikiScreenshotDataset,
    *,
    k_values: list[int],
    device: torch.device | str | None = None,
) -> dict:
    """One-shot Protocol-A recall (library API parallel to ``phase1_recall``)."""
    if device is None:
        device = next(model.parameters()).device
    embs, page_ids = _encode_all_crops(model, dataset, device)
    return compute_protocol_a_recall(embs, page_ids, k_values)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_cfg(run_dir: Path) -> Config:
    cfg_path = run_dir / "config.resolved.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path} — did training finish?")
    return Config.model_validate(yaml.safe_load(cfg_path.read_text()))


def _rebuild_eval_dataset(cfg: Config, *, min_text_ratio: float = 0.05,
                          blank_page_top_frac: float = 0.0) -> LightWikiScreenshotDataset:
    """Same eval split as ``eval_phase1`` but with the in-distribution
    ``min_text_ratio`` filter so blank crops aren't queried.

    If ``blank_page_top_frac > 0`` the loader paints the top fraction
    of every page white before cropping; this is the trained-model
    H-OCR ablation."""
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
        min_text_ratio=min_text_ratio,
        target_size=cfg.cropper.target_size,
    )
    ds = LightWikiScreenshotDataset(
        cache_dir,
        indices=eval_idx,
        cropper=eval_cropper,
        k_per_page=2,
        seed=cfg.train.seed + 1,
    )
    # If requested, monkey-patch the dataset's image-loader to blank the
    # top of each page before cropping. This avoids touching
    # LightWikiScreenshotDataset internals.
    if blank_page_top_frac > 0.0:
        from PIL import Image, ImageDraw

        def _load_with_blanking(row_idx: int):
            row = ds.rows[row_idx]
            with Image.open(cache_dir / row["image_path"]) as im:
                im = im.convert("RGB")
                im.load()
                w, h = im.size
                cutoff = int(round(h * blank_page_top_frac))
                if cutoff > 0:
                    im = im.copy()
                    ImageDraw.Draw(im).rectangle(
                        (0, 0, w, cutoff), fill=(255, 255, 255))
                return ds.cropper(im)
        ds._load_and_crop = _load_with_blanking  # type: ignore[attr-defined]
    return ds


def _build_model_from_cfg(cfg: Config) -> torch.nn.Module:
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
    blob = torch.load(ckpt_path, map_location=device)
    state = blob.get("model_state_dict", blob)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return blob


def run_protocol_a_cli(
    run_dir: Path, *, checkpoint: str = "best_phase1.pt",
    blank_page_top_frac: float = 0.0,
    min_text_ratio: float = 0.05,
    save_descriptors: Path | None = None,
) -> dict:
    run_dir = run_dir.resolve()
    cfg = _load_cfg(run_dir)
    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}. Run training first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_ds = _rebuild_eval_dataset(cfg, min_text_ratio=min_text_ratio,
                                     blank_page_top_frac=blank_page_top_frac)
    model = _build_model_from_cfg(cfg)
    blob = _load_checkpoint(model, ckpt_path, device)

    if save_descriptors is not None:
        embs, page_ids = _encode_all_crops(model, eval_ds, device)
        save_descriptors.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_descriptors, emb=embs.cpu().numpy(),
                 page_ids=page_ids.cpu().numpy())
        result = compute_protocol_a_recall(embs, page_ids, cfg.eval.k_values)
    else:
        result = protocol_a_recall(model, eval_ds, k_values=cfg.eval.k_values, device=device)
    payload = {
        "protocol": "A_holdout_page_mean",
        "checkpoint": str(ckpt_path),
        "checkpoint_step": blob.get("step"),
        "num_pages_evaluated": result["num_pages"],
        "num_crops": result["num_crops"],
        "num_queries_eligible": result["num_queries_eligible"],
        "recall": result["recall"],
        "sanity": result["sanity"],
        "min_text_ratio_for_query": 0.05,
        "blank_page_top_frac": blank_page_top_frac,
    }
    if blank_page_top_frac > 0.0:
        suffix = f"_blank{int(blank_page_top_frac * 100):02d}"
    else:
        suffix = ""
    out_path = run_dir / f"phase1_holdout{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default="best_phase1.pt",
                   help="filename under <run-dir>/checkpoints/")
    p.add_argument("--blank-page-top-frac", type=float, default=0.0,
                   help="If >0, paint the top fraction of every page "
                        "white before cropping (H-OCR ablation).")
    p.add_argument("--min-text-ratio", type=float, default=0.05,
                   help="Cropper filter; lower to keep all grid cells "
                        "(needed for matched orig/blank descriptor pairs "
                        "in LEACE).")
    p.add_argument("--save-descriptors", type=Path, default=None,
                   help="Dump (emb, page_ids) as .npz for downstream "
                        "LEACE / probe analyses.")
    args = p.parse_args(argv)
    payload = run_protocol_a_cli(args.run_dir, checkpoint=args.checkpoint,
                                 blank_page_top_frac=args.blank_page_top_frac,
                                 min_text_ratio=args.min_text_ratio,
                                 save_descriptors=args.save_descriptors)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

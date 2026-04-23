"""One-shot interpretability driver (PROJECT_SPEC.md §8 / §10.3).

Given a training run directory, loads the best checkpoint and writes
every §8 artefact into ``<run-dir>/interpret/``:

* ``attention_sample{0..k-1}.png`` + JSON sidecars
* ``salad_clusters_sample{0..k-1}.png``      (top cluster's OT mass)
* ``dustbin_map_sample{0..k-1}.png``         (per-patch dustbin mass)
* ``patch_neighbours_sample{0..k-1}.png``
* ``cls_vs_vlad.png`` + ``cls_vs_vlad.json``  (aggregate over eval set)
* ``dustbin_evolution.png``                  (read from metrics.jsonl)
* ``salad_hooks.json``                       (resolved submodule names)

All heavy work is device-aware (uses CUDA if available). Samples are
drawn deterministically from the eval split so repeated runs of the
interpret job produce identical artefacts.

CLI::

    python -m visword.interpret --run-dir runs/<id> [--k 4] [--checkpoint best_phase1.pt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from visword.config import Config
from visword.data.cropper import NonOverlappingCropper
from visword.data.light_dataset import LightWikiScreenshotDataset, default_transform
from visword.eval_phase1 import _build_model_from_cfg, _load_checkpoint, _load_cfg
from visword.interpret.attention import compute_cls_to_patch_attention, render_overlay
from visword.interpret.cls_vs_vlad import decompose, plot_cls_vs_vlad, write_report
from visword.interpret.dustbin import plot_dustbin_evolution
from visword.interpret.patch_neighbours import find_patch_matches, render_matches
from visword.interpret.salad_internals import (
    capture_score_tensor,
    discover_salad_submodules,
    render_cluster_heatmap,
    save_hooks_json,
    sinkhorn_assignment,
)


def _rebuild_eval_dataset(cfg: Config) -> LightWikiScreenshotDataset:
    """Deterministic eval-split reconstruction (matches eval_phase1 + train.py)."""
    import numpy as np
    from visword.data import manifest as M

    manifest = M.read_manifest(Path(cfg.data.wiki_ss_cache_dir))
    perm = np.random.default_rng(cfg.train.seed).permutation(manifest["num_rows"])
    eval_idx = perm[cfg.data.num_train_samples : cfg.data.num_train_samples + cfg.data.num_eval_samples].tolist()
    cropper = NonOverlappingCropper(
        crop_size=cfg.cropper.crop_size,
        overlap=cfg.cropper.overlap,
        min_text_ratio=0.0,
        target_size=cfg.cropper.target_size,
    )
    return LightWikiScreenshotDataset(
        Path(cfg.data.wiki_ss_cache_dir),
        indices=eval_idx,
        cropper=cropper,
        k_per_page=2,
        seed=cfg.train.seed + 1,
    )


def _load_crop(eval_ds: LightWikiScreenshotDataset, row_idx: int):
    """Return the first crop of row ``row_idx`` as (PIL image, model input tensor)."""
    from PIL import Image

    cache_dir = Path(eval_ds.cache_dir)
    row = eval_ds.rows[row_idx]
    with Image.open(cache_dir / row["image_path"]) as im:
        im = im.convert("RGB")
        crops = eval_ds.cropper(im)
    if not crops:
        return None, None
    pil = crops[0]
    return pil, eval_ds.transform(pil)


def _save_attention(backbone, eval_ds, k, interpret_dir, device):
    """Write attention_sample{0..k-1}.png + JSON sidecars."""
    pairs = []
    for i in range(min(k, len(eval_ds))):
        pil, tensor = _load_crop(eval_ds, i)
        if pil is None:
            continue
        pairs.append((i, pil, tensor))
    if not pairs:
        return

    batch = torch.stack([t for _, _, t in pairs]).to(device)
    captures = compute_cls_to_patch_attention(backbone, batch)
    for (i, pil, _), cap in zip(pairs, captures):
        render_overlay(pil, cap, interpret_dir / f"attention_sample{i}.png")


def _save_salad_heatmaps(aggregator, backbone, cfg: Config, eval_ds, k, interpret_dir, device):
    """Write salad_clusters_sample*.png + dustbin_map_sample*.png using the Sinkhorn assignment."""
    hooks = discover_salad_submodules(aggregator, num_channels=cfg.backbone.feature_dim)
    save_hooks_json(hooks, interpret_dir.parent)     # interpret/salad_hooks.json

    target_hw = cfg.cropper.target_size // 14

    for i in range(min(k, len(eval_ds))):
        pil, tensor = _load_crop(eval_ds, i)
        if pil is None:
            continue
        batch = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            patches, cls = backbone(batch)
        score = capture_score_tensor(aggregator, patches, cls, hooks.score)
        assignment = sinkhorn_assignment(score, aggregator.dust_bin)   # (1, m+1, H*W)

        # Pick the cluster with the largest total mass as the "top cluster" overlay.
        cluster_totals = assignment[0, :-1, :].sum(dim=-1)             # (m,)
        top_cluster = int(cluster_totals.argmax())

        render_cluster_heatmap(
            pil, assignment, side=target_hw, row=top_cluster,
            out_path=interpret_dir / f"salad_clusters_sample{i}.png",
            title=f"top cluster #{top_cluster}  mass={float(cluster_totals[top_cluster]):.2f}",
        )
        render_cluster_heatmap(
            pil, assignment, side=target_hw, row="dustbin",
            out_path=interpret_dir / f"dustbin_map_sample{i}.png",
            title=f"dustbin mass  frac={float(assignment[0, -1, :].sum() / assignment.sum()):.3f}",
        )


def _save_patch_neighbours(backbone, cfg: Config, eval_ds, k, interpret_dir, device):
    """Anchor/pos/neg patch-triplet views.

    Uses pairs of rows from the eval split: (2i, 2i+1) as anchor/positive from
    the same page (their index in ``eval_ds.rows`` is distinct so it's a pseudo
    positive; the anchors cache's actual positives would need the triplets
    file — kept simple here).
    """
    samples: list[tuple[int, object, object, object, object, object, object]] = []
    for j in range(0, min(2 * k, len(eval_ds) - 1), 2):
        pil_a, t_a = _load_crop(eval_ds, j)
        pil_p, t_p = _load_crop(eval_ds, (j + 1) % len(eval_ds))
        neg_idx = (j + 7) % len(eval_ds)     # anything far in the permutation
        pil_n, t_n = _load_crop(eval_ds, neg_idx)
        if None in (pil_a, pil_p, pil_n):
            continue
        samples.append((j, pil_a, t_a, pil_p, t_p, pil_n, t_n))

    for idx, pil_a, t_a, pil_p, t_p, pil_n, t_n in samples[:k]:
        matches = find_patch_matches(
            backbone, t_a.to(device), t_p.to(device), t_n.to(device),
            k_examples=1,                      # one per page-triplet is enough
        )
        # render_matches emits patch_neighbours_sample{0..k-1}.png; use a
        # per-sample subdirectory so we don't overwrite across triplets.
        side = cfg.cropper.target_size // 14
        render_matches(
            pil_a, pil_p, pil_n, matches,
            side=side, image_size=cfg.cropper.target_size,
            out_dir=interpret_dir / f"patch_triplet_{idx}",
        )


def _save_cls_vs_vlad(model, eval_ds, cfg: Config, interpret_dir, device):
    """Encode the eval set and write the CLS-vs-VLAD decomposition."""
    embs, page_ids = [], []
    with torch.no_grad():
        for local_idx, crops in eval_ds.iter_all_crops():
            if not crops:
                continue
            z = model(torch.stack(crops).to(device)).cpu()
            embs.append(z)
            page_ids.extend([local_idx] * z.shape[0])
    if not embs:
        return
    z = torch.cat(embs, dim=0)
    labels = torch.as_tensor(page_ids)
    result = decompose(cfg, z, labels)
    write_report(result, interpret_dir / "cls_vs_vlad.json")
    plot_cls_vs_vlad(result, interpret_dir / "cls_vs_vlad.png")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default="best_phase1.pt")
    p.add_argument("--k", type=int, default=4, help="samples per per-crop figure")
    args = p.parse_args(argv)

    run_dir = args.run_dir.resolve()
    cfg = _load_cfg(run_dir)
    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model_from_cfg(cfg)
    _load_checkpoint(model, ckpt_path, device)

    eval_ds = _rebuild_eval_dataset(cfg)
    interpret_dir = run_dir / "interpret"
    interpret_dir.mkdir(parents=True, exist_ok=True)

    # (1) attention heatmaps — only DINOv2-family backbones (others
    # have different internal layouts and the attention extractor would crash)
    if cfg.model_kind in {"salad", "cls", "linear_probe"}:
        try:
            _save_attention(model.backbone, eval_ds, args.k, interpret_dir, device)
        except Exception as exc:
            sys.stderr.write(f"interpret: attention skipped ({type(exc).__name__}: {exc})\n")

    # (2) SALAD cluster + dustbin maps. Only meaningful when the Sinkhorn-OT
    # forward is active AND the descriptor includes the VLAD branch — i.e.
    # the "full" ablation. token_only / softmax_assign skip Sinkhorn;
    # vlad_only has no token branch so the cls_vs_vlad split is also moot.
    salad_full = cfg.model_kind == "salad" and cfg.salad.ablation == "full"
    if salad_full:
        _save_salad_heatmaps(
            model.aggregator, model.backbone, cfg, eval_ds,
            args.k, interpret_dir, device,
        )

    # (3) patch neighbours — DINOv2-family only (uses backbone.model.blocks)
    if cfg.model_kind in {"salad", "cls", "linear_probe"}:
        try:
            _save_patch_neighbours(model.backbone, cfg, eval_ds, args.k, interpret_dir, device)
        except Exception as exc:
            sys.stderr.write(f"interpret: patch_neighbours skipped ({type(exc).__name__}: {exc})\n")

    # (4) CLS-vs-VLAD aggregate (only meaningful for full SALAD).
    if salad_full:
        _save_cls_vs_vlad(model, eval_ds, cfg, interpret_dir, device)

    # (5) dustbin evolution from metrics.jsonl
    plot_dustbin_evolution(run_dir / "metrics.jsonl", interpret_dir / "dustbin_evolution.png")

    sys.stdout.write(f"interpret artefacts written under {interpret_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

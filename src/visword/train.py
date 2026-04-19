"""Training entry point (PROJECT_SPEC.md §6).

CLI::

    python -m visword.train --config configs/salad_main.yaml \\
        [--set train.epochs=5 train.lr_head=1e-3] \\
        [--run-name "my-experiment"]

Contract (enforced at startup):
  * No downloads — ``HF_HUB_OFFLINE=1`` is set before any import of HF
    machinery; missing data cache aborts loudly pointing to
    ``scripts/prefetch_data.py``.
  * No side effects outside ``runs/<id>/``.
  * Every structured event goes to ``metrics.jsonl``; stdout is kept to
    a one-line progress heartbeat.
"""
from __future__ import annotations

# --- No downloads at import time ------------------------------------------
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# --------------------------------------------------------------------------

import argparse
import itertools
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from visword.config import Config, config_hash
from visword.data import manifest as M
from visword.data.cropper import NonOverlappingCropper
from visword.data.light_dataset import (
    LightWikiScreenshotDataset,
    multi_positive_collate,
)
from visword.diagnostics.batch_stats import (
    BatchStats,
    compute_batch_stats,
    write_report,
)
from visword.eval_phase1 import phase1_recall
from visword.losses import build_loss
from visword.paths import PROJECT_ROOT, expand_env
from visword.reporting.jsonl_logger import JsonlLogger, read_jsonl
from visword.reporting.plots import plot_train_curves
from visword.reporting.run_dir import create_run_dir
from visword.seed import seed_everything


# ---------------------------------------------------------------------------
# Config resolution (same algorithm as scripts/resolve_config.py)
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _apply_set(d: dict, assignment: str) -> None:
    if "=" not in assignment:
        raise SystemExit(f"--set expects key.path=value, got {assignment!r}")
    key_path, raw = assignment.split("=", 1)
    value = yaml.safe_load(raw)
    cursor: Any = d
    parts = key_path.split(".")
    for p in parts[:-1]:
        cursor = cursor.setdefault(p, {})
    cursor[parts[-1]] = value


def resolve_config(config_path: Path, overrides: list[str]) -> Config:
    default = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text()) or {}
    named = yaml.safe_load(Path(config_path).read_text()) or {}
    merged = _deep_merge(default, named)
    for ov in overrides or []:
        _apply_set(merged, ov)
    merged = expand_env(merged)
    return Config.model_validate(merged)


# ---------------------------------------------------------------------------
# Split the manifest deterministically into train / eval indices.
# ---------------------------------------------------------------------------


def _split_indices(
    num_rows: int, num_train: int, num_eval: int, seed: int
) -> tuple[list[int], list[int]]:
    if num_train + num_eval > num_rows:
        raise SystemExit(
            f"data.num_train_samples + data.num_eval_samples = {num_train + num_eval} "
            f"exceeds cache size {num_rows}. Re-run prefetch with a larger target."
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_rows)
    return perm[:num_train].tolist(), perm[num_train : num_train + num_eval].tolist()


# ---------------------------------------------------------------------------
# Model / loss / optimizer builders
# ---------------------------------------------------------------------------


def _build_model(cfg: Config) -> torch.nn.Module:
    if cfg.model_kind == "salad":
        from visword.models.dinov2_salad import DINOv2SALAD
        return DINOv2SALAD(cfg)
    from visword.models.dinov2_cls import DINOv2CLS
    return DINOv2CLS(cfg)


class _DustbinTracker:
    """Forward-hook wrapper that reports the SALAD dustbin mass per step.

    Registers once on the aggregator's ``score`` submodule (discovered by
    shape). On each forward we cache the score logits; ``last_value`` then
    runs the Sinkhorn step on demand (cheap — 3 iterations on a
    (B, num_clusters, H*W) tensor) and returns the dustbin fraction.
    """

    def __init__(self, model: torch.nn.Module, cfg: Config) -> None:
        from visword.interpret.salad_internals import (
            discover_salad_submodules,
            dustbin_mass_fraction,
            sinkhorn_assignment,
        )
        self._dustbin_fn = dustbin_mass_fraction
        self._sinkhorn_fn = sinkhorn_assignment

        aggregator = model.aggregator
        hooks = discover_salad_submodules(
            aggregator,
            num_channels=cfg.backbone.feature_dim,
        )
        target = dict(aggregator.named_modules())[hooks.score]
        self._aggregator = aggregator
        self._last_score: torch.Tensor | None = None

        def _capture(_m, _inp, out):
            self._last_score = out.detach()

        self._handle = target.register_forward_hook(_capture)

    @property
    def last_value(self) -> float | None:
        if self._last_score is None:
            return None
        assignment = self._sinkhorn_fn(self._last_score, self._aggregator.dust_bin)
        return self._dustbin_fn(assignment)

    def close(self) -> None:
        self._handle.remove()
        self._last_score = None


def _build_loss(cfg: Config) -> torch.nn.Module:
    if cfg.train.loss == "infonce":
        return build_loss("infonce", temperature=cfg.train.temperature)
    return build_loss(cfg.train.loss)


def _param_groups(model: torch.nn.Module, cfg: Config) -> list[dict]:
    bb, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if name.startswith("backbone.") else head).append(p)
    return [
        {"params": bb, "lr": cfg.train.lr_backbone, "name": "backbone"},
        {"params": head, "lr": cfg.train.lr_head, "name": "head"},
    ]


def _lr_schedule(total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(round(warmup_ratio * total_steps)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return lr_lambda


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _top1_acc(z: torch.Tensor, labels: torch.Tensor) -> float:
    """For each anchor, does its nearest (non-self) neighbour share a label?"""
    sim = z @ z.T
    sim.fill_diagonal_(float("-inf"))
    nearest = sim.argmax(dim=1)
    return float((labels[nearest] == labels).float().mean())


def _gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_allocated() / (1024**3), 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--run-name", default=None, type=str)
    p.add_argument("--runs-root", default=None, type=Path,
                   help="override runs/ location (used by tests).")
    args = p.parse_args(argv)

    cfg = resolve_config(args.config, args.overrides)
    seed_everything(cfg.train.seed)

    # ---- Data cache sanity checks (fail loudly per CONTEXT.md §1) --------
    cache_dir = Path(cfg.data.wiki_ss_cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"ERROR: data cache is empty or missing at {cache_dir}.\n"
            f"Run: scripts/submit.sh prefetch <target_rows>   (or)\n"
            f"     scripts/prefetch_data.py --data-dir $DATA_DIR --dataset wiki-ss --target-rows N"
        )
    if not M.verify_fingerprint(cache_dir):
        raise SystemExit(
            f"ERROR: data cache fingerprint mismatch at {cache_dir}. "
            f"Cache is stale or corrupted — re-run the prefetch script."
        )
    fingerprint = (cache_dir / ".fingerprint").read_text().strip()

    # ---- Run directory + provenance --------------------------------------
    run_dir = create_run_dir(
        cfg,
        run_name=args.run_name,
        data_fingerprint=fingerprint,
        runs_root=args.runs_root,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data split & datasets -------------------------------------------
    manifest = M.read_manifest(cache_dir)
    train_idx, eval_idx = _split_indices(
        manifest["num_rows"],
        cfg.data.num_train_samples,
        cfg.data.num_eval_samples,
        cfg.train.seed,
    )

    cropper = NonOverlappingCropper(
        crop_size=cfg.cropper.crop_size,
        overlap=cfg.cropper.overlap,
        min_text_ratio=cfg.cropper.min_text_ratio,
        target_size=cfg.cropper.target_size,
    )
    train_ds = LightWikiScreenshotDataset(
        cache_dir,
        indices=train_idx,
        cropper=cropper,
        k_per_page=cfg.train.k_per_page,
        seed=cfg.train.seed,
    )
    # Eval: force min_text_ratio=0 so every non-overlapping tile is kept, and
    # k_per_page doesn't matter (eval uses iter_all_crops).
    eval_cropper = NonOverlappingCropper(
        crop_size=cfg.cropper.crop_size,
        overlap=cfg.cropper.overlap,
        min_text_ratio=0.0,
        target_size=cfg.cropper.target_size,
    )
    eval_ds = LightWikiScreenshotDataset(
        cache_dir,
        indices=eval_idx,
        cropper=eval_cropper,
        k_per_page=2,
        seed=cfg.train.seed + 1,
    )

    loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=multi_positive_collate,
        num_workers=0,          # lazy-decode + small batches: don't eat RAM
        drop_last=True,
    )

    # ---- Model, loss, optimizer, schedule --------------------------------
    model = _build_model(cfg).to(device)
    loss_fn = _build_loss(cfg)
    # Dustbin tracker only meaningful when Sinkhorn-OT is in the forward path.
    # token_only never invokes the score module → discovery would fail.
    # softmax_assign skips Sinkhorn → the metric would be misleading.
    _dustbin_compatible = cfg.model_kind == "salad" and cfg.salad.ablation in {"full", "vlad_only"}
    dustbin_tracker = _DustbinTracker(model, cfg) if _dustbin_compatible else None

    groups = _param_groups(model, cfg)
    optim = torch.optim.AdamW(
        groups,
        weight_decay=cfg.train.weight_decay,
    )
    total_steps = max(1, len(loader) * cfg.train.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, _lr_schedule(total_steps, cfg.train.warmup_ratio)
    )

    # ---- Untrained batch diagnostics (§9) --------------------------------
    diag_batches: list[BatchStats] = []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in itertools.islice(loader, cfg.train.num_diag_batches):
            z = model(imgs.to(device))
            diag_batches.append(compute_batch_stats(z.cpu(), lbls))
    if diag_batches:
        write_report(
            run_dir, diag_batches,
            batch_size=cfg.train.batch_size, k_per_page=cfg.train.k_per_page,
        )

    # ---- Training loop ---------------------------------------------------
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    best_r10 = -1.0
    global_step = 0
    t_start = time.time()

    try:
        for epoch in range(cfg.train.epochs):
            model.train()
            for imgs, lbls in loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)

                optim.zero_grad(set_to_none=True)
                z = model(imgs)
                loss = loss_fn(z, lbls)
                loss.backward()
                if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.train.grad_clip,
                    )
                optim.step()
                scheduler.step()

                # Per-step telemetry — cheap, all on-device.
                with torch.no_grad():
                    stats = compute_batch_stats(z.detach(), lbls)
                    top1 = _top1_acc(z.detach(), lbls)

                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "top1_acc": top1,
                    "pos_sim_mean": stats.pos_sim_mean,
                    "neg_sim_mean": stats.neg_sim_mean,
                    "lr_bb": float(optim.param_groups[0]["lr"]),
                    "lr_head": float(optim.param_groups[1]["lr"]),
                    "gpu_mem_gb": _gpu_mem_gb(),
                    "wall_time_s": round(time.time() - t_start, 2),
                }
                if dustbin_tracker is not None:
                    db = dustbin_tracker.last_value
                    if db is not None:
                        row["dustbin_mass"] = db
                logger.log(row)

                sys.stdout.write(
                    f"\rstep {global_step}/{total_steps}  loss {float(loss):.4f}  "
                    f"top1 {top1:.3f}"
                )
                sys.stdout.flush()

                # Periodic eval ---------------------------------------------
                if (cfg.train.eval_every_steps > 0
                        and global_step > 0
                        and global_step % cfg.train.eval_every_steps == 0):
                    r10 = _eval_and_log(
                        model, eval_ds, cfg, logger, global_step, device,
                    )
                    if r10 > best_r10:
                        best_r10 = r10
                        _save_ckpt(run_dir / "checkpoints" / "best_phase1.pt",
                                   model, step=global_step, recall10=r10)
                    model.train()

                global_step += 1

        # End-of-training eval + checkpoints --------------------------------
        r10 = _eval_and_log(model, eval_ds, cfg, logger, global_step, device)
        if r10 > best_r10:
            best_r10 = r10
            _save_ckpt(run_dir / "checkpoints" / "best_phase1.pt",
                       model, step=global_step, recall10=r10)
        _save_ckpt(run_dir / "checkpoints" / "last.pt",
                   model, step=global_step, recall10=r10)

    finally:
        if dustbin_tracker is not None:
            dustbin_tracker.close()
        logger.close()

    # ---- Final plot ------------------------------------------------------
    rows = read_jsonl(run_dir / "metrics.jsonl")
    plot_train_curves(rows, run_dir / "train_curves.png")

    sys.stdout.write("\n")
    sys.stdout.write(f"run_dir: {run_dir}\n")
    return 0


def _eval_and_log(
    model: torch.nn.Module,
    eval_ds: LightWikiScreenshotDataset,
    cfg: Config,
    logger: JsonlLogger,
    step: int,
    device: torch.device,
) -> float:
    model.eval()
    result = phase1_recall(model, eval_ds, k_values=cfg.eval.k_values, device=device)
    row: dict[str, Any] = {
        "eval_step": step,
        "phase1_num_crops": result["num_crops"],
        "phase1_num_pages": result["num_pages"],
        "phase1_sanity_gap": result["sanity"]["gap"],
        "phase1_monotonic": result["sanity"]["monotonic"],
    }
    for k, v in result["recall"].items():
        row[f"phase1_recall@{k}"] = v
    logger.log(row)
    return float(result["recall"].get("10", 0.0))


def _save_ckpt(path: Path, model: torch.nn.Module, *, step: int, recall10: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "step": step, "phase1_recall@10": recall10},
        path,
    )


if __name__ == "__main__":
    raise SystemExit(main())

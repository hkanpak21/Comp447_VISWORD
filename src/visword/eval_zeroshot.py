"""Zero-shot eval driver — runs Phase 1 + Phase 2 on a frozen backbone.

Bypasses `_load_checkpoint` (there is no training for zero-shot rows of the
method ladder). Creates a fresh `runs/<timestamp>_zeroshot_<label>/` dir,
writes `config.resolved.yaml`, `provenance.json`, `phase1_recall.json`,
`phase2_recall.json`.

Usage (via slurm/eval_zeroshot.sbatch):
    python -m visword.eval_zeroshot --config configs/row04_dinov2_zeroshot.yaml \
        --run-name row04_dinov2_zeroshot
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

# Install DNS shim BEFORE any HF / CLIP / timm imports — compute nodes
# SERVFAIL huggingface.co on the internal resolver.
from visword.hf_dns_shim import install as _install_dns_shim  # noqa: E402
_install_dns_shim()
from pathlib import Path

import torch
import yaml

from visword.config import Config, config_hash
from visword.eval_phase1 import _rebuild_eval_dataset, phase1_recall
from visword.eval_phase2 import phase2_recall
from visword.models.zeroshot import ZeroShotDINOv2
from visword.seed import seed_everything
from visword.train import resolve_config


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()[:12]
    except Exception:
        return "nogit"


def _build_zeroshot_model(cfg: Config) -> torch.nn.Module:
    from visword.models.zeroshot import ZeroShotCLIPImage, ZeroShotImageNetViT
    if cfg.model_kind == "zeroshot_dinov2_cls":
        return ZeroShotDINOv2(cfg, mode="cls")
    if cfg.model_kind == "zeroshot_dinov2_mean":
        return ZeroShotDINOv2(cfg, mode="mean_patch")
    if cfg.model_kind == "zeroshot_clip_image":
        return ZeroShotCLIPImage(cfg)
    if cfg.model_kind == "zeroshot_imagenet_vit":
        return ZeroShotImageNetViT(cfg)
    raise SystemExit(
        f"eval_zeroshot: unsupported model_kind={cfg.model_kind!r}. "
        f"Expected one of: zeroshot_dinov2_cls, zeroshot_dinov2_mean, "
        f"zeroshot_clip_image, zeroshot_imagenet_vit."
    )


def _make_run_dir(cfg: Config, run_name: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    cfg_hash = config_hash(cfg)[:8]
    git = _git_sha()[:8]
    suffix = cfg_hash[:4]
    run_dir = project_root / "runs" / f"{ts}_{git}_{run_name}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_zeroshot(config_path: Path, run_name: str, max_triplets: int | None = None) -> dict:
    cfg = resolve_config(config_path, [])
    seed_everything(cfg.train.seed)

    run_dir = _make_run_dir(cfg, run_name)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    (run_dir / "provenance.json").write_text(json.dumps({
        "git_sha": _git_sha(),
        "config_hash": config_hash(cfg),
        "zeroshot": True,
        "run_name": run_name,
    }, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_zeroshot_model(cfg).to(device).eval()

    # ---------- Phase 1 ----------
    eval_ds = _rebuild_eval_dataset(cfg)
    p1 = phase1_recall(model, eval_ds, k_values=cfg.eval.k_values, device=device)
    p1_payload = {
        "checkpoint": None,
        "checkpoint_step": None,
        "num_pages_evaluated": p1["num_pages"],
        "num_crops": p1["num_crops"],
        "recall": p1["recall"],
        "sanity": p1["sanity"],
    }
    (run_dir / "phase1_recall.json").write_text(json.dumps(p1_payload, indent=2))

    # ---------- Phase 2 ----------
    p2 = phase2_recall(
        model,
        Path(cfg.data.anchors_cache_dir),
        target_size=cfg.cropper.target_size,
        k_values=cfg.eval.k_values,
        max_triplets=max_triplets,
        device=device,
    )
    p2_payload = {"checkpoint": None, "checkpoint_step": None, **p2}
    (run_dir / "phase2_recall.json").write_text(json.dumps(p2_payload, indent=2))

    sys.stdout.write(f"zero-shot eval done — {run_dir}\n")
    sys.stdout.write(f"  Phase1 R@10 = {p1['recall'].get(10, 'n/a')}\n")
    sys.stdout.write(f"  Phase2 R@1  = {p2['recall'].get(1, 'n/a')}\n")
    return {"run_dir": str(run_dir), "phase1": p1_payload, "phase2": p2_payload}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--run-name", required=True, type=str)
    p.add_argument("--max-triplets", type=int, default=None)
    args = p.parse_args(argv)
    run_zeroshot(args.config, args.run_name, max_triplets=args.max_triplets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Ticket 01 data-run: legible crops + a disjoint eval slice.

Produces, into one self-contained run dir:
  1. ``eval_split.json`` — a deterministic, held-out page slice disjoint from any
     training prefix (eval = the LAST ``num_eval`` pages of a seeded permutation;
     training draws from the head, so the two never overlap for any
     ``num_train <= num_rows - num_eval``).
  2. ``samples/page_<idx>_old_vs_new.png`` — per sample page, a montage comparing
     the OLD cropper (NonOverlappingCropper 490->224, the 2.19x shrink that made body
     text illegible) against the NEW TextAwareCropper (native 224, line-gap snapped).
     This is the operator eyeball check for AC4.
  3. ``summary.json`` + ``provenance.json`` — reproducibility (git SHA, args, host).

Reuses existing conventions: manifest loading (visword.data.manifest), the seeded-
permutation split (mirrors train.py), and the run-dir layout. No experiment trackers.

Run (on a dev worktree, login node — this is light CPU + image I/O):
    PYTHONPATH=src python scripts/build_legible_crops.py \
        --cache-dir "$DATA_DIR/wiki_ss" --num-eval 2000 --sample-pages 16
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from visword.data import manifest as M  # noqa: E402
from visword.data.cropper import NonOverlappingCropper, TextAwareCropper  # noqa: E402


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def disjoint_eval_indices(num_rows: int, num_eval: int, seed: int) -> np.ndarray:
    """Held-out tail of a seeded permutation — disjoint from any training head."""
    perm = np.random.default_rng(seed).permutation(num_rows)
    return perm[num_rows - num_eval :]


def montage(page_img: Image.Image, old_crops, new_crops, out_path: Path, n: int = 6) -> None:
    """Original page (left) + a row of OLD crops + a row of NEW crops."""
    n_old, n_new = min(n, len(old_crops)), min(n, len(new_crops))
    ncol = 1 + max(n_old, n_new)
    fig, axes = plt.subplots(2, ncol, figsize=(2.0 * ncol, 4.4))
    # Original page spanning both rows in column 0.
    gs = axes[0, 0].get_gridspec()
    for ax in axes[:, 0]:
        ax.remove()
    ax_page = fig.add_subplot(gs[:, 0])
    ax_page.imshow(page_img)
    ax_page.set_title("page", fontsize=8)
    ax_page.axis("off")
    for j in range(1, ncol):
        for r, (crops, label) in enumerate(((old_crops, "old 490->224"), (new_crops, "new native"))):
            ax = axes[r, j]
            k = j - 1
            if k < len(crops):
                ax.imshow(crops[k])
                if k == 0:
                    ax.set_ylabel(label, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(out_path.stem, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--num-eval", type=int, default=2000)
    ap.add_argument("--sample-pages", type=int, default=16)
    ap.add_argument("--crop-size", type=int, default=224, help="native window (== target)")
    ap.add_argument("--old-crop-size", type=int, default=490)
    ap.add_argument("--old-target-size", type=int, default=224)
    ap.add_argument("--min-text-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = M.read_manifest(args.cache_dir)
    rows = manifest["rows"]
    num_rows = manifest.get("num_rows", len(rows))
    if args.num_eval > num_rows:
        args.num_eval = num_rows

    eval_idx = disjoint_eval_indices(num_rows, args.num_eval, args.seed)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.runs_root / f"{ts}_{git_sha(root)}_legible_crops"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)

    (run_dir / "provenance.json").write_text(json.dumps({
        "ts_utc": ts, "host": socket.gethostname(), "git_sha": git_sha(root),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "num_rows": int(num_rows),
    }, indent=2))

    (run_dir / "eval_split.json").write_text(json.dumps({
        "seed": args.seed, "num_rows": int(num_rows), "num_eval": int(args.num_eval),
        "definition": "tail of seeded permutation; training draws from the head (disjoint)",
        "eval_idx": [int(i) for i in eval_idx],
        "eval_docids": [rows[int(i)].get("docid") for i in eval_idx],
    }, indent=2))

    new_cropper = TextAwareCropper(
        crop_size=args.crop_size, target_size=args.crop_size, min_text_ratio=args.min_text_ratio
    )
    old_cropper = NonOverlappingCropper(
        crop_size=args.old_crop_size, target_size=args.old_target_size,
        min_text_ratio=args.min_text_ratio,
    )

    n_old_total = n_new_total = 0
    sample = eval_idx[: args.sample_pages]
    for gi in sample:
        row = rows[int(gi)]
        with Image.open(args.cache_dir / row["image_path"]) as im:
            page = im.convert("RGB")
            old_crops = old_cropper(page)
            new_crops = new_cropper(page)
            n_old_total += len(old_crops)
            n_new_total += len(new_crops)
            montage(page, old_crops, new_crops, run_dir / "samples" / f"page_{int(gi)}_old_vs_new.png")

    (run_dir / "summary.json").write_text(json.dumps({
        "sample_pages": int(len(sample)),
        "mean_crops_per_page_old": n_old_total / max(1, len(sample)),
        "mean_crops_per_page_new": n_new_total / max(1, len(sample)),
        "new_cropper": {"crop_size": args.crop_size, "target_size": args.crop_size},
        "old_cropper": {"crop_size": args.old_crop_size, "target_size": args.old_target_size},
    }, indent=2))

    print(f"run_dir: {run_dir}")
    print(f"eval slice: {args.num_eval} pages (seed {args.seed}) -> eval_split.json")
    print(f"sample montages: {len(sample)} -> {run_dir}/samples/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

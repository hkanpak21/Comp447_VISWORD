"""V1 report figures (legible-resolution, page-level same-page re-id on the 2000-page slice).

Two PNGs into paper/report_template/figures/ (additive — Barış can include them):
  1. fig_v1_reader_progression.png — "teaching a non-reader to read": frozen MAE → our
     reader variants, with CLIP and the perfect-text ceiling as reference lines.
  2. fig_v1_encoder_grid.png — the legible-resolution zero-shot encoder grid (R@10).

Data-driven where possible: the encoder grid is read from the grid run-dir; reader numbers
are passed in (they live across several run-dirs). Run on a node with matplotlib.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def reader_progression_fig(out_path: Path, dit_r10: float | None) -> None:
    # 2000-page-gallery R@10 (from RESULTS.md ticket 04/04b/07).
    bars = [
        ("frozen MAE", 0.036, "#bbbbbb"),
        ("regress→CLS", 0.039, "#cda0a0"),
        ("contrastive", 0.098, "#7fa7d0"),
        ("+title-mask", 0.143, "#3f7fc0"),
    ]
    if dit_r10 is not None:
        bars.append(("DiT +mask", round(dit_r10, 3), "#2a9d5c"))
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    cols = [b[2] for b in bars]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    xs = range(len(bars))
    ax.bar(xs, vals, color=cols, width=0.62)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.736, ls="--", c="#d08a2a", lw=1.3)
    ax.text(len(bars) - 0.5, 0.748, "CLIP (zero-shot) 0.736", ha="right", c="#d08a2a", fontsize=8)
    ax.axhline(0.938, ls=":", c="#444444", lw=1.3)
    ax.text(len(bars) - 0.5, 0.95, "perfect-text ceiling 0.938", ha="right", c="#444444", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("page-level R@10 (2000-page gallery)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Teaching a non-reader to read: MAE/DiT reader at legible resolution", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def encoder_grid_fig(out_path: Path, grid_json: Path) -> None:
    if not grid_json.exists():
        print(f"[skip grid fig] {grid_json} not found"); return
    rows = sorted(json.loads(grid_json.read_text()), key=lambda r: r["recall"]["10"])
    labels = [r["encoder"] for r in rows]
    vals = [r["recall"]["10"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(range(len(rows)), vals, color="#5b8db8")
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("page-level R@10 (frozen, native-224)")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title("Legible-resolution zero-shot encoder grid", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", type=Path,
                    default=Path("paper/report_template/figures"))
    ap.add_argument("--grid-json", type=Path,
                    default=Path("runs/rebaseline_grid_v1/grid_summary.json"))
    ap.add_argument("--dit-final", type=Path, default=Path("runs/dit_reader_v2/final_eval.json"),
                    help="DiT reader final_eval.json (its R@10 added to the progression if present)")
    args = ap.parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)

    dit_r10 = None
    if args.dit_final.exists():
        dit_r10 = json.loads(args.dit_final.read_text())["recall"]["10"]

    reader_progression_fig(args.figdir / "fig_v1_reader_progression.png", dit_r10)
    encoder_grid_fig(args.figdir / "fig_v1_encoder_grid.png", args.grid_json)
    print(f"wrote figures to {args.figdir} (dit_r10={dit_r10})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

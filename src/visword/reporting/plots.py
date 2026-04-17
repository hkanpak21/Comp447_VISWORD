"""Matplotlib helpers with a blueish palette (PROJECT_SPEC.md §6 / §1.4).

No external plotting tooling — matplotlib is the only dep per spec §1.4.
Keep the palette boring on purpose so plots are readable in both light and
dark backgrounds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)   # non-GUI backend — we only write PNGs.

import matplotlib.pyplot as plt     # noqa: E402


# Blueish palette: cool blue, teal, steel, slate, sky.
PALETTE = ["#1f4e79", "#2e7d8d", "#4682b4", "#6e86a8", "#87ceeb"]


def _apply_style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_train_curves(metrics_rows: list[dict], out_path: Path) -> Path:
    """Render ``train_curves.png`` from a list of metrics.jsonl rows.

    Produces a 1×2 figure:
      * left:  loss vs step
      * right: mean pos-sim / mean neg-sim (and phase1_recall@10 on a twin
               axis if any eval rows are present).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows = [r for r in metrics_rows if "loss" in r and "step" in r]
    eval_rows = [r for r in metrics_rows if "eval_step" in r]

    fig, (ax_loss, ax_sim) = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

    if train_rows:
        steps = [r["step"] for r in train_rows]
        ax_loss.plot([r["step"] for r in train_rows],
                     [r["loss"] for r in train_rows],
                     color=PALETTE[0], linewidth=1.2, label="train loss")
        ax_loss.set_xlabel("step")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title("Training loss")

        def _line(key: str, colour: str, label: str) -> None:
            ys = [r.get(key) for r in train_rows]
            if any(y is not None for y in ys):
                ax_sim.plot([s for s, y in zip(steps, ys) if y is not None],
                            [y for y in ys if y is not None],
                            color=colour, linewidth=1.2, label=label)

        _line("pos_sim_mean", PALETTE[1], "pos_sim_mean")
        _line("neg_sim_mean", PALETTE[3], "neg_sim_mean")
        ax_sim.set_xlabel("step")
        ax_sim.set_ylabel("cosine similarity")
        ax_sim.set_title("Batch-level pos/neg sim")
        ax_sim.legend(loc="best", fontsize=8)

    if eval_rows:
        ax_eval = ax_sim.twinx()
        eval_steps = [r["eval_step"] for r in eval_rows]
        recall10 = [r.get("phase1_recall@10") for r in eval_rows]
        ax_eval.plot(eval_steps, recall10, color=PALETTE[4],
                     linewidth=1.5, marker="o", markersize=4, label="R@10")
        ax_eval.set_ylabel("phase1 R@10", color=PALETTE[4])
        ax_eval.tick_params(axis="y", labelcolor=PALETTE[4])

    for ax in (ax_loss, ax_sim):
        _apply_style(ax)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path

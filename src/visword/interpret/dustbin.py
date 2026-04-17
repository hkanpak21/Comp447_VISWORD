"""Dustbin-mass evolution plot (PROJECT_SPEC.md §8).

Reads ``dustbin_mass`` values out of ``metrics.jsonl`` (one value per
training step, if the SALAD dustbin hook was registered) and plots them
against step. If the column is missing — e.g. a CLS-only baseline run or
a run where the hook was disabled — emits a one-panel placeholder so the
downstream eval job doesn't error out.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_dustbin_series(metrics_path: Path) -> tuple[list[int], list[float]]:
    """Return ``(steps, dustbin_values)`` from rows that have both fields."""
    steps: list[int] = []
    values: list[float] = []
    for line in Path(metrics_path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "dustbin_mass" in row and "step" in row:
            steps.append(int(row["step"]))
            values.append(float(row["dustbin_mass"]))
    return steps, values


def plot_dustbin_evolution(metrics_path: Path, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from visword.reporting.plots import PALETTE

    steps, values = load_dustbin_series(metrics_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    if values:
        ax.plot(steps, values, color=PALETTE[0], linewidth=1.2, marker="o", markersize=3)
        ax.set_xlabel("step")
        ax.set_ylabel("SALAD dustbin mass fraction")
        ax.set_title("Dustbin mass over training")
        ax.set_ylim(0, max(0.05, max(values) * 1.1))
    else:
        ax.text(0.5, 0.5, "no dustbin_mass rows in metrics.jsonl",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_axis_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


__all__ = ["load_dustbin_series", "plot_dustbin_evolution"]

"""LEACE rank-sweep figure for the appendix.

Plots R@10 vs LEACE rank for trained DINOv2-SALAD-50k. Pixel-blanking
delta drawn as a horizontal reference line. The curve crosses the
blanking delta near rank-64.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/scratch/hkanpak21/VISWORD")
OUT = ROOT / "paper/report_template/figures/leace_rank_sweep.pdf"

# From runs/_trained_descr/dinov2_salad_50k_rank_sweep.txt
ranks       = np.array([1,    4,    16,   64,   256,  1024, 4000])
expl_var    = np.array([0.20, 0.31, 0.52, 0.70, 0.83, 0.95, 1.00])
r10_leace   = np.array([0.912, 0.881, 0.776, 0.646, 0.526, 0.440, 0.417])
r10_orig    = 0.911
r10_blank   = 0.670

fig, ax = plt.subplots(1, 1, figsize=(4.4, 2.8))
ax.semilogx(ranks, r10_leace, "o-", color="#3F7AB1", linewidth=1.6,
            markersize=5, label="LEACE-erased R@10")
ax.axhline(r10_orig,  color="#7A9E7E", linestyle="--", linewidth=1.0,
           label=f"original ({r10_orig:.3f})")
ax.axhline(r10_blank, color="#C56B5C", linestyle="--", linewidth=1.0,
           label=f"pixel blank-15 ({r10_blank:.3f})")
ax.set_xlabel("LEACE protected-attribute rank", fontsize=9)
ax.set_ylabel("Protocol-A R@10", fontsize=9)
ax.set_xticks(ranks)
ax.set_xticklabels([str(r) for r in ranks], fontsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.set_ylim(0.35, 1.0)
ax.grid(alpha=0.25)
ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
ax.set_title("trained DINOv2-SALAD-50k: rank-64 LEACE matches blanking",
             fontsize=9)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")

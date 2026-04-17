"""CLS-vs-VLAD similarity decomposition (PROJECT_SPEC.md §8).

The official SALAD aggregator concatenates a (num_clusters * cluster_dim)-d
VLAD block and a (token_dim)-d MLP-projected CLS token, then L2-normalises
the concatenation. Because the two halves live on disjoint dimensions of
the same unit vector, the *cosine similarity* between two descriptors
splits exactly into a VLAD-only contribution + a CLS-only contribution::

    cos(a, b) = a . b
              = sum_{i in VLAD} a_i b_i + sum_{i in CLS} a_i b_i

So we can ask "how much of a same-page similarity comes from the CLS
token vs the dense VLAD aggregation?" without re-forwarding.

This is the fix we didn't have in week 2 (CONTEXT.md session 2): our
reimplemented SALAD dropped the CLS branch and silently underperformed.
With the official module + this decomposition, that kind of regression
becomes visible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import torch

from visword.config import Config


@dataclass
class CLSVsVLADResult:
    num_clusters: int
    cluster_dim: int
    token_dim: int
    vlad_dim: int                       # num_clusters * cluster_dim
    total_dim: int                      # vlad_dim + token_dim
    same_page_full_cos: float           # mean over same-page pairs
    same_page_vlad_cos: float
    same_page_cls_cos: float
    diff_page_full_cos: float
    diff_page_vlad_cos: float
    diff_page_cls_cos: float

    def to_dict(self) -> dict:
        return asdict(self)


def _split_descriptor(z: torch.Tensor, vlad_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if z.shape[-1] < vlad_dim:
        raise ValueError(
            f"descriptor has dim {z.shape[-1]} < vlad_dim {vlad_dim}; "
            f"check cfg.salad.num_clusters and cfg.salad.cluster_dim"
        )
    return z[..., :vlad_dim], z[..., vlad_dim:]


def pairwise_cos_by_half(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    vlad_dim: int,
) -> CLSVsVLADResult:
    """Compute the decomposition over a batch of descriptors.

    The VLAD and CLS halves are taken *as-is* from the already-normalised
    full descriptor — so vlad_cos + cls_cos = full_cos exactly (up to fp
    round-off). Means are taken over all same-page and diff-page pairs
    excluding the diagonal.
    """
    vlad, cls = _split_descriptor(embeddings, vlad_dim)

    full_sim = embeddings @ embeddings.T
    vlad_sim = vlad @ vlad.T
    cls_sim = cls @ cls.T

    n = embeddings.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    diff = (~same) & ~eye

    def _mean(mat: torch.Tensor, mask: torch.Tensor) -> float:
        if not mask.any():
            return float("nan")
        return float(mat[mask].mean())

    return CLSVsVLADResult(
        num_clusters=0,    # populated by caller
        cluster_dim=0,
        token_dim=embeddings.shape[-1] - vlad_dim,
        vlad_dim=vlad_dim,
        total_dim=embeddings.shape[-1],
        same_page_full_cos=_mean(full_sim, same),
        same_page_vlad_cos=_mean(vlad_sim, same),
        same_page_cls_cos=_mean(cls_sim, same),
        diff_page_full_cos=_mean(full_sim, diff),
        diff_page_vlad_cos=_mean(vlad_sim, diff),
        diff_page_cls_cos=_mean(cls_sim, diff),
    )


def decompose(
    cfg: Config,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> CLSVsVLADResult:
    """Wrapper that reads vlad_dim from the config."""
    vlad_dim = cfg.salad.num_clusters * cfg.salad.cluster_dim
    result = pairwise_cos_by_half(embeddings, labels, vlad_dim)
    result.num_clusters = cfg.salad.num_clusters
    result.cluster_dim = cfg.salad.cluster_dim
    return result


def plot_cls_vs_vlad(
    result: CLSVsVLADResult,
    out_path: Path,
) -> Path:
    """Render ``cls_vs_vlad.png`` — a stacked bar comparing contributions."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from visword.reporting.plots import PALETTE

    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    labels = ["same-page", "diff-page"]
    vlad_vals = [result.same_page_vlad_cos, result.diff_page_vlad_cos]
    cls_vals = [result.same_page_cls_cos, result.diff_page_cls_cos]

    ax.bar(labels, vlad_vals, color=PALETTE[0], label="VLAD half")
    ax.bar(labels, cls_vals, bottom=vlad_vals, color=PALETTE[3], label="CLS half")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("mean cosine similarity")
    ax.set_title(f"CLS-vs-VLAD contribution (slice at {result.vlad_dim}/{result.total_dim})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def write_report(result: CLSVsVLADResult, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.write_text(json.dumps(result.to_dict(), indent=2))
    return out_path


__all__ = [
    "CLSVsVLADResult",
    "decompose",
    "pairwise_cos_by_half",
    "plot_cls_vs_vlad",
    "write_report",
]

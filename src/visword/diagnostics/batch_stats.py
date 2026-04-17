"""Pre-training batch diagnostics (PROJECT_SPEC.md §9).

Called once before training starts against random-init embeddings. Writes
``diagnostics/untrained_batch_stats.json`` into the run dir. Every field in
this report has caught a real bug during prior sessions (CONTEXT.md); they
are more useful than a single hero accuracy number.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass
class BatchStats:
    positives_per_query_mean: float
    negatives_per_query_mean: float
    pos_sim_mean: float
    neg_sim_mean: float
    hard_neg_frac: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_batch_stats(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    hard_neg_threshold: float | None = None,
) -> BatchStats:
    """Stats over a single flattened batch of L2-normed embeddings.

    Args:
        embeddings: ``(N, D)`` L2-normed.
        labels: ``(N,)`` integer page labels. Same label ⇒ positives.
        hard_neg_threshold: a negative is "hard" if its cosine similarity
            is greater than this threshold. If None, use the mean positive
            similarity (a negative is hard if it's at least as close as
            a typical true positive).
    """
    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError("embeddings must be (N,D) and labels must be (N,)")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("batch size mismatch between embeddings and labels")

    sim = embeddings @ embeddings.T
    n = embeddings.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=labels.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    neg_mask = ~pos_mask & ~self_mask

    pos_counts = pos_mask.sum(dim=1).float()
    neg_counts = neg_mask.sum(dim=1).float()

    pos_sim = sim[pos_mask]
    neg_sim = sim[neg_mask]

    pos_sim_mean = float(pos_sim.mean()) if pos_sim.numel() > 0 else float("nan")
    neg_sim_mean = float(neg_sim.mean()) if neg_sim.numel() > 0 else float("nan")

    if hard_neg_threshold is None:
        hard_neg_threshold = pos_sim_mean if pos_sim.numel() > 0 else 0.0
    if neg_sim.numel() > 0:
        hard_neg_frac = float((neg_sim > hard_neg_threshold).float().mean())
    else:
        hard_neg_frac = float("nan")

    return BatchStats(
        positives_per_query_mean=float(pos_counts.mean()),
        negatives_per_query_mean=float(neg_counts.mean()),
        pos_sim_mean=pos_sim_mean,
        neg_sim_mean=neg_sim_mean,
        hard_neg_frac=hard_neg_frac,
    )


def aggregate_stats(per_batch: list[BatchStats]) -> dict:
    """Mean-over-batches summary, matching the schema in spec §9."""
    if not per_batch:
        raise ValueError("aggregate_stats needs at least one batch")
    keys = ("positives_per_query_mean", "negatives_per_query_mean",
            "pos_sim_mean", "neg_sim_mean", "hard_neg_frac")
    out = {k: statistics.fmean(getattr(b, k) for b in per_batch) for k in keys}
    out["n_batches_sampled"] = len(per_batch)
    return out


def write_report(
    run_dir: Path,
    batches: list[BatchStats],
    *,
    batch_size: int,
    k_per_page: int,
    note: str = "Expected ~0.5 hard-neg fraction at initialisation; should drop over training.",
) -> Path:
    """Write ``diagnostics/untrained_batch_stats.json`` under ``run_dir``."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        **aggregate_stats(batches),
        "batch_size": batch_size,
        "k_per_page": k_per_page,
        "note": note,
    }
    # Rename the aggregate key to match the spec §9 example exactly.
    payload["hard_neg_frac_mean"] = payload.pop("hard_neg_frac")

    target = out_dir / "untrained_batch_stats.json"
    target.write_text(json.dumps(payload, indent=2))
    return target

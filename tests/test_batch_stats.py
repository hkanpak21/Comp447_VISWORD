"""Phase B / B4 tests for the pre-training diagnostics (TESTS.md B4)."""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from visword.diagnostics.batch_stats import (
    BatchStats,
    aggregate_stats,
    compute_batch_stats,
    write_report,
)


def _rand_normed(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_batch_report_fields_and_positives_per_query():
    """Every required field is populated; positives_per_query_mean == k-1."""
    B, K, D = 8, 4, 32
    n = B * K
    z = _rand_normed(n, D, seed=0)
    # 8 pages, 4 crops each → same-page positives per query = 3 = K - 1.
    labels = torch.arange(B).repeat_interleave(K)

    stats = compute_batch_stats(z, labels)

    # Shape
    expected_fields = {
        "positives_per_query_mean",
        "negatives_per_query_mean",
        "pos_sim_mean",
        "neg_sim_mean",
        "hard_neg_frac",
    }
    assert set(stats.to_dict().keys()) == expected_fields

    # Exactly K-1 positives per anchor.
    assert stats.positives_per_query_mean == float(K - 1), stats.positives_per_query_mean
    # Negatives = N - 1 (self) - (K-1) (positives) = N - K.
    assert stats.negatives_per_query_mean == float(n - K), stats.negatives_per_query_mean

    # Random-init cosine sims should be near zero with D=32; loose bound.
    assert -0.5 < stats.pos_sim_mean < 0.5
    assert -0.5 < stats.neg_sim_mean < 0.5


def test_aggregate_over_multiple_batches():
    B, K, D = 4, 4, 16
    n = B * K
    labels = torch.arange(B).repeat_interleave(K)

    batches = [
        compute_batch_stats(_rand_normed(n, D, seed=s), labels)
        for s in range(3)
    ]
    agg = aggregate_stats(batches)

    assert agg["n_batches_sampled"] == 3
    assert agg["positives_per_query_mean"] == float(K - 1)
    for key in ("pos_sim_mean", "neg_sim_mean", "hard_neg_frac"):
        assert key in agg


def test_write_report_produces_expected_schema(tmp_path):
    B, K, D = 4, 4, 16
    n = B * K
    labels = torch.arange(B).repeat_interleave(K)
    batches = [compute_batch_stats(_rand_normed(n, D, seed=s), labels) for s in range(2)]

    out = write_report(tmp_path, batches, batch_size=B, k_per_page=K)
    data = json.loads(out.read_text())

    # Match the spec §9 example shape.
    for key in (
        "n_batches_sampled",
        "batch_size",
        "k_per_page",
        "positives_per_query_mean",
        "negatives_per_query_mean",
        "pos_sim_mean",
        "neg_sim_mean",
        "hard_neg_frac_mean",
        "note",
    ):
        assert key in data, f"missing {key!r} in {list(data)}"

    assert data["batch_size"] == B
    assert data["k_per_page"] == K
    assert data["positives_per_query_mean"] == float(K - 1)

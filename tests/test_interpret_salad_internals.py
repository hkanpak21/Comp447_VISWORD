"""Phase E / E2 — SALAD internals hook discovery + Sinkhorn sanity (TESTS.md E2).

These tests work on **fresh** OfficialSALAD instances (random weights); no
trained model required. Discovery is by shape and the Sinkhorn algorithm
is deterministic, so CPU is sufficient.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from visword.interpret.salad_internals import (
    capture_score_tensor,
    discover_salad_submodules,
    dustbin_mass_fraction,
    load_hooks_json,
    save_hooks_json,
    sinkhorn_assignment,
)
from visword.models.salad_bridge import OfficialSALAD


NUM_CHANNELS = 768
NUM_CLUSTERS = 64
CLUSTER_DIM = 128
TOKEN_DIM = 256
H = W = 16


def _build_aggregator() -> OfficialSALAD:
    return OfficialSALAD(
        num_channels=NUM_CHANNELS,
        num_clusters=NUM_CLUSTERS,
        cluster_dim=CLUSTER_DIM,
        token_dim=TOKEN_DIM,
        dropout=0.0,
    )


def test_discover_submodules_identifies_all_three():
    agg = _build_aggregator()
    hooks = discover_salad_submodules(
        agg, num_channels=NUM_CHANNELS, H=H, W=W,
    )
    assert hooks.score, "score submodule not identified"
    assert hooks.cluster_features, "cluster_features submodule not identified"
    assert hooks.token_features, "token_features submodule not identified"

    # Each resolved name should actually exist inside the aggregator.
    named = dict(agg.named_modules())
    for attr in ("score", "cluster_features", "token_features"):
        path = getattr(hooks, attr)
        assert path in named, f"{attr} -> {path} not in aggregator submodules"


def test_hooks_json_roundtrip(tmp_path):
    agg = _build_aggregator()
    hooks = discover_salad_submodules(agg, num_channels=NUM_CHANNELS, H=H, W=W)
    save_hooks_json(hooks, tmp_path)
    reloaded = load_hooks_json(tmp_path)
    assert reloaded.to_dict() == hooks.to_dict()


def test_sinkhorn_assignment_is_doubly_stochastic():
    torch.manual_seed(0)
    agg = _build_aggregator().eval()
    hooks = discover_salad_submodules(agg, num_channels=NUM_CHANNELS, H=H, W=W)

    patches = torch.randn(2, NUM_CHANNELS, H, W)
    cls = torch.randn(2, NUM_CHANNELS)

    score = capture_score_tensor(agg, patches, cls, hooks.score)
    assert score.shape == (2, NUM_CLUSTERS, H, W)

    with torch.no_grad():
        assignment = sinkhorn_assignment(score, agg.dust_bin)

    # Shape: (B, num_clusters + 1, H*W)
    assert assignment.shape == (2, NUM_CLUSTERS + 1, H * W)

    # Column sums (per patch total mass) close to 1 for every patch.
    col_sum = assignment.sum(dim=1)                  # (B, H*W)
    assert torch.allclose(col_sum, torch.ones_like(col_sum), atol=1e-3), (
        f"column-sum deviation: {(col_sum - 1).abs().max().item()}"
    )

    # Cluster-row sums close to 1 (dustbin row is a special mass sink and
    # can differ; §8's "doubly-stochastic" refers to the m × n cluster block).
    cluster_row_sum = assignment[:, :NUM_CLUSTERS, :].sum(dim=2)    # (B, num_clusters)
    assert torch.allclose(cluster_row_sum, torch.ones_like(cluster_row_sum), atol=1e-3), (
        f"cluster-row-sum deviation: {(cluster_row_sum - 1).abs().max().item()}"
    )


def test_dustbin_mass_in_unit_interval():
    torch.manual_seed(1)
    agg = _build_aggregator().eval()
    hooks = discover_salad_submodules(agg, num_channels=NUM_CHANNELS, H=H, W=W)

    patches = torch.randn(1, NUM_CHANNELS, H, W)
    cls = torch.randn(1, NUM_CHANNELS)
    score = capture_score_tensor(agg, patches, cls, hooks.score)

    with torch.no_grad():
        assignment = sinkhorn_assignment(score, agg.dust_bin)
    mass = dustbin_mass_fraction(assignment)
    assert 0.0 <= mass <= 1.0, f"dustbin mass out of [0, 1]: {mass}"

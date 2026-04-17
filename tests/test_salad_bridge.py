"""Phase A tests A1 — SALAD bridge module (PROJECT_SPEC.md §5.2, TESTS.md A1)."""
from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDORED = PROJECT_ROOT / "third_party" / "salad"


def _purge_bridge_modules() -> None:
    """Remove any cached bridge / SALAD modules so re-imports actually re-run."""
    for name in list(sys.modules):
        if (
            name == "visword.models.salad_bridge"
            or name.startswith("models.aggregators")
            or name.startswith("models.backbones")
            or name == "models"
        ):
            del sys.modules[name]


def test_bridge_imports_official_classes() -> None:
    _purge_bridge_modules()
    bridge = importlib.import_module("visword.models.salad_bridge")

    assert bridge.OfficialSALAD is not None
    assert bridge.OfficialDINOv2 is not None

    salad_path = Path(sys.modules[bridge.OfficialSALAD.__module__].__file__).resolve()
    dinov2_path = Path(sys.modules[bridge.OfficialDINOv2.__module__].__file__).resolve()
    assert VENDORED in salad_path.parents, (
        f"OfficialSALAD must come from third_party/salad/, got {salad_path}"
    )
    assert VENDORED in dinov2_path.parents, (
        f"OfficialDINOv2 must come from third_party/salad/, got {dinov2_path}"
    )


def test_bridge_fails_cleanly_without_vendored_repo(tmp_path: Path) -> None:
    if not VENDORED.exists():
        pytest.skip("vendored SALAD missing; cannot test the rename round-trip")

    _purge_bridge_modules()
    hidden = VENDORED.with_name("salad__hidden_for_test")
    VENDORED.rename(hidden)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            importlib.import_module("visword.models.salad_bridge")
        assert "scripts/vendor_salad.sh" in str(exc_info.value)
    finally:
        hidden.rename(VENDORED)
        _purge_bridge_modules()


def test_salad_forward_shape() -> None:
    _purge_bridge_modules()
    bridge = importlib.import_module("visword.models.salad_bridge")

    num_channels, num_clusters, cluster_dim, token_dim = 768, 64, 128, 256
    salad = bridge.OfficialSALAD(
        num_channels=num_channels,
        num_clusters=num_clusters,
        cluster_dim=cluster_dim,
        token_dim=token_dim,
        dropout=0.0,
    ).eval()

    B, side = 2, 16
    patches = torch.randn(B, num_channels, side, side)
    cls = torch.randn(B, num_channels)

    with torch.no_grad():
        out = salad((patches, cls))

    expected_dim = num_clusters * cluster_dim + token_dim  # = 8448
    assert out.shape == (B, expected_dim), f"got {tuple(out.shape)}"
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), f"not L2-normalised: {norms}"

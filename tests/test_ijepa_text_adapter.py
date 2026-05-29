from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.ijepa_text_adapter import (
    ADAPTER_KINDS,
    AdapterSpec,
    build_text_adapter,
    count_parameters,
    no_adapter_baseline,
    run_one_adapter,
    write_summary,
)


def test_text_adapter_output_shapes() -> None:
    x = torch.randn(4, 1280)
    for kind in ADAPTER_KINDS:
        adapter = build_text_adapter(kind, d_in=1280, d_out=768, rank=64, hidden_dim=128)
        y = adapter(x)
        assert y.shape == (4, 768)
        assert count_parameters(adapter) > 0


def test_tiny_synthetic_adapter_report(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(64, 1280)).astype(np.float32)
    y_train = rng.normal(size=(64, 768)).astype(np.float32)
    x_eval = rng.normal(size=(32, 1280)).astype(np.float32)
    y_eval = rng.normal(size=(32, 768)).astype(np.float32)

    args = SimpleNamespace(
        out_dir=tmp_path,
        n_train=64,
        n_eval=32,
        seed=42,
        text_source="title",
        steps=2,
        batch_size=16,
        lr=1e-3,
        tau=0.07,
        weight_decay=1e-4,
        rank=64,
        hidden_dim=128,
    )
    baseline = no_adapter_baseline(x_eval, y_eval)
    report = run_one_adapter(
        AdapterSpec("mlp", hidden_dim=128),
        x_train,
        y_train,
        x_eval,
        y_eval,
        baseline,
        args,
        torch.device("cpu"),
    )
    summary_path = write_summary([report], baseline, args)

    assert (tmp_path / "mlp_h128_seed42.json").exists()
    assert summary_path.exists()
    assert set(report["adapted"]["recall"]) == {"1", "5", "10", "20"}
    assert "gap" in report["adapted"]["sanity"]
    assert report["adapter"]["parameter_count"] == count_parameters(
        build_text_adapter("mlp", 1280, 768, hidden_dim=128)
    )

"""Phase C / C1 — end-to-end debug training (TESTS.md C1).

Runs the real ``visword.train`` entry point against the prefetched
50-row cache under ``$DATA_DIR``. Requires GPU (``@pytest.mark.gpu``) and
a populated cache (``@pytest.mark.integration``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# ``debug_run`` fixture is now session-scoped in tests/conftest.py.


def test_run_dir_created_with_required_artifacts(debug_run):
    must_exist = [
        "config.resolved.yaml",
        "provenance.json",
        "metrics.jsonl",
        "train_curves.png",
        "diagnostics/untrained_batch_stats.json",
        "checkpoints/last.pt",
        "checkpoints/best_phase1.pt",
    ]
    missing = [p for p in must_exist if not (debug_run / p).exists()]
    assert not missing, f"missing from run dir: {missing}"


def test_metrics_jsonl_schema(debug_run):
    rows = [json.loads(l) for l in (debug_run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) >= 1

    train_rows = [r for r in rows if "loss" in r]
    assert train_rows, "no training rows in metrics.jsonl"
    r0 = train_rows[0]
    for field in ("step", "epoch", "loss", "top1_acc",
                  "pos_sim_mean", "neg_sim_mean",
                  "lr_bb", "lr_head", "gpu_mem_gb", "wall_time_s"):
        assert field in r0, f"missing {field!r} in {list(r0)}"

    # Step-0 loss is finite and positive.
    step0 = next(r for r in train_rows if r["step"] == 0)
    assert step0["loss"] > 0.0 and step0["loss"] < 1e6

    # At least one eval row emitted.
    eval_rows = [r for r in rows if "eval_step" in r]
    assert eval_rows, "no eval rows emitted; check train.eval_every_steps"
    assert "phase1_recall@10" in eval_rows[0]


def test_no_file_written_outside_run_dir(debug_run):
    """The run dir's parent contains only the run dir — no stray output."""
    runs_root = debug_run.parent
    entries = list(runs_root.iterdir())
    assert entries == [debug_run], entries

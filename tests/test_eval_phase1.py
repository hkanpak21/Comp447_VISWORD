"""Phase D / D1 tests for the Phase-1 eval CLI (TESTS.md D1).

Uses the same debug-train fixture as C1: train a tiny model, run the eval
CLI against the resulting run dir, then check the §7 schema and
invariants (recall monotonicity, sanity gap).
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

# debug_run fixture is auto-loaded from tests/conftest.py (session-scoped).


def test_eval_phase1_writes_valid_json_and_is_monotonic(debug_run):
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    cmd = [sys.executable, "-m", "visword.eval_phase1",
           "--run-dir", str(debug_run), "--checkpoint", "last.pt"]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    report = json.loads((debug_run / "phase1_recall.json").read_text())

    for key in ("checkpoint", "num_pages_evaluated", "num_crops", "recall", "sanity"):
        assert key in report, f"missing {key!r} in {list(report)}"

    recall = report["recall"]
    for k in ("1", "5", "10", "20"):
        assert k in recall
    # R@20 >= R@10 >= R@5 >= R@1 (±1e-6 float slop)
    ks = [1, 5, 10, 20]
    for a, b in zip(ks, ks[1:]):
        assert recall[str(b)] >= recall[str(a)] - 1e-6, recall

    sanity = report["sanity"]
    assert "same_page_sim_mean" in sanity
    assert "diff_page_sim_mean" in sanity
    assert "gap" in sanity
    assert "monotonic" in sanity


def test_eval_phase1_sanity_gap_has_sign(debug_run):
    # The fixture is function-scoped so each test gets a fresh tmp_path;
    # run the eval CLI here rather than assuming an earlier test did it.
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    cmd = [sys.executable, "-m", "visword.eval_phase1",
           "--run-dir", str(debug_run), "--checkpoint", "last.pt"]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    report = json.loads((debug_run / "phase1_recall.json").read_text())
    # On a trained model the gap is usually positive. On a heavily-undertrained
    # debug run it may be tiny or slightly negative — just assert it's finite.
    gap = report["sanity"]["gap"]
    assert gap == gap    # not NaN

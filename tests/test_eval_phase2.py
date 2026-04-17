"""Phase D / D2 tests for the Phase-2 eval CLI (TESTS.md D2).

Needs both the wiki_ss and wiki_ss_anchors caches populated — D1's
fixture gives us the trained run dir; the anchors cache must exist
under $DATA_DIR/wiki_ss_anchors.
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


def test_eval_phase2_writes_valid_json_with_recall_monotonic(debug_run):
    data_dir = os.environ.get("DATA_DIR") or str(PROJECT_ROOT / "data")
    triplets = Path(data_dir) / "wiki_ss_anchors" / "triplets_val.jsonl"
    if not triplets.exists():
        pytest.skip(f"anchors cache missing: {triplets}")

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src"), "DATA_DIR": data_dir}
    cmd = [sys.executable, "-m", "visword.eval_phase2",
           "--run-dir", str(debug_run),
           "--checkpoint", "last.pt",
           "--max-triplets", "8"]      # tiny smoke — full 571 triplets take minutes
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    report = json.loads((debug_run / "phase2_recall.json").read_text())
    for key in ("checkpoint", "num_triplets", "num_anchors",
                "num_pool_images", "recall", "sanity"):
        assert key in report, f"missing {key}"

    assert report["num_triplets"] > 0

    ks = [1, 5, 10, 20]
    recall = report["recall"]
    for a, b in zip(ks, ks[1:]):
        assert recall[str(b)] >= recall[str(a)] - 1e-6, recall

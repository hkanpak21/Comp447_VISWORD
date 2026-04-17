"""Phase C / C2 — training aborts loudly on an empty / missing cache.

``HF_HUB_OFFLINE=1`` is already set by ``visword.train`` at import time;
this test confirms that without a prefetched cache, training does NOT
silently try to download. It must fail fast with a message that points
the operator at ``scripts/prefetch_data.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_aborts_on_missing_cache(tmp_path):
    empty_data_dir = tmp_path / "empty-data"
    empty_data_dir.mkdir()

    env = {
        **os.environ,
        "DATA_DIR": str(empty_data_dir),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }
    cmd = [
        sys.executable, "-m", "visword.train",
        "--config", "configs/debug.yaml",
        "--runs-root", str(tmp_path / "runs"),
        "--set", "data.num_train_samples=4",
        "--set", "data.num_eval_samples=2",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)

    assert result.returncode != 0, (
        f"train should have failed, stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = (result.stdout + "\n" + result.stderr).lower()
    assert "prefetch" in combined, (
        f"error message must point to prefetch script; got:\n{result.stderr}"
    )

    # No run dir should have been created either.
    runs_root = tmp_path / "runs"
    if runs_root.exists():
        assert not list(runs_root.iterdir()), f"empty-cache run dir created: {list(runs_root.iterdir())}"

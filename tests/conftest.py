"""Shared pytest setup + cross-file fixtures."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Session-scoped debug-training fixture — reused by Phase C and Phase D tests.
# Training ~1 min on a T4; without session scope we'd retrain 8+ times per run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def debug_run(tmp_path_factory) -> Path:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    data_dir = os.environ.get("DATA_DIR") or str(PROJECT_ROOT / "data")
    cache_manifest = Path(data_dir) / "wiki_ss" / "manifest.json"
    if not cache_manifest.exists():
        pytest.skip(f"no cache at {cache_manifest}; run scripts/submit.sh prefetch first")

    runs_root = tmp_path_factory.mktemp("runs")
    env = {**os.environ, "DATA_DIR": data_dir, "PYTHONPATH": str(SRC)}
    cmd = [
        sys.executable, "-m", "visword.train",
        "--config", "configs/debug.yaml",
        "--runs-root", str(runs_root),
        "--run-name", "phaseC-test",
        "--set", "data.num_train_samples=32",
        "--set", "data.num_eval_samples=10",
        "--set", "train.batch_size=8",
        "--set", "train.k_per_page=2",
        "--set", "train.epochs=1",
        "--set", "train.eval_every_steps=2",
        "--set", "train.num_diag_batches=1",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"debug training exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1, run_dirs
    return run_dirs[0]

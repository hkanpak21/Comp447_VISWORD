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


@pytest.fixture
def debug_run(tmp_path):
    data_dir = os.environ.get("DATA_DIR") or str(PROJECT_ROOT / "data")
    cache_manifest = Path(data_dir) / "wiki_ss" / "manifest.json"
    if not cache_manifest.exists():
        pytest.skip(f"no cache at {cache_manifest}; run scripts/submit.sh prefetch first")

    runs_root = tmp_path / "runs"
    env = {**os.environ, "DATA_DIR": data_dir, "PYTHONPATH": str(PROJECT_ROOT / "src")}
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
        f"train exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1, run_dirs
    return run_dirs[0]


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


def test_no_file_written_outside_run_dir(debug_run, tmp_path):
    """Walk tmp_path/runs and confirm nothing landed outside."""
    runs_root = tmp_path / "runs"
    # Every path under runs_root is either the run dir or under it.
    for path in runs_root.rglob("*"):
        assert debug_run in path.parents or path == debug_run or path == runs_root, path

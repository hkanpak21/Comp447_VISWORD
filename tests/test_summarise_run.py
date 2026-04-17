"""Cross-cutting / X3 — summarise_run output fields (TESTS.md X3).

Runs against a debug run (needs GPU to populate). CPU-only smoke check
is also provided against a hand-rolled run-dir fixture.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "fake-run"
    run.mkdir(parents=True)

    (run / "provenance.json").write_text(json.dumps({
        "visword_git_sha": "a1b2c3d4",
        "salad_vendor_sha": "6aede13a3f6c25750bf7fde10209c06cb73060bb",
        "python": "3.9.25",
        "torch": "2.3.0+cu121",
        "cuda": "12.1",
        "gpu": "Tesla T4",
        "hostname": "ai01",
        "slurm_job_id": "999999",
        "config_hash": "deadbeef",
        "data_fingerprint": "123456789abcdef0",
        "created_at": "2026-04-17T10:00:00Z",
    }))
    (run / "config.resolved.yaml").write_text(
        "experiment_name: visword-test\nmodel_kind: salad\n"
    )
    (run / "metrics.jsonl").write_text("\n".join([
        json.dumps({"step": 0, "epoch": 0, "loss": 0.8, "top1_acc": 0.12,
                    "wall_time_s": 0.5}),
        json.dumps({"step": 1, "epoch": 0, "loss": 0.7, "top1_acc": 0.25,
                    "wall_time_s": 1.1}),
        json.dumps({"eval_step": 1, "phase1_recall@10": 0.42}),
    ]))
    (run / "phase1_recall.json").write_text(json.dumps({
        "checkpoint": "checkpoints/last.pt",
        "num_pages_evaluated": 10,
        "num_crops": 40,
        "recall": {"1": 0.1, "5": 0.3, "10": 0.55, "20": 0.8},
        "sanity": {"same_page_sim_mean": 0.5, "diff_page_sim_mean": 0.1,
                   "gap": 0.4, "monotonic": True},
    }))
    return run


def test_summarise_contains_required_fields(tmp_path):
    run = _make_fake_run(tmp_path)
    env = {"PYTHONPATH": str(PROJECT_ROOT / "src"),
           "PATH": __import__("os").environ.get("PATH", "")}
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "summarise_run.py"), str(run)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    text = result.stdout

    # TESTS.md X3: output must contain experiment_name, final loss,
    # final recall@10, SALAD SHA, git SHA, wall time.
    for needle in (
        "visword-test",                                            # experiment_name
        "0.7",                                                     # final loss value
        "0.55",                                                    # R@10 from phase1_recall
        "6aede13a3f6c25750bf7fde10209c06cb73060bb",                # SALAD vendor SHA
        "a1b2c3d4",                                                # visword git SHA (short)
        "1.1",                                                     # wall time in s
    ):
        assert needle in text, f"missing {needle!r} in summarise output:\n{text}"

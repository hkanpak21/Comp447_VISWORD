"""Integration: ``python -m visword.interpret --run-dir …`` produces every §8 artefact.

Depends on the shared ``debug_run`` fixture (session-scoped training
run). Validates the single driver that ``slurm/eval.sbatch`` calls in
production, so if this passes the full eval SLURM job will too.
"""
from __future__ import annotations

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


def test_interpret_driver_writes_all_artefacts(debug_run):
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    cmd = [sys.executable, "-m", "visword.interpret",
           "--run-dir", str(debug_run),
           "--checkpoint", "last.pt",
           "--k", "2"]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"interpret driver failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    interpret_dir = debug_run / "interpret"
    must_exist = [
        "salad_hooks.json",
        "attention_sample0.png",
        "attention_sample0.json",
        "attention_sample1.png",
        "salad_clusters_sample0.png",
        "dustbin_map_sample0.png",
        "cls_vs_vlad.json",
        "cls_vs_vlad.png",
        "dustbin_evolution.png",
    ]
    missing = [p for p in must_exist if not (interpret_dir / p).exists()]
    assert not missing, f"missing artefacts: {missing}"

    # At least one patch-neighbours triplet dir with a PNG inside.
    triplets = list(interpret_dir.glob("patch_triplet_*"))
    assert triplets, "no patch_triplet_* subdirectories produced"
    assert any(p.stat().st_size > 0 for p in triplets[0].glob("*.png")), (
        f"triplet dir {triplets[0]} has no readable PNGs"
    )

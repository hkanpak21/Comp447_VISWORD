"""Ticket 04 — MAE body-target reader: model contract + "a run finishes" smoke.

Marked integration (loads MAE/BERT weights, the one-step run needs the data cache + GPU).
Directly guards the prior "training never finished" failure mode.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"


@pytest.mark.integration
def test_mae_reader_forward_unitnorm_and_param_efficient() -> None:
    import torch
    from visword.models.mae_reader import MAEBodyReader

    r = MAEBodyReader(num_trainable_blocks=2).eval()
    x = torch.randn(2, 3, 224, 224)
    z = r(x)
    assert z.shape == (2, 768) and r.descriptor_dim == 768
    assert torch.allclose(z.norm(p=2, dim=-1), torch.ones(2), atol=1e-4)
    n_train = sum(p.numel() for p in r.trainable_parameters())
    n_tot = sum(p.numel() for p in r.parameters())
    assert 0 < n_train < n_tot, "only head + last-N blocks should train"


@pytest.mark.integration
def test_mae_text_training_one_run_finishes(tmp_path) -> None:
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    data_dir = os.environ.get("DATA_DIR") or str(PROJECT_ROOT / "data")
    cache = Path(data_dir) / "wiki_ss"
    if not (cache / "manifest.json").exists():
        pytest.skip("no wiki_ss cache")

    out = tmp_path / "maerun"
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    cmd = [
        sys.executable, "-m", "visword.train_mae_text",
        "--cache-dir", str(cache), "--out", str(out),
        "--num-train", "8", "--num-eval", "8", "--eval-pages", "4",
        "--max-crops-per-page", "2", "--eval-max-crops", "2",
        "--epochs", "1", "--batch-pages", "2", "--num-workers", "0",
        "--eval-every-steps", "2", "--max-text-tokens", "64",
    ]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"train exited {res.returncode}\n{res.stdout}\n{res.stderr}"
    assert (out / "config.resolved.json").exists()
    assert (out / "checkpoints" / "last.pt").exists()
    assert (out / "final_eval.json").exists()
    assert json.loads((out / "config.resolved.json").read_text())["text_source"] == "body"
    rows = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x.strip()]
    assert any("loss" in r for r in rows), "no training loss logged"

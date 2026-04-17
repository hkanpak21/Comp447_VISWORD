"""Cross-cutting / X1 — ``visword.train`` succeeds with no network (TESTS.md X1).

Runs in a subprocess with a ``sitecustomize.py`` that replaces
``socket.socket``'s constructor with one that raises. If the training
path really is offline (HF_HUB_OFFLINE=1, DINOv2 hub cache populated,
manifest + fingerprint verified locally), it should still complete on
the real GPU.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_completes_with_socket_raise(tmp_path):
    data_dir = os.environ.get("DATA_DIR") or str(PROJECT_ROOT / "data")
    if not (Path(data_dir) / "wiki_ss" / "manifest.json").exists():
        pytest.skip(f"no cache at {data_dir}/wiki_ss")

    # sitecustomize.py preloaded ahead of everything else makes every
    # socket.socket() construction raise — any would-be network call
    # becomes visible as an exception.
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(textwrap.dedent("""\
        import socket

        class _BlockedSocket:
            def __init__(self, *a, **kw):
                raise OSError('socket disabled by test_no_internet_at_runtime')

        socket.socket = _BlockedSocket
    """))

    runs_root = tmp_path / "runs"
    env = {
        **os.environ,
        "DATA_DIR": data_dir,
        "PYTHONPATH": f"{tmp_path}:{PROJECT_ROOT / 'src'}",    # sitecustomize wins
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    cmd = [
        sys.executable, "-m", "visword.train",
        "--config", "configs/debug.yaml",
        "--runs-root", str(runs_root),
        "--run-name", "no-internet",
        "--set", "data.num_train_samples=16",
        "--set", "data.num_eval_samples=4",
        "--set", "train.batch_size=4",
        "--set", "train.k_per_page=2",
        "--set", "train.epochs=1",
        "--set", "train.eval_every_steps=2",
        "--set", "train.num_diag_batches=1",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"training failed under socket-block:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

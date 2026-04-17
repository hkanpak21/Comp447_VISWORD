"""Phase D / D3 — surface decoupling between eval and train (TESTS.md D3).

Running the Phase-1 or Phase-2 eval must not pull ``visword.train`` into
``sys.modules``. Guards against accidentally reaching for training-side
helpers in the eval modules.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = []

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _import_and_dump_modules(module_name: str) -> list[str]:
    """Fresh subprocess: import ``module_name``, print all sys.modules keys."""
    script = (
        "import sys; "
        f"import {module_name}; "
        "print('\\n'.join(sorted(sys.modules.keys())))"
    )
    env = {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT, env={**env, **{k: v for k, v in __import__("os").environ.items()
                                         if k.startswith(("PATH", "HOME", "LANG", "LD_", "CONDA"))}},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_eval_phase1_does_not_import_train():
    loaded = _import_and_dump_modules("visword.eval_phase1")
    assert "visword.train" not in loaded, (
        "visword.eval_phase1 transitively imported visword.train — break the coupling"
    )


def test_eval_phase2_does_not_import_train():
    loaded = _import_and_dump_modules("visword.eval_phase2")
    assert "visword.train" not in loaded, (
        "visword.eval_phase2 transitively imported visword.train — break the coupling"
    )

"""Phase C / C3 — run-dir contract check (TESTS.md C3 / PROJECT_SPEC.md §11).

Relies on the fixture from test_train_debug.py producing a run dir. This
test just enumerates the §11 contract and asserts each entry is present.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


# debug_run fixture is auto-loaded from tests/conftest.py (session-scoped).


SPEC_SECTION_11_FILES = [
    "config.resolved.yaml",
    "provenance.json",
    "metrics.jsonl",
    "train_curves.png",
    "diagnostics/untrained_batch_stats.json",
    "checkpoints/last.pt",
    "checkpoints/best_phase1.pt",
]


def test_all_section_11_artifacts_present(debug_run):
    missing = [p for p in SPEC_SECTION_11_FILES if not (debug_run / p).exists()]
    assert not missing, f"run-dir contract breach — missing: {missing}"


def test_provenance_json_has_required_fields(debug_run):
    import json
    prov = json.loads((debug_run / "provenance.json").read_text())
    for key in ("visword_git_sha", "salad_vendor_sha", "python",
                "torch", "cuda", "gpu", "hostname", "config_hash",
                "data_fingerprint", "created_at"):
        assert key in prov, f"provenance.json missing {key}"


def test_config_resolved_yaml_parses(debug_run):
    import yaml
    data = yaml.safe_load((debug_run / "config.resolved.yaml").read_text())
    assert data["experiment_name"] == "visword-debug"
    assert data["model_kind"] in ("salad", "cls")

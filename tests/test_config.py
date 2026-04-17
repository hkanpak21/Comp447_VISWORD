"""Phase A tests A4 — config merge + canonical hash (TESTS.md A4)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visword.config import Config, config_hash  # noqa: E402


def _resolve_via_cli(named: str, overrides: list[str]) -> dict:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "resolve_config.py"), named]
    for ov in overrides:
        cmd += ["--set", ov]
    out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True, env={
        **__import__("os").environ,
        "DATA_DIR": str(PROJECT_ROOT / "data"),
    })
    return yaml.safe_load(out)


def test_config_merge() -> None:
    resolved = _resolve_via_cli("configs/debug.yaml", ["train.epochs=2"])

    # debug.yaml override + --set override + default fallback all visible:
    assert resolved["experiment_name"] == "visword-debug"  # from debug.yaml
    assert resolved["train"]["epochs"] == 2  # from --set, overrides debug.yaml's 1
    assert resolved["train"]["batch_size"] == 8  # from debug.yaml
    assert resolved["train"]["loss"] == "multisim"  # from default.yaml
    assert resolved["data"]["num_train_samples"] == 500  # debug.yaml override
    assert resolved["data"]["num_eval_samples"] == 50

    # ${DATA_DIR} expansion happened.
    assert "${" not in resolved["data"]["wiki_ss_cache_dir"]


def test_config_hash_deterministic(tmp_path: Path) -> None:
    base = {
        "experiment_name": "x",
        "data": {
            "wiki_ss_cache_dir": "/tmp/a",
            "anchors_cache_dir": "/tmp/b",
            "num_train_samples": 10,
            "num_eval_samples": 5,
        },
    }
    cfg1 = Config.model_validate(base)
    cfg2 = Config.model_validate(base)
    assert config_hash(cfg1) == config_hash(cfg2)

    # Identical content but YAML-dumped with shuffled top-level keys → same hash
    # (canonical_dump sorts keys before hashing).
    shuffled = {k: base[k] for k in reversed(list(base.keys()))}
    cfg3 = Config.model_validate(shuffled)
    assert config_hash(cfg3) == config_hash(cfg1)

    # Different content → different hash.
    diff = json.loads(json.dumps(base))
    diff["data"]["num_train_samples"] = 11
    cfg4 = Config.model_validate(diff)
    assert config_hash(cfg4) != config_hash(cfg1)

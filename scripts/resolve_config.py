#!/usr/bin/env python
"""Merge default ← named config ← --set k=v overrides; print resolved YAML.

Usage::

    python scripts/resolve_config.py configs/salad_main.yaml \
        [--set train.epochs=5 train.lr_head=1e-3] \
        [--print-hash]

Algorithm:

1. Load ``configs/default.yaml``.
2. Deep-merge the named config (CLI positional arg) on top.
3. Apply each ``--set key.path=value`` (parsed via YAML so ints/floats/bools work).
4. Recursively expand ``${VAR}`` substrings via ``visword.paths.expand_env``.
5. Validate as ``visword.config.Config`` and emit canonical YAML to stdout.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from visword.config import Config, config_hash  # noqa: E402
from visword.paths import expand_env  # noqa: E402


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def apply_set_override(d: dict[str, Any], assignment: str) -> None:
    if "=" not in assignment:
        raise SystemExit(f"--set expects key.path=value, got {assignment!r}")
    key_path, raw = assignment.split("=", 1)
    value = yaml.safe_load(raw)
    parts = key_path.split(".")
    cursor: Any = d
    for p in parts[:-1]:
        cursor = cursor.setdefault(p, {})
    cursor[parts[-1]] = value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def resolve(named_path: Path, overrides: list[str]) -> dict[str, Any]:
    default = load_yaml(HERE / "configs" / "default.yaml")
    named = load_yaml(named_path)
    merged = deep_merge(default, named)
    for ov in overrides:
        apply_set_override(merged, ov)
    return expand_env(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="named config under configs/")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--print-hash", action="store_true")
    args = parser.parse_args()

    resolved = resolve(args.config, args.overrides)
    cfg = Config.model_validate(resolved)

    if args.print_hash:
        print(config_hash(cfg))
        return

    yaml.safe_dump(cfg.model_dump(mode="json"), sys.stdout, sort_keys=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Pretty-print a run directory (PROJECT_SPEC.md §15/Phase D). Phase A stub."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    args = p.parse_args(argv)

    rd = args.run_dir
    if not rd.exists():
        raise SystemExit(f"no such run dir: {rd}")

    prov = rd / "provenance.json"
    cfg = rd / "config.resolved.yaml"
    phase1 = rd / "phase1_recall.json"
    phase2 = rd / "phase2_recall.json"

    print(f"# {rd}")
    if prov.exists():
        print("provenance:")
        print(json.dumps(json.loads(prov.read_text()), indent=2))
    if cfg.exists():
        print("\nconfig:")
        print(cfg.read_text())
    for label, path in (("phase1", phase1), ("phase2", phase2)):
        if path.exists():
            print(f"\n{label} recall:")
            print(json.dumps(json.loads(path.read_text()), indent=2))

    print("\n(summarise_run full renderer lands in Phase D)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Pin the DINOv2 torch.hub cache to a Python 3.9-compatible commit.

Background
----------
``third_party/salad/models/backbones/dinov2.py`` calls
``torch.hub.load('facebookresearch/dinov2', model_name)`` with no ``ref``,
so torch.hub fetches the latest ``main`` of ``facebookresearch/dinov2``.
Since late 2024 that branch uses PEP 604 union syntax (``float | None``),
which is a Python 3.10+ feature and breaks our Python 3.9 env (cloned from
``he_ofl``).

We work around this by pre-populating the hub cache directory
``~/.cache/torch/hub/facebookresearch_dinov2_main/`` with a checkout of an
older commit that still uses ``Optional[...]`` annotations. torch.hub will
not re-fetch when the cache is already present.

Default pinned commit: ``e1277af2...`` ("Fix interpolation of positional
embeddings", #378) — last pre-PEP-604 commit on main.

Run this script on the login node (internet) once per env. Rerun when
``~/.cache/torch/hub/`` is cleared or after updating to a newer env.

Usage::

    python scripts/ensure_dinov2_hub.py                  # use default pin
    python scripts/ensure_dinov2_hub.py --commit <sha>   # override
    python scripts/ensure_dinov2_hub.py --force          # re-pin even if cached
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_COMMIT = "e1277af2bb80e1fae53751d38b1eb6a4ce9b48ed"   # pre-PEP-604 on facebookresearch/dinov2 main
REMOTE = "https://github.com/facebookresearch/dinov2"
HUB_CACHE = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commit", default=DEFAULT_COMMIT,
                   help=f"DINOv2 commit to pin (default: {DEFAULT_COMMIT[:12]})")
    p.add_argument("--force", action="store_true",
                   help="Replace the hub cache even if it already exists.")
    args = p.parse_args(argv)

    if HUB_CACHE.exists() and not args.force:
        print(f"hub cache already exists: {HUB_CACHE}")
        print("pass --force to re-pin")
        return 0

    # git must be on PATH. On Valar this means `module load git/2.9.5`.
    if not shutil.which("git"):
        print("ERROR: git not on PATH. Run `module load git/2.9.5` first.", file=sys.stderr)
        return 2

    HUB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if HUB_CACHE.exists():
        shutil.rmtree(HUB_CACHE)

    print(f"cloning {REMOTE} → {HUB_CACHE} @ {args.commit[:12]} …")
    subprocess.run(["git", "clone", "--quiet", REMOTE, str(HUB_CACHE)], check=True)
    subprocess.run(["git", "-C", str(HUB_CACHE), "checkout", "--quiet", args.commit], check=True)
    shutil.rmtree(HUB_CACHE / ".git")
    print(f"done — hub cache pinned at {args.commit[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

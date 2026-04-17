#!/usr/bin/env python
"""CLI wrapper around ``visword.data.prefetch.main`` (PROJECT_SPEC.md §4.3)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from visword.data.prefetch import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

"""Append-only JSONL logger (PROJECT_SPEC.md §6 / §11).

Motivation: we don't run an experiment tracker (§1.4). Every structured
event is a JSON line; ``metrics.jsonl`` is authoritative — plots and
summaries read from it, nothing else.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Context-manager-friendly JSONL writer.

    Writes are flushed after every line so a killed job still leaves
    readable metrics. ``elapsed_s`` is appended to every row if
    ``with_elapsed=True`` (default).
    """

    def __init__(
        self,
        path: Path,
        *,
        with_elapsed: bool = True,
        buffering: int = 1,   # line-buffered
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: io.TextIOWrapper | None = open(self.path, "a", buffering=buffering)
        self._t0 = time.time()
        self.with_elapsed = with_elapsed

    def log(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlLogger is closed")
        if self.with_elapsed:
            record = {**record, "elapsed_s": round(time.time() - self._t0, 3)}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file; skip blank lines."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

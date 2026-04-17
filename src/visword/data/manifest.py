"""Manifest writer / reader / fingerprint for the prefetched data cache.

Layout (PROJECT_SPEC.md §4.2):

    <cache_dir>/manifest.json            # final manifest
    <cache_dir>/manifest.json.partial    # in-progress (overwritten every N rows)
    <cache_dir>/.fingerprint             # sha256 over canonical sorted rows
    <cache_dir>/blobs/{idx//1000:02d}/{idx:07d}.png

A row is a dict::

    {
        "idx": int,
        "docid": str,
        "title": str,
        "text_path": str,        # relative to cache_dir; may be ""
        "image_path": str,       # relative to cache_dir
        "image_sha256": str,
    }

The fingerprint is sha256 over the canonical-JSON rendering of the rows sorted
by ``idx``. Tampering with any byte of any blob → fingerprint mismatch on
re-verification (because ``image_sha256`` recomputes from disk in
``verify_fingerprint``).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_ROW_KEYS = ("idx", "docid", "title", "text_path", "image_path", "image_sha256")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        ({k: r[k] for k in REQUIRED_ROW_KEYS} for r in rows),
        key=lambda r: r["idx"],
    )


def compute_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    """Deterministic sha256 over canonical-sorted rows."""
    payload = json.dumps(_canonical_rows(rows), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def write_manifest(
    cache_dir: Path,
    *,
    dataset: str,
    hf_revision: str | None,
    rows: Iterable[Mapping[str, Any]],
    prefetched_at: str | None = None,
) -> Path:
    """Write the final ``manifest.json`` and ``.fingerprint`` atomically.

    Returns the manifest path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows_canon = _canonical_rows(rows)
    manifest = {
        "dataset": dataset,
        "hf_revision": hf_revision,
        "prefetched_at": prefetched_at or _now_utc_iso(),
        "num_rows": len(rows_canon),
        "rows": rows_canon,
    }
    manifest_path = cache_dir / "manifest.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    _atomic_write_text(cache_dir / ".fingerprint", compute_fingerprint(rows_canon))
    return manifest_path


def read_manifest(cache_dir: Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    return json.loads((cache_dir / "manifest.json").read_text())


def verify_fingerprint(cache_dir: Path) -> bool:
    """Recompute the fingerprint from on-disk blobs and compare to ``.fingerprint``.

    Each row's ``image_sha256`` is recomputed from the actual blob bytes, then
    the canonical fingerprint is recomputed from the resulting rows. This
    detects: (a) a stored ``.fingerprint`` not matching the manifest rows, and
    (b) blob bytes that have been mutated since prefetch time.
    """
    cache_dir = Path(cache_dir)
    stored = (cache_dir / ".fingerprint").read_text().strip()
    manifest = read_manifest(cache_dir)
    rebuilt_rows: list[dict[str, Any]] = []
    for row in manifest["rows"]:
        blob_path = cache_dir / row["image_path"]
        actual_sha = hashlib.sha256(blob_path.read_bytes()).hexdigest()
        rebuilt_rows.append({**row, "image_sha256": actual_sha})
    return compute_fingerprint(rebuilt_rows) == stored


# ----- partial-manifest helpers used by prefetch resume ----------------------

def partial_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "manifest.json.partial"


def load_partial(cache_dir: Path) -> dict[str, Any] | None:
    p = partial_path(cache_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_partial(
    cache_dir: Path,
    *,
    dataset: str,
    hf_revision: str | None,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Atomically dump in-progress rows to ``manifest.json.partial``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows_canon = _canonical_rows(rows)
    payload = {
        "dataset": dataset,
        "hf_revision": hf_revision,
        "prefetched_at": _now_utc_iso(),
        "num_rows": len(rows_canon),
        "rows": rows_canon,
    }
    p = partial_path(cache_dir)
    _atomic_write_text(p, json.dumps(payload, indent=2))
    return p


def finalize_partial(cache_dir: Path) -> Path:
    """Promote ``manifest.json.partial`` → ``manifest.json`` and write fingerprint."""
    cache_dir = Path(cache_dir)
    p = partial_path(cache_dir)
    data = json.loads(p.read_text())
    write_manifest(
        cache_dir,
        dataset=data["dataset"],
        hf_revision=data.get("hf_revision"),
        rows=data["rows"],
        prefetched_at=data.get("prefetched_at"),
    )
    p.unlink()
    return cache_dir / "manifest.json"

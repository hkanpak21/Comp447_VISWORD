"""Prefetch HuggingFace datasets into the local cache (PROJECT_SPEC.md §4).

Two dataset modes:

- ``wiki-ss``: stream ``Tevatron/wiki-ss-corpus``, re-encode each image as PNG
  (compress_level=1) into ``blobs/{idx//1000:02d}/{idx:07d}.png``, capture
  text into ``texts/...`` and write ``manifest.json`` + ``.fingerprint``.
  Resume-safe: partial manifest updated every 500 rows, blob writes atomic.

- ``wiki-ss-anchors``: ``snapshot_download`` of ``hkanpak21/Wikipedia_SS_withanchors``
  with ``allow_patterns=["images/*", "splits.json", "*.json", "*.jsonl"]``;
  copy/move into the cache verbatim (the original repo's structure is the
  contract per CONTEXT.md week 1 lessons — never re-render placeholders).

Exit codes (only ``main`` honours these):
- 0 → ``--target-rows`` met (or all available, if smaller).
- 2 → partial completion (some rows skipped after 3 retries, or stream exhausted early).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from . import manifest as M


logger = logging.getLogger("visword.prefetch")


def _ensure_logger() -> None:
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _encode_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def _row_paths(idx: int) -> tuple[str, str]:
    shard = f"{idx // 1000:02d}"
    return (
        f"blobs/{shard}/{idx:07d}.png",
        f"texts/{shard}/{idx:07d}.txt",
    )


def _load_existing_rows(cache_dir: Path) -> tuple[list[dict], int]:
    """Return any rows already on disk (from manifest or partial) and next idx."""
    final = cache_dir / "manifest.json"
    if final.exists():
        rows = M.read_manifest(cache_dir)["rows"]
        return list(rows), max((r["idx"] for r in rows), default=-1) + 1
    partial = M.load_partial(cache_dir)
    if partial is not None:
        return list(partial["rows"]), max((r["idx"] for r in partial["rows"]), default=-1) + 1
    return [], 0


def _stream_wiki_ss(start_idx: int) -> Iterator[tuple[int, dict]]:
    from datasets import load_dataset

    ds = load_dataset("Tevatron/wiki-ss-corpus", split="train", streaming=True)
    for offset, row in enumerate(ds):
        idx = offset
        if idx < start_idx:
            continue
        yield idx, row


def _process_row(
    idx: int,
    row: dict[str, Any],
    cache_dir: Path,
    *,
    max_retries: int = 3,
) -> tuple[dict, int] | None:
    """Encode + write a row. Returns (manifest_row, bytes_written) or None on failure."""
    image_rel, text_rel = _row_paths(idx)
    image_path = cache_dir / image_rel
    text_path = cache_dir / text_rel

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            png = _encode_png(row["image"])
            _atomic_write_bytes(image_path, png)
            text_bytes = (row.get("text") or "").encode("utf-8")
            _atomic_write_bytes(text_path, text_bytes)
            sha = hashlib.sha256(png).hexdigest()
            return (
                {
                    "idx": idx,
                    "docid": str(row.get("docid", "")),
                    "title": str(row.get("title", "")),
                    "text_path": text_rel,
                    "image_path": image_rel,
                    "image_sha256": sha,
                },
                len(png) + len(text_bytes),
            )
        except Exception as exc:
            last_exc = exc
            sleep_for = 2.0 ** attempt
            logger.warning(
                "row %d attempt %d/%d failed (%s); retrying in %.1fs",
                idx, attempt + 1, max_retries, exc, sleep_for,
            )
            time.sleep(sleep_for)
    logger.error("row %d failed after %d retries: %s", idx, max_retries, last_exc)
    return None


# ---------------------------------------------------------------------------
# wiki-ss prefetch driver
# ---------------------------------------------------------------------------

def prefetch_wiki_ss(
    cache_dir: Path,
    target_rows: int,
    *,
    resume: bool = True,
    partial_every: int = 500,
) -> dict[str, Any]:
    """Run the wiki-ss-corpus prefetch.

    Returns a summary dict with keys ``rows_requested``, ``rows_written``,
    ``rows_failed``, ``total_bytes``, ``elapsed_seconds``, ``status``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_rows, next_idx = _load_existing_rows(cache_dir) if resume else ([], 0)
    if existing_rows and (cache_dir / "manifest.json").exists() and len(existing_rows) >= target_rows:
        logger.info("manifest.json already has %d rows >= target %d; nothing to do",
                    len(existing_rows), target_rows)
        return {
            "rows_requested": target_rows,
            "rows_written": len(existing_rows),
            "rows_failed": 0,
            "total_bytes": 0,
            "elapsed_seconds": 0.0,
            "status": "already_complete",
        }

    rows: list[dict] = list(existing_rows)
    if rows:
        logger.info("resuming from idx %d (%d rows already on disk)", next_idx, len(rows))

    failed = 0
    bytes_written = 0
    started = time.time()

    for idx, row in _stream_wiki_ss(start_idx=next_idx):
        if len(rows) >= target_rows:
            break
        result = _process_row(idx, row, cache_dir)
        if result is None:
            failed += 1
            continue
        manifest_row, nb = result
        rows.append(manifest_row)
        bytes_written += nb
        if len(rows) % partial_every == 0:
            M.write_partial(
                cache_dir, dataset="Tevatron/wiki-ss-corpus", hf_revision=None, rows=rows
            )
            logger.info("checkpointed partial manifest at %d rows", len(rows))

    elapsed = time.time() - started
    summary = {
        "rows_requested": target_rows,
        "rows_written": len(rows),
        "rows_failed": failed,
        "total_bytes": bytes_written,
        "elapsed_seconds": round(elapsed, 2),
    }

    if len(rows) >= target_rows:
        # Always rewrite the partial first so finalize has the latest state, then promote.
        M.write_partial(
            cache_dir, dataset="Tevatron/wiki-ss-corpus", hf_revision=None, rows=rows[:target_rows]
        )
        M.finalize_partial(cache_dir)
        summary["status"] = "complete"
        logger.info("DONE — %s", summary)
        return summary

    # Stream exhausted before target — leave the partial for the next resume.
    M.write_partial(
        cache_dir, dataset="Tevatron/wiki-ss-corpus", hf_revision=None, rows=rows
    )
    summary["status"] = "partial"
    logger.error("PARTIAL — wrote %d of requested %d rows (%d failed). %s",
                 len(rows), target_rows, failed, summary)
    return summary


# ---------------------------------------------------------------------------
# anchors prefetch driver
# ---------------------------------------------------------------------------

def prefetch_anchors(cache_dir: Path) -> dict[str, Any]:
    """Snapshot the anchors repo into ``cache_dir`` (CONTEXT.md week 1 lesson)."""
    from huggingface_hub import snapshot_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    snap_path = snapshot_download(
        repo_id="hkanpak21/Wikipedia_SS_withanchors",
        repo_type="dataset",
        allow_patterns=["images/*", "splits.json", "*.json", "*.jsonl"],
    )
    snap = Path(snap_path)
    # Copy into the canonical cache dir so training jobs only need DATA_DIR.
    for entry in snap.iterdir():
        target = cache_dir / entry.name
        if target.exists():
            continue
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)
    elapsed = time.time() - started
    summary = {
        "snapshot_source": str(snap),
        "cache_dir": str(cache_dir),
        "elapsed_seconds": round(elapsed, 2),
        "status": "complete",
    }
    logger.info("anchors snapshot OK — %s", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prefetch VisWord datasets to the local cache.")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=["wiki-ss", "wiki-ss-anchors"])
    p.add_argument("--target-rows", type=int, default=21000,
                   help="Only used for --dataset wiki-ss.")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p


def main(argv: list[str] | None = None) -> int:
    _ensure_logger()
    args = build_parser().parse_args(argv)

    sub_dir_name = {"wiki-ss": "wiki_ss", "wiki-ss-anchors": "wiki_ss_anchors"}[args.dataset]
    cache_dir = args.data_dir / sub_dir_name

    if args.dataset == "wiki-ss":
        summary = prefetch_wiki_ss(cache_dir, args.target_rows, resume=args.resume)
        return 0 if summary["status"] in ("complete", "already_complete") else 2

    if args.dataset == "wiki-ss-anchors":
        prefetch_anchors(cache_dir)
        return 0

    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())

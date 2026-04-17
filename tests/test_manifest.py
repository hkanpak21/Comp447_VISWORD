"""Phase A tests A2 — manifest writer/reader/fingerprint (TESTS.md A2)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from visword.data import manifest as M


def _make_blob(cache_dir: Path, idx: int, colour: tuple[int, int, int]) -> dict:
    rel = f"blobs/{idx // 1000:02d}/{idx:07d}.png"
    full = cache_dir / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (8, 8), colour)
    img.save(full, format="PNG")
    return {
        "idx": idx,
        "docid": f"doc-{idx}",
        "title": f"Title {idx}",
        "text_path": "",
        "image_path": rel,
        "image_sha256": hashlib.sha256(full.read_bytes()).hexdigest(),
    }


def test_manifest_roundtrip(tmp_path: Path) -> None:
    rows = [_make_blob(tmp_path, i, (i, 0, 0)) for i in range(3)]
    M.write_manifest(tmp_path, dataset="fake", hf_revision="abc", rows=rows)

    back = M.read_manifest(tmp_path)
    assert back["dataset"] == "fake"
    assert back["hf_revision"] == "abc"
    assert back["num_rows"] == 3
    assert back["rows"] == sorted(rows, key=lambda r: r["idx"])


def test_fingerprint_detects_tampering(tmp_path: Path) -> None:
    rows = [_make_blob(tmp_path, i, (i, 0, 0)) for i in range(3)]
    M.write_manifest(tmp_path, dataset="fake", hf_revision=None, rows=rows)
    assert M.verify_fingerprint(tmp_path) is True

    # Mutate one PNG byte → recomputed fingerprint should mismatch.
    victim = tmp_path / rows[1]["image_path"]
    Image.new("RGB", (8, 8), (255, 255, 255)).save(victim, format="PNG")
    assert M.verify_fingerprint(tmp_path) is False


def test_manifest_partial_resume(tmp_path: Path) -> None:
    # Phase 1: 5/10 rows written + a partial manifest.
    first_five = [_make_blob(tmp_path, i, (i, 0, 0)) for i in range(5)]
    M.write_partial(tmp_path, dataset="fake", hf_revision="rev", rows=first_five)
    partial = M.load_partial(tmp_path)
    assert partial is not None and partial["num_rows"] == 5

    # Phase 2: resume — append rows 5..9 and finalise.
    later_five = [_make_blob(tmp_path, i, (0, 0, i)) for i in range(5, 10)]
    M.write_partial(
        tmp_path,
        dataset="fake",
        hf_revision="rev",
        rows=partial["rows"] + later_five,
    )
    M.finalize_partial(tmp_path)

    final = M.read_manifest(tmp_path)
    assert final["num_rows"] == 10
    # Original rows 0..4 must keep their original sha256.
    by_idx = {r["idx"]: r for r in final["rows"]}
    for r in first_five:
        assert by_idx[r["idx"]]["image_sha256"] == r["image_sha256"]
    assert not M.partial_path(tmp_path).exists()
    assert M.verify_fingerprint(tmp_path) is True

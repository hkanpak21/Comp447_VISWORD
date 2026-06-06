"""Ticket 01 — TextAwareCropper: legible (native-resolution) text-aware cropping.

Tests the NEW cropper that (a) keeps tiles at native resolution (no 2.19x shrink)
and (b) snaps vertical cut boundaries to the whitespace gaps between text lines so
no glyph row is sliced, and rejects blank / fragment tiles. The existing
NonOverlappingCropper and its tests are left untouched (additive).

Synthetic pages are white with solid black horizontal "lines" separated by white
"gaps", so the geometry is known exactly and assertions are deterministic.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

from visword.data.cropper import TextAwareCropper


def striped_page(
    width: int,
    layout: List[Tuple[str, int]],
) -> Tuple[Image.Image, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Build a white page with black horizontal bars per `layout`.

    layout is a list of ("line"|"gap", height) bands stacked top to bottom.
    Returns (image, line_intervals, gap_intervals) as [start, end) row ranges.
    """
    total = sum(h for _, h in layout)
    arr = np.full((total, width, 3), 255, np.uint8)
    lines, gaps = [], []
    y = 0
    for kind, h in layout:
        if kind == "line":
            arr[y : y + h, :, :] = 0
            lines.append((y, y + h))
        else:
            gaps.append((y, y + h))
        y += h
    return Image.fromarray(arr), lines, gaps


def test_inter_line_gaps_found_between_text_lines() -> None:
    # gap(10) line(20) gap(10) line(20) gap(10) line(20) gap(10) -> 100 tall
    img, lines, gaps = striped_page(
        224,
        [("gap", 10), ("line", 20), ("gap", 10), ("line", 20),
         ("gap", 10), ("line", 20), ("gap", 10)],
    )
    cropper = TextAwareCropper(crop_size=224)

    found = cropper.inter_line_gaps(img)  # list of [start, end) whitespace bands
    found_mids = sorted((a + b) // 2 for a, b in found)

    # The two interior gaps (between consecutive lines) must be detected:
    # one centred ~35 (rows 30-40) and one ~65 (rows 60-70).
    assert any(30 <= m <= 40 for m in found_mids), found
    assert any(60 <= m <= 70 for m in found_mids), found
    # No detected "gap" may overlap the centre of a text line.
    line_mids = [(a + b) // 2 for a, b in lines]
    for a, b in found:
        for lm in line_mids:
            assert not (a <= lm < b), f"gap {(a, b)} swallows line centre {lm}"


def test_y_cuts_snap_into_gaps_never_slicing_a_line() -> None:
    # crop_size 50; 5 lines 20px tall separated by 10px gaps -> each tile holds
    # ~1 line, and every boundary must land in whitespace.
    layout = [("gap", 10)]
    for _ in range(5):
        layout += [("line", 20), ("gap", 10)]
    # heights: 10 + 5*(20+10) = 160
    img, lines, gaps = striped_page(224, layout)
    cropper = TextAwareCropper(crop_size=50)

    bands = cropper.y_cuts(img)  # list of (top, bottom) source row ranges

    # Contiguous tiling covering the whole page.
    assert bands[0][0] == 0, bands
    assert bands[-1][1] == img.height, bands
    for (a1, b1), (a2, b2) in zip(bands, bands[1:]):
        assert b1 == a2, bands
    # No band taller than crop_size.
    assert all(b - a <= cropper.crop_size for a, b in bands), bands

    # Every internal boundary sits inside a detected gap and slices no line.
    internal = [b for _, b in bands[:-1]]
    for y in internal:
        assert any(g0 <= y <= g1 for g0, g1 in gaps), f"{y} not in a gap {gaps}"
        for l0, l1 in lines:
            assert not (l0 < y < l1), f"boundary {y} slices line {(l0, l1)}"


def test_crop_is_native_resolution_pixel_exact() -> None:
    # A single-tile, fully-inked page: with target_size == crop_size the returned
    # tile must be the source pixels verbatim — no bilinear shrink anywhere.
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 200, (224, 224, 3), dtype=np.uint8)  # busy, non-white
    img = Image.fromarray(arr)

    cropper = TextAwareCropper(crop_size=224)  # target defaults to crop_size

    tiles = cropper.crop(img)
    assert len(tiles) == 1
    assert tiles[0].size == (224, 224)
    assert np.array_equal(np.asarray(tiles[0]), arr), "tile was resampled, not native"


def test_blank_tiles_dropped_and_all_white_page_falls_back() -> None:
    # One inked crop at the top, then a tall blank region: only the inked band
    # survives; every blank band is dropped.
    arr = np.full((300, 60, 3), 255, np.uint8)
    arr[0:60, :, :] = 0
    img = Image.fromarray(arr)
    cropper = TextAwareCropper(crop_size=60, min_text_ratio=0.05)
    tiles = cropper.crop(img)
    assert len(tiles) == 1, f"blank bands not dropped: {len(tiles)} tiles"
    assert (np.asarray(tiles[0]) < 245).any(-1).mean() > 0.5

    # Fully blank page -> a single padded fallback tile, never zero tiles.
    white = Image.new("RGB", (200, 400), (255, 255, 255))
    fb = cropper.crop(white)
    assert len(fb) == 1
    assert fb[0].size == (60, 60)


def test_every_text_line_lies_within_a_single_band() -> None:
    # No line may straddle a band boundary (the "no fragment / half-line" guarantee).
    layout = [("gap", 10)]
    for _ in range(5):
        layout += [("line", 20), ("gap", 10)]
    img, lines, _ = striped_page(224, layout)
    cropper = TextAwareCropper(crop_size=50)

    bands = cropper.y_cuts(img)
    for l0, l1 in lines:
        assert any(a <= l0 and l1 <= b for a, b in bands), \
            f"line {(l0, l1)} split across bands {bands}"


def test_page_margin_gap_does_not_create_a_sliver_cut() -> None:
    # Leading 5px white margin, then a 48px line (fits crop_size=50) whose trailing
    # gap is just out of reach from row 0. With margins wrongly used as cut points,
    # the margin midpoint (~row 2) becomes the lone in-reach candidate -> a (0,2) sliver.
    img, _, _ = striped_page(
        60, [("gap", 5), ("line", 48), ("gap", 10), ("line", 20), ("gap", 5)]
    )
    cropper = TextAwareCropper(crop_size=50)
    bands = cropper.y_cuts(img)
    # No band may be a sub-line sliver carved out of the page margin.
    assert all((b - a) > cropper.gap_min_height for a, b in bands), bands
    # No internal boundary sits in the leading/trailing page margin.
    for y in (b for _, b in bands[:-1]):
        assert 5 <= y <= 55, f"boundary {y} in page margin; bands={bands}"


def test_line_taller_than_crop_is_hard_cut_into_native_tiles() -> None:
    # A single line taller than crop_size has no interior gap to snap to, so it is
    # hard-cut into crop_size bands (documented best-effort limitation). Pin it.
    img, _, _ = striped_page(50, [("line", 120)])  # width == crop_size -> single column
    cropper = TextAwareCropper(crop_size=50)
    assert cropper.y_cuts(img) == [(0, 50), (50, 100), (100, 120)]
    tiles = cropper.crop(img)
    assert len(tiles) == 3 and all(t.size == (50, 50) for t in tiles)


def test_crop_is_deterministic() -> None:
    img, _, _ = striped_page(
        224, [("gap", 10), ("line", 20), ("gap", 10), ("line", 20), ("gap", 10)]
    )
    cropper = TextAwareCropper(crop_size=50)
    a, b = cropper.crop(img), cropper.crop(img)
    assert len(a) == len(b)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))

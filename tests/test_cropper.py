"""Phase A tests A3 — NonOverlappingCropper (TESTS.md A3)."""
from __future__ import annotations

from PIL import Image

from visword.data.cropper import NonOverlappingCropper


def test_non_overlapping_crops() -> None:
    cropper = NonOverlappingCropper(
        crop_size=490, overlap=0.0, min_text_ratio=0.0, target_size=224
    )
    # 980 = 2 * 490 → exactly 4 quadrants, zero pixel overlap.
    img = Image.new("RGB", (980, 980), (0, 0, 0))
    crops = cropper(img)
    assert len(crops) == 4

    # Re-derive bbox grid from the cropper's stride logic and verify disjoint.
    grid = cropper._grid(980)
    assert grid == [0, 490]
    boxes = [(x, y, x + 490, y + 490) for y in grid for x in grid]
    for i, b1 in enumerate(boxes):
        for b2 in boxes[i + 1 :]:
            no_overlap_x = b1[2] <= b2[0] or b2[2] <= b1[0]
            no_overlap_y = b1[3] <= b2[1] or b2[3] <= b1[1]
            assert no_overlap_x or no_overlap_y, f"overlap between {b1} and {b2}"


def test_min_text_ratio_filter() -> None:
    cropper = NonOverlappingCropper(
        crop_size=490, overlap=0.0, min_text_ratio=0.05, target_size=224
    )
    # All-white page → every tile fails the threshold → fallback centre crop.
    white = Image.new("RGB", (980, 980), (255, 255, 255))
    crops_white = cropper(white)
    assert len(crops_white) == 1

    # All-black page → every pixel "non-white" → all 4 quadrants kept.
    black = Image.new("RGB", (980, 980), (0, 0, 0))
    crops_black = cropper(black)
    assert len(crops_black) == 4


def test_target_size_resize() -> None:
    img = Image.new("RGB", (980, 980), (0, 0, 0))

    for crop_size in (224, 490, 700):
        cropper = NonOverlappingCropper(
            crop_size=crop_size, overlap=0.0, min_text_ratio=0.0, target_size=224
        )
        for tile in cropper(img):
            assert tile.size == (224, 224)

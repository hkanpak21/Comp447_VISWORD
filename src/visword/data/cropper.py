"""Non-overlapping sliding-window cropper for tall page screenshots.

Lessons from CONTEXT.md week 1: ``stride == crop_size`` (overlap=0) is the
only honest setting — any overlap leaks shared pixels between anchor and
positive crops and produces artefactual 100% R@1. Empty / nearly-blank
images fall back to a single centre crop instead of returning zero crops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image


@dataclass
class NonOverlappingCropper:
    """Tile an image into ``crop_size``-square crops with optional overlap.

    Attributes:
        crop_size: edge length of each crop, in original image pixels.
        overlap: fraction of overlap between adjacent crops; 0.0 = no overlap.
            Stride is ``round(crop_size * (1 - overlap))`` and clamped to >= 1.
        min_text_ratio: a crop is kept only if its "non-near-white" pixel
            fraction is >= this value. Lets us skip header/footer whitespace.
        target_size: every returned crop is resized to (target_size, target_size).
        whiteness_threshold: pixels with all RGB channels >= this value are
            considered "near-white" / background.
    """

    crop_size: int = 490
    overlap: float = 0.0
    min_text_ratio: float = 0.05
    target_size: int = 224
    whiteness_threshold: int = 245

    def __post_init__(self) -> None:
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")
        self.stride = max(1, round(self.crop_size * (1.0 - self.overlap)))

    # ------------------------------------------------------------------

    def _grid(self, side: int) -> List[int]:
        """Top-left coordinates along one axis, last one snapped to fit."""
        if side <= self.crop_size:
            return [0]
        coords = list(range(0, side - self.crop_size + 1, self.stride))
        if coords[-1] + self.crop_size < side:
            coords.append(side - self.crop_size)
        return coords

    def _is_kept(self, crop: Image.Image) -> bool:
        arr = np.asarray(crop)
        if arr.ndim == 2:
            mask = arr < self.whiteness_threshold
        else:
            mask = (arr[..., :3] < self.whiteness_threshold).any(axis=-1)
        text_ratio = float(mask.mean())
        return text_ratio >= self.min_text_ratio

    def _centre_crop(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        side = min(w, h, self.crop_size)
        left = (w - side) // 2
        top = (h - side) // 2
        return image.crop((left, top, left + side, top + side))

    # ------------------------------------------------------------------

    def crop(self, image: Image.Image) -> List[Image.Image]:
        """Return the list of crops (each ``target_size`` x ``target_size``)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size

        kept: List[Image.Image] = []
        for y in self._grid(h):
            for x in self._grid(w):
                tile = image.crop((x, y, x + self.crop_size, y + self.crop_size))
                if self._is_kept(tile):
                    kept.append(tile)

        if not kept:
            kept = [self._centre_crop(image)]

        return [c.resize((self.target_size, self.target_size), Image.BILINEAR) for c in kept]

    __call__ = crop


@dataclass
class TextAwareCropper:
    """Legible, text-aware cropper (ticket 01 / E1).

    Two changes vs :class:`NonOverlappingCropper`, motivated by the legibility
    root cause (a 490->224 bilinear shrink turns body glyphs into ~6px smears):

    1. **Native resolution.** ``target_size`` defaults to ``crop_size`` so a tile
       is returned at its source pixels (no downsample). Set them unequal only for
       a deliberate, clearly-labelled secondary high-res path.
    2. **Text-aware vertical cuts.** A horizontal projection profile (per-row ink
       fraction) locates the whitespace gaps between text lines; vertical tile
       boundaries are snapped into those gaps so no glyph row is sliced, and
       blank / fragment tiles are dropped.

    Attributes:
        crop_size: edge length of each native window, in source pixels.
        target_size: output edge length; ``None`` -> equals ``crop_size`` (no shrink).
        min_text_ratio: keep a tile only if its non-near-white fraction >= this.
        whiteness_threshold: RGB value at/above which a pixel is "near-white" bg.
        line_ink_threshold: a row counts as background if its non-near-white
            fraction is below this (so faint anti-aliasing does not seal a gap).
        gap_min_height: minimum consecutive background rows to count as a gap.
    """

    crop_size: int = 224
    target_size: int | None = None
    min_text_ratio: float = 0.05
    whiteness_threshold: int = 245
    line_ink_threshold: float = 0.01
    gap_min_height: int = 2

    def __post_init__(self) -> None:
        if self.target_size is None:
            self.target_size = self.crop_size

    # ------------------------------------------------------------------

    def _row_ink(self, image: Image.Image) -> np.ndarray:
        """Per-row fraction of non-near-white pixels (length = image height)."""
        arr = np.asarray(image if image.mode == "RGB" else image.convert("RGB"))
        mask = (arr[..., :3] < self.whiteness_threshold).any(axis=-1)
        return mask.mean(axis=1)

    def inter_line_gaps(self, image: Image.Image) -> List[tuple]:
        """Whitespace bands between text lines, as ``[start, end)`` row ranges.

        A maximal run of >= ``gap_min_height`` background rows (row ink below
        ``line_ink_threshold``) is a gap.
        """
        is_bg = self._row_ink(image) < self.line_ink_threshold
        gaps: List[tuple] = []
        start = None
        for i, bg in enumerate(is_bg):
            if bg and start is None:
                start = i
            elif not bg and start is not None:
                if i - start >= self.gap_min_height:
                    gaps.append((start, i))
                start = None
        if start is not None and len(is_bg) - start >= self.gap_min_height:
            gaps.append((start, len(is_bg)))
        return gaps

    def y_cuts(self, image: Image.Image) -> List[tuple]:
        """Contiguous ``(top, bottom)`` source row bands, each <= ``crop_size`` tall.

        Boundaries are snapped to inter-line gap midpoints so a band edge never
        slices a glyph row. Greedy: from the current top, take the farthest gap
        midpoint still within ``crop_size``; if no gap is within reach (a single
        line taller than ``crop_size``), fall back to a hard cut at ``crop_size``.
        """
        h = image.height
        if h <= self.crop_size:
            return [(0, h)]
        # Only INTERIOR gaps are cut candidates — a leading/trailing page margin is
        # not an inter-line gap, and using it would carve off degenerate sliver bands.
        mids = [(g0 + g1) // 2 for g0, g1 in self.inter_line_gaps(image) if g0 > 0 and g1 < h]
        cands = sorted({0, h, *(m for m in mids if 0 < m < h)})
        bands: List[tuple] = []
        cur = 0
        while cur < h:
            reach = cur + self.crop_size
            nxt = max((c for c in cands if cur < c <= reach), default=min(reach, h))
            bands.append((cur, nxt))
            cur = nxt
        return bands

    def _x_grid(self, w: int) -> List[int]:
        """Native-resolution column left-edges (stride = crop_size, last snapped)."""
        if w <= self.crop_size:
            return [0]
        coords = list(range(0, w - self.crop_size + 1, self.crop_size))
        if coords[-1] + self.crop_size < w:
            coords.append(w - self.crop_size)
        return coords

    def _keep(self, tile: Image.Image) -> bool:
        arr = np.asarray(tile)
        mask = (arr[..., :3] < self.whiteness_threshold).any(axis=-1)
        return float(mask.mean()) >= self.min_text_ratio

    def crop(self, image: Image.Image) -> List[Image.Image]:
        """Return native-resolution, line-aligned tiles (``target_size`` square).

        Vertical bands come from :meth:`y_cuts` (boundaries in inter-line gaps);
        columns from :meth:`_x_grid`. Each band/column region is pasted top-left
        onto a white ``crop_size`` canvas (so short bands are white-padded, never
        stretched), keeping pixels native. Blank / low-text tiles are dropped.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, _ = image.size
        cs = self.crop_size

        kept: List[Image.Image] = []
        for top, bot in self.y_cuts(image):
            for x in self._x_grid(w):
                region = image.crop((x, top, min(x + cs, w), bot))
                canvas = Image.new("RGB", (cs, cs), (255, 255, 255))
                canvas.paste(region, (0, 0))
                if self._keep(canvas):
                    kept.append(canvas)

        if not kept:  # fully blank page -> single centre window, padded to crop_size
            side = min(image.size[0], image.size[1], cs)
            left = (image.size[0] - side) // 2
            top = (image.size[1] - side) // 2
            region = image.crop((left, top, left + side, top + side))
            canvas = Image.new("RGB", (cs, cs), (255, 255, 255))
            canvas.paste(region, (0, 0))
            kept = [canvas]

        if self.target_size != cs:
            kept = [c.resize((self.target_size, self.target_size), Image.BILINEAR) for c in kept]
        return kept

    __call__ = crop

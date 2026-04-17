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

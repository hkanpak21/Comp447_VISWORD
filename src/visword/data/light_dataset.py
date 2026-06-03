"""Lazy-decoding multi-positive Wikipedia-screenshot dataset.

Streams rows from the prefetched cache (PROJECT_SPEC.md §4.2). Opens each
PNG only in ``__getitem__``, closes the PIL handle right after cropping.
RSS at steady state scales with ``num_workers * batch_size``, not with
``len(dataset)`` — the CONTEXT.md session-3 RAM fix.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from visword.data.cropper import NonOverlappingCropper


ImageTransform = Callable[[Image.Image], torch.Tensor]


def default_transform() -> ImageTransform:
    """Standard ImageNet normalisation for ViT inputs — matches DINOv2 defaults."""
    import torchvision.transforms as T
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class LightWikiScreenshotDataset(Dataset):
    """Multi-positive crop dataset for contrastive training.

    ``__getitem__(i)`` reads row ``i``'s PNG, cropper-tiles it, then samples
    ``k_per_page`` crops (with replacement if fewer are available). Returns
    ``(crops: (K, 3, H, W) tensor, label: int)``.

    For eval, set ``k_per_page=2`` for anchor/positive pairs, or use the
    full crop list via ``iter_all_crops``.
    """

    def __init__(
        self,
        cache_dir: Path,
        indices: list[int],
        *,
        cropper: NonOverlappingCropper,
        transform: ImageTransform | None = None,
        k_per_page: int = 4,
        seed: int = 0,
        return_text: bool = False,
        text_source: str = "title",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json at {self.cache_dir}. "
                f"Run scripts/prefetch_data.py or scripts/submit.sh prefetch first."
            )
        self.manifest = json.loads(manifest_path.read_text())

        all_rows = self.manifest["rows"]
        if max(indices) >= len(all_rows):
            raise IndexError(
                f"requested index {max(indices)} but cache has only {len(all_rows)} rows"
            )
        self.rows = [all_rows[i] for i in indices]

        self.cropper = cropper
        self.transform = transform or default_transform()
        self.k_per_page = k_per_page
        self._rng = random.Random(seed)
        self.return_text = return_text
        self.text_source = text_source

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    def _load_and_crop(self, row_idx: int) -> list[Image.Image]:
        blob_path = self.cache_dir / self.rows[row_idx]["image_path"]
        with Image.open(blob_path) as im:
            im.load()   # force decode so we can close the file handle
            crops = self.cropper(im)   # already resized to target_size
        return crops

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, str, int]:
        crops = self._load_and_crop(i)
        if len(crops) >= self.k_per_page:
            sampled = self._rng.sample(crops, self.k_per_page)
        else:
            # Pad by resampling — rare on real Wikipedia screenshots, common on tests.
            sampled = list(crops) + self._rng.choices(crops, k=self.k_per_page - len(crops))
        tensors = torch.stack([self.transform(c) for c in sampled])    # (K, 3, H, W)
        
        if self.return_text:
            row = self.rows[i]
            text = ""
            if self.text_source == "text" and "text_path" in row:
                text_path = self.cache_dir / row["text_path"]
                if text_path.exists():
                    text = text_path.read_text(encoding="utf-8")
                else:
                    text = row.get("title", "")
            else:
                text = row.get("title", "")
            return tensors, text, i

        return tensors, i     # use local row index as the "page label"

    # ------------------------------------------------------------------

    def iter_all_crops(self) -> "list[tuple[int, list[torch.Tensor]]]":
        """Eager list of ``(local_idx, [transformed crops...])`` — used by Phase-1 eval."""
        out: list[tuple[int, list[torch.Tensor]]] = []
        for i in range(len(self)):
            crops = self._load_and_crop(i)
            out.append((i, [self.transform(c) for c in crops]))
        return out


def multi_positive_collate(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ``(B,)`` of ``(K, 3, H, W)`` tensors into ``(B*K, 3, H, W)``.

    Labels are repeated so samples from the same page share a label
    (required for multi-positive losses — losses.py assumes this).
    """
    tensors, labels = zip(*batch)
    K = tensors[0].shape[0]
    imgs = torch.cat(list(tensors), dim=0)                     # (B*K, 3, H, W)
    lbls = torch.as_tensor(np.repeat(labels, K), dtype=torch.long)
    return imgs, lbls


__all__ = [
    "LightWikiScreenshotDataset",
    "default_transform",
    "multi_positive_collate",
]

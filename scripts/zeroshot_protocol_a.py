#!/usr/bin/env python3
"""Zero-shot Protocol-A retrieval for one encoder.

Usage::

    PYTHONPATH=src python -m scripts.zeroshot_protocol_a \\
        --encoder dinov2 --out runs/_zeroshot/dinov2_protocolA.json \\
        --num-pages 2000 --seed 42

Encoders are the strings accepted by ``visword.analysis.platonic_alignment``
plus the trained ZeroShot wrappers in ``visword.models.zeroshot`` for
DINOv2 / CLIP / ImageNet-ViT (which use the dataset-pipeline transforms
rather than per-encoder preprocessors).

This script is the "produce a row of the zero-shot retrieval table"
primitive used by Track C in the 72h plan.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from visword.data import manifest as M
from visword.data.cropper import NonOverlappingCropper
from visword.eval_phase1_holdout import compute_protocol_a_recall


@torch.no_grad()
def _encode_with_callable(
    encode_fn,                # paths_or_pil_list -> (N, D) np.ndarray
    img_paths: list[Path],
    device: torch.device,
) -> np.ndarray:
    """Adapter for the ``encode_*`` helpers in ``platonic_alignment``."""
    return encode_fn(img_paths, device)


def _load_eval_pages(cache_dir: Path, num_pages: int, seed: int) -> list[dict]:
    manifest = M.read_manifest(cache_dir)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(manifest["num_rows"])
    picks = perm[:num_pages].tolist()
    return [manifest["rows"][i] for i in picks]


def _crop_pages_to_pil(rows: list[dict], cache_dir: Path,
                       cropper: NonOverlappingCropper) -> tuple[list[Image.Image], np.ndarray]:
    """For each page row, generate non-overlapping crops with min_text_ratio
    filter. Returns ``(pil_crops, page_ids)``."""
    pils: list[Image.Image] = []
    page_ids: list[int] = []
    for local_idx, row in enumerate(rows):
        path = cache_dir / row["image_path"]
        with Image.open(path) as im:
            crops = cropper.crop(im.convert("RGB"))
        for c in crops:
            # cropper returns PIL, target_size already applied.
            pils.append(c.copy())
            page_ids.append(local_idx)
    return pils, np.asarray(page_ids)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path,
                   default="/scratch/hkanpak21/VISWORD/data/wiki_ss")
    p.add_argument("--encoder", required=True,
                   choices=["dinov2_cls", "clip_image", "siglip_image",
                            "imagenet_vit", "plain_vit", "ijepa"])
    p.add_argument("--num-pages", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crop-size", type=int, default=490)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--min-text-ratio", type=float, default=0.05)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    rows = _load_eval_pages(args.cache_dir, args.num_pages, args.seed)
    cropper = NonOverlappingCropper(
        crop_size=args.crop_size, overlap=0.0,
        min_text_ratio=args.min_text_ratio,
        target_size=args.target_size,
    )
    t0 = time.time()
    print(f"cropping {len(rows)} pages...", flush=True)
    pil_crops, page_ids = _crop_pages_to_pil(rows, args.cache_dir, cropper)
    print(f"  -> {len(pil_crops)} crops from {len(rows)} pages "
          f"(mean {len(pil_crops)/len(rows):.1f} crops/page) in {time.time()-t0:.1f}s",
          flush=True)

    # Encode using encode_* helpers from platonic_alignment (each builds its
    # own model + transform internally; consistent with the Platonic grid).
    from visword.analysis import platonic_alignment as PA
    enc_map = {
        "dinov2_cls": PA.encode_dinov2,
        "clip_image": PA.encode_clip_image,
        "siglip_image": PA.encode_siglip_image,
        "imagenet_vit": PA.encode_imagenet_vit,
        "plain_vit": PA.encode_plain_vit,
        "ijepa": PA.encode_ijepa,
    }
    encode_fn = enc_map[args.encoder]

    # Adapter: encode_* expects file paths but we have in-memory PIL crops.
    # Quick-and-dirty: persist crops to a tmp dir and pass paths.
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        crop_paths = []
        for i, c in enumerate(pil_crops):
            cp = td_path / f"{i:08d}.png"
            c.save(cp, format="PNG", compress_level=1)
            crop_paths.append(cp)

        print(f"encoding {len(crop_paths)} crops with {args.encoder}...", flush=True)
        t1 = time.time()
        emb = encode_fn(crop_paths, device)              # (N, D) np.ndarray
        print(f"  -> shape={emb.shape} in {time.time()-t1:.1f}s", flush=True)

    embs_t = torch.from_numpy(emb)
    pids_t = torch.from_numpy(page_ids).long()

    print("computing Protocol-A recall...", flush=True)
    result = compute_protocol_a_recall(embs_t, pids_t, k_values=[1, 5, 10, 20])

    payload = {
        "encoder": args.encoder,
        "num_pages_requested": args.num_pages,
        "num_pages_evaluated": result["num_pages"],
        "num_crops": result["num_crops"],
        "num_queries_eligible": result["num_queries_eligible"],
        "min_text_ratio_for_query": args.min_text_ratio,
        "recall": result["recall"],
        "sanity": result["sanity"],
        "seed": args.seed,
        "embedding_dim": int(emb.shape[1]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

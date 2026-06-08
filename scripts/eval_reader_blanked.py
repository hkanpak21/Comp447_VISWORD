"""Ticket 07 — title-erasure check on a trained MAE reader.

Loads a reader checkpoint and evaluates page-level same-page re-id on the eval slice
BOTH normally and with the top title region painted white, then reports the delta. A
near-zero delta means the reader does not bind page identity to the title bar (the
layout-fingerprint shortcut is gone). Use on the unmasked reader AND a reader fine-tuned
with random title-masking (ticket 07) to compare.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from visword.data import manifest as M
from visword.data.cropper import TextAwareCropper
from visword.models.mae_reader import build_reader
from visword.page_reid import page_reid_recall

_T = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def blank_top(page: Image.Image, frac: float) -> Image.Image:
    if frac <= 0:
        return page
    arr = np.asarray(page).copy()
    arr[: int(round(arr.shape[0] * frac)), :, :] = 255
    return Image.fromarray(arr)


@torch.no_grad()
def eval_recall(reader, cache_dir, rows, eval_idx, cropper, device, blank_frac=0.0,
                batch=64, max_crops=12):
    embs, pids, buf, bpid = [], [], [], []

    def flush():
        if buf:
            embs.append(reader(torch.stack(buf).to(device)).cpu()); pids.extend(bpid)
            buf.clear(); bpid.clear()

    for local, gi in enumerate(eval_idx):
        with Image.open(cache_dir / rows[int(gi)]["image_path"]) as im:
            crops = cropper(blank_top(im.convert("RGB"), blank_frac))[:max_crops]
        for c in crops:
            buf.append(_T(c)); bpid.append(local)
            if len(buf) >= batch:
                flush()
    flush()
    E = torch.cat(embs) if embs else torch.empty(0, reader.descriptor_dim)
    return page_reid_recall(E, torch.tensor(pids), k_values=(1, 5, 10, 20))["recall"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-eval", type=int, default=2000)
    ap.add_argument("--eval-pages", type=int, default=2000)
    ap.add_argument("--blank-frac", type=float, default=0.25)
    ap.add_argument("--num-trainable-blocks", type=int, default=4)
    ap.add_argument("--backbone", choices=["mae", "dit"], default="mae")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = M.read_manifest(args.cache_dir); rows = man["rows"]
    n = man.get("num_rows", len(rows))
    eval_idx = np.random.default_rng(args.seed).permutation(n)[n - args.num_eval:][:args.eval_pages]
    cropper = TextAwareCropper(crop_size=224, target_size=224)

    reader = build_reader(args.backbone, num_trainable_blocks=args.num_trainable_blocks).to(device).eval()
    reader.load_state_dict(torch.load(args.ckpt, map_location=device)["reader"])

    normal = eval_recall(reader, args.cache_dir, rows, eval_idx, cropper, device, blank_frac=0.0)
    blanked = eval_recall(reader, args.cache_dir, rows, eval_idx, cropper, device, blank_frac=args.blank_frac)
    delta = {k: round(blanked[k] - normal[k], 4) for k in normal}

    args.out.mkdir(parents=True, exist_ok=True)
    res = {"ckpt": str(args.ckpt), "eval_pages": int(len(eval_idx)), "blank_frac": args.blank_frac,
           "recall_normal": normal, "recall_blanked": blanked, "delta_blanked_minus_normal": delta}
    (args.out / "title_erasure.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

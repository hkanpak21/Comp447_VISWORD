#!/usr/bin/env python
"""Ticket 02 — re-baseline the encoder grid at legible (native-224) resolution.

For each encoder (6 originals + MAE), over the disjoint eval slice:
  page -> TextAwareCropper native-224 legible crops -> shared-wrapper embeddings
  -> page-level same-page re-identification recall@k (leave-one-out gallery).
Also records total params and crops/sec throughput so "efficiently" is measured.

Resumable: writes ``results/<encoder>.json`` as each encoder finishes and skips any
already present, so an interrupted job (Valar QOS=1, <=8h walls) just continues.
Numbers are NOT directly comparable to the v3 baseline (that was crop->page at the
490->224 squash); this is the new legible-resolution, page-level protocol.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from visword.config import Config
from visword.data import manifest as M
from visword.data.cropper import TextAwareCropper
from visword.models.encoder_wrapper import ENCODER_NAMES, build_encoder
from visword.page_reid import page_reid_recall

_IMAGENET = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def disjoint_eval_indices(num_rows: int, num_eval: int, seed: int) -> np.ndarray:
    """Held-out tail of a seeded permutation (identical to build_legible_crops)."""
    return np.random.default_rng(seed).permutation(num_rows)[num_rows - num_eval :]


def default_cfg(root: Path) -> Config:
    raw = yaml.safe_load((root / "configs" / "default.yaml").read_text())
    return Config.model_validate(raw)


@torch.no_grad()
def encode_slice(encoder, cache_dir, rows, eval_idx, cropper, device, batch_size):
    """Return (embeddings (N,D) cpu, page_ids (N,) cpu, n_crops, seconds)."""
    embs, page_ids = [], []
    buf, buf_pid = [], []
    t0 = time.time()

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        z = encoder(x).detach().cpu()
        embs.append(z)
        page_ids.extend(buf_pid)
        buf.clear(); buf_pid.clear()

    for local, gi in enumerate(eval_idx):
        with Image.open(cache_dir / rows[int(gi)]["image_path"]) as im:
            crops = cropper(im.convert("RGB"))
        for c in crops:
            buf.append(_IMAGENET(c))
            buf_pid.append(local)
            if len(buf) >= batch_size:
                flush()
    flush()
    E = torch.cat(embs) if embs else torch.empty(0)
    return E, torch.tensor(page_ids, dtype=torch.long), int(E.shape[0]), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="run dir (created/reused)")
    ap.add_argument("--encoders", nargs="*", default=list(ENCODER_NAMES))
    ap.add_argument("--num-eval", type=int, default=2000, help="eval slice size (tail)")
    ap.add_argument("--num-pages", type=int, default=2000, help="pages actually scored")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--min-text-ratio", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--k-values", nargs="*", type=int, default=[1, 5, 10, 20])
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = default_cfg(root)

    manifest = M.read_manifest(args.cache_dir)
    rows = manifest["rows"]
    num_rows = manifest.get("num_rows", len(rows))
    eval_idx = disjoint_eval_indices(num_rows, min(args.num_eval, num_rows), args.seed)
    eval_idx = eval_idx[: args.num_pages]

    res_dir = args.out / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "provenance.json").write_text(json.dumps({
        "ts_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "host": socket.gethostname(), "git_sha": git_sha(root), "device": str(device),
        "num_rows": int(num_rows), "num_pages_scored": int(len(eval_idx)),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }, indent=2))

    cropper = TextAwareCropper(crop_size=args.crop_size, target_size=args.crop_size,
                              min_text_ratio=args.min_text_ratio)

    for name in args.encoders:
        out_path = res_dir / f"{name}.json"
        if out_path.exists():
            print(f"[skip] {name} (already done)", flush=True)
            continue
        print(f"[run ] {name} on {len(eval_idx)} pages ...", flush=True)
        enc = build_encoder(name, cfg).to(device).eval()
        n_params = int(sum(p.numel() for p in enc.parameters()))
        E, pid, n_crops, secs = encode_slice(
            enc, args.cache_dir, rows, eval_idx, cropper, device, args.batch_size)
        rec = page_reid_recall(E, pid, k_values=tuple(args.k_values))
        row = {
            "encoder": name, "descriptor_dim": int(enc.descriptor_dim),
            "total_params": n_params, "trainable_params": 0,
            "num_pages": int(len(eval_idx)), "num_crops": n_crops,
            "crops_per_sec": round(n_crops / secs, 1) if secs else None,
            "recall": rec["recall"], "sanity": rec["sanity"],
            "num_queries_eligible": rec["num_queries_eligible"],
        }
        out_path.write_text(json.dumps(row, indent=2))
        del enc
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[done] {name}: R@10={rec['recall'].get('10')}  "
              f"{row['crops_per_sec']} crops/s  params={n_params/1e6:.0f}M", flush=True)

    # Consolidated table.
    table = [json.loads((res_dir / f"{n}.json").read_text())
             for n in args.encoders if (res_dir / f"{n}.json").exists()]
    (args.out / "grid_summary.json").write_text(json.dumps(table, indent=2))
    print(f"\nwrote {len(table)} encoder rows -> {args.out}/grid_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

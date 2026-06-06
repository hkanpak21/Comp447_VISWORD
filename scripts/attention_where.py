"""Ticket 06 — attention "where it reads" (focused: frozen MAE vs the trained reader).

Both are ViT-MAE, so attention extracts cleanly via output_attentions (eager). For a few
eval pages' native-224 crops, compute the last-layer CLS->patch attention map and an
"attention-on-text" score = fraction of attention mass on INK (non-white) patches vs
whitespace. Answers, for OUR model: did teaching it to read shift attention onto text?
Saves heatmap overlays. (Extending to CLIP/DINOv2 needs hook-based extraction — a
follow-up; those are noted, not done here.)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from visword.data import manifest as M  # noqa: E402
from visword.data.cropper import TextAwareCropper  # noqa: E402

_T = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def load_eager_mae(ckpt: Path | None, device):
    from transformers import ViTMAEModel
    m = ViTMAEModel.from_pretrained("facebook/vit-mae-base", attn_implementation="eager")
    m.config.mask_ratio = 0.0
    if hasattr(m, "embeddings"):
        m.embeddings.config.mask_ratio = 0.0
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu")["reader"]
        enc = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        missing, unexpected = m.load_state_dict(enc, strict=False)
        print(f"  loaded reader encoder ({len(enc)} keys; {len(unexpected)} unexpected)", flush=True)
    return m.to(device).eval()


def ink_per_patch(crop: Image.Image, grid: int = 14, white: int = 245) -> np.ndarray:
    """Fraction of non-white pixels per 16x16 patch -> (grid*grid,)."""
    arr = np.asarray(crop.resize((224, 224)))
    ink = (arr[..., :3] < white).any(-1).astype(np.float32)        # (224,224)
    p = 224 // grid
    return ink.reshape(grid, p, grid, p).mean(axis=(1, 3)).reshape(-1)


@torch.no_grad()
def attn_on_text(model, crops, device, ink_thresh: float = 0.05):
    """Mean attention-on-text over crops + the per-patch CLS->patch attention maps."""
    scores, maps = [], []
    for s in range(0, len(crops), 32):
        batch = crops[s:s + 32]
        x = torch.stack([_T(c) for c in batch]).to(device)
        att = model(pixel_values=x, output_attentions=True).attentions[-1]  # (B,heads,seq,seq)
        cls2patch = att[:, :, 0, 1:].mean(1)                                  # (B, P) CLS->patches
        cls2patch = cls2patch / cls2patch.sum(1, keepdim=True).clamp(min=1e-9)
        for j, c in enumerate(batch):
            ink = torch.tensor(ink_per_patch(c) >= ink_thresh, device=device)
            a = cls2patch[j]
            scores.append(float(a[ink].sum()) if ink.any() else 0.0)
            maps.append(a.detach().cpu().numpy())
    return float(np.mean(scores)) if scores else 0.0, maps


def save_heatmap(crop, amap, path, grid=14):
    fig, ax = plt.subplots(1, 2, figsize=(5, 2.6))
    ax[0].imshow(crop.resize((224, 224))); ax[0].set_title("crop", fontsize=8); ax[0].axis("off")
    hm = amap.reshape(grid, grid)
    ax[1].imshow(crop.resize((224, 224))); ax[1].imshow(
        np.kron(hm, np.ones((16, 16))), cmap="jet", alpha=0.5)
    ax[1].set_title("CLS->patch attn", fontsize=8); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reader-ckpt", type=Path, default=None)
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--max-crops", type=int, default=8)
    ap.add_argument("--num-eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = M.read_manifest(args.cache_dir); rows = man["rows"]
    n = man.get("num_rows", len(rows))
    eval_idx = np.random.default_rng(args.seed).permutation(n)[n - args.num_eval:][:args.pages]
    cropper = TextAwareCropper(crop_size=224, target_size=224)

    crops, page_of = [], []
    for local, gi in enumerate(eval_idx):
        with Image.open(args.cache_dir / rows[int(gi)]["image_path"]) as im:
            cs = cropper(im.convert("RGB"))[:args.max_crops]
        crops += cs; page_of += [int(gi)] * len(cs)

    args.out.mkdir(parents=True, exist_ok=True)
    models = {"mae_frozen": load_eager_mae(None, device)}
    if args.reader_ckpt and args.reader_ckpt.exists():
        models["mae_reader"] = load_eager_mae(args.reader_ckpt, device)

    result = {"ts_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
              "host": socket.gethostname(), "num_pages": int(len(eval_idx)),
              "num_crops": len(crops), "attention_on_text": {}}
    for name, model in models.items():
        score, maps = attn_on_text(model, crops, device)
        result["attention_on_text"][name] = round(score, 4)
        for k in range(min(4, len(crops))):
            save_heatmap(crops[k], maps[k], args.out / f"{name}_crop{k}.png")
        print(f"{name}: attention-on-text = {score:.4f}", flush=True)

    (args.out / "attention_on_text.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["attention_on_text"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

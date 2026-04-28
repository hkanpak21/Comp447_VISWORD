"""Visualise per-encoder saliency vs BERT word importance for a few pages.

Outputs a single multi-row PNG per page: row 1 is the page with detected
words boxed (colour = BERT importance). Rows 2..5 are the same page with
the CLIP/SigLIP/DINOv2/I-JEPA saliency map overlaid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

from visword.data import manifest as M
from scripts.attention_text_correlation import (
    SALIENCY_FNS, ocr_words, bert_word_importance,
    saliency_mass_in_bbox,
)


def overlay_saliency(img: Image.Image, sal_grid: np.ndarray,
                     cmap_name: str = "inferno", alpha: float = 0.5) -> Image.Image:
    H, W = img.size[1], img.size[0]
    s = sal_grid - sal_grid.min()
    s = s / max(s.max(), 1e-8)
    smap = Image.fromarray((s * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
    smap_arr = np.asarray(smap, dtype=np.float32) / 255.0
    cmap = colormaps[cmap_name]
    rgba = (cmap(smap_arr) * 255).astype(np.uint8)         # (H, W, 4)
    overlay = Image.fromarray(rgba[..., :3], "RGB")
    return Image.blend(img.convert("RGB"), overlay, alpha)


def boxed_image(img: Image.Image, ocr: list, bert_imp: dict,
                font: ImageFont.ImageFont) -> Image.Image:
    """Returns a copy with bboxes coloured by BERT importance."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    if not bert_imp:
        return out
    cmap = colormaps["viridis"]
    vals = [bert_imp.get(w.lower().strip(",.;:!?\"'()[]"), 0.0) for w, _ in ocr]
    vmax = max(vals) if vals else 1.0
    for (w, bbox), v in zip(ocr, vals):
        if v <= 0:
            continue
        rgb = (cmap(v / max(vmax, 1e-8))[:3])
        rgba = tuple(int(c * 255) for c in rgb) + (180,)
        draw.rectangle(bbox, outline=rgba, width=3)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path,
                   default="/scratch/hkanpak21/VISWORD/data/wiki_ss")
    p.add_argument("--summary-json", type=Path,
                   default="/scratch/hkanpak21/VISWORD/runs/_attn_corr/n200.json")
    p.add_argument("--out-dir", type=Path,
                   default="/scratch/hkanpak21/VISWORD/runs/_attn_corr/viz")
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    summary = json.loads(args.summary_json.read_text())
    vis = summary["vis_payload"][: args.n]
    encoders = ["clip", "siglip", "dinov2", "ijepa"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    import easyocr
    reader = easyocr.Reader(["en"], gpu=str(device).startswith("cuda"))
    font = ImageFont.load_default()

    for entry in vis:
        page_path = Path(entry["image_path"])
        title = entry["title"]
        print(f"=== {title} ({page_path}) ===", flush=True)
        img = Image.open(page_path).convert("RGB")

        # Cap rendering size so figures aren't ridiculous.
        max_dim = 800
        scale = min(1.0, max_dim / max(img.size))
        if scale < 1.0:
            img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)),
                             resample=Image.BILINEAR)

        # OCR + BERT importance.
        text_path = args.cache_dir / Path(*page_path.parts[-3:]).with_suffix(".txt")
        text_path = args.cache_dir / "texts" / page_path.parts[-2] / (page_path.stem + ".txt")
        text = text_path.read_text(errors="ignore")
        ocr = ocr_words(img, reader)
        bert_imp = bert_word_importance(text, device)

        # Saliency per encoder.
        sals = {}
        for enc in encoders:
            try:
                sal, _ = SALIENCY_FNS[enc](img, device)
                if sal is not None:
                    sals[enc] = sal
            except Exception as e:
                print(f"  {enc} failed: {e}")

        fig, axes = plt.subplots(1, len(encoders) + 1, figsize=(3.5 * (len(encoders) + 1), 4.0))
        axes[0].imshow(boxed_image(img, ocr, bert_imp, font))
        axes[0].set_title(f"BERT word importance\n{title}", fontsize=10)
        axes[0].axis("off")
        for ax, enc in zip(axes[1:], encoders):
            sal = sals.get(enc)
            if sal is None:
                ax.set_title(f"{enc}: n/a"); ax.axis("off"); continue
            ax.imshow(overlay_saliency(img, sal, cmap_name="inferno", alpha=0.55))
            ax.set_title(enc, fontsize=10)
            ax.axis("off")
        out_pdf = args.out_dir / f"{page_path.stem}.pdf"
        plt.tight_layout()
        plt.savefig(out_pdf, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"  -> {out_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

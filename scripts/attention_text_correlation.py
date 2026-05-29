#!/usr/bin/env python3
"""Spatial alignment between vision-encoder saliency maps and a text
encoder's per-word importance on the same wiki-ss page.

Pipeline (per page):
  1. Load original screenshot at full resolution.
  2. OCR via easyocr -> list of (word, bbox_in_page_coords).
  3. For each vision encoder E in {clip, siglip, dinov2, ijepa}:
        a. Resize page to 224x224 (CLIP/SigLIP/DINOv2) or 14x14 grid.
        b. Forward pass; extract CLS token + patch tokens.
        c. Saliency_i = cos(phi(CLS), phi(patch_i)) for each patch.
        d. Reshape to 2D grid (14x14 for ViT-B/16, 16x16 for ViT-B/14).
        e. Bilinearly upsample to original page dims.
        f. For each detected word w with bbox b_w, integrate the
           saliency map mass inside b_w; gives one scalar per word.
  4. For BERT, embed the page text once with a token-level forward
     pass; importance(token) = max attention from [CLS] to that token
     in the last layer (averaged across heads). For multi-token words,
     average the per-token importance.
  5. Match OCR words to BERT-tokenised words by lower-cased string;
     keep words that appear in both (typically 60-80% of detections).
  6. Spearman rho between (vision saliency mass) and (BERT importance)
     across the matched words for that page.

Aggregate across pages: mean rho per encoder.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr

from visword.data import manifest as M


# ---------------------------------------------------------------------------
# Vision-encoder saliency extraction (returns 2D map at native patch grid).
# ---------------------------------------------------------------------------


@torch.no_grad()
def saliency_clip(img: Image.Image, device) -> tuple[np.ndarray, tuple]:
    """Returns (saliency 14x14 numpy, image_size_used (224,224))."""
    import open_clip
    if not hasattr(saliency_clip, "_m"):
        m, _, tf = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        saliency_clip._m = m.to(device).eval()
        saliency_clip._tf = tf
    m = saliency_clip._m
    x = saliency_clip._tf(img).unsqueeze(0).to(device)
    visual = m.visual
    h = visual.conv1(x)
    B, C, H, W = h.shape
    h = h.reshape(B, C, H * W).permute(0, 2, 1)
    cls = visual.class_embedding.to(h.dtype) + torch.zeros(B, 1, C, dtype=h.dtype, device=h.device)
    h = torch.cat([cls, h], dim=1)
    h = h + visual.positional_embedding.to(h.dtype)
    h = visual.ln_pre(h)
    h = h.permute(1, 0, 2)
    h = visual.transformer(h)
    h = h.permute(1, 0, 2)
    h = visual.ln_post(h)
    h = F.normalize(h, p=2, dim=-1)                       # (1, 197, 768)
    sal = (h[0, 0:1] @ h[0, 1:].T).squeeze(0).cpu().numpy()
    sal = sal.reshape(14, 14)
    return sal, (224, 224)


@torch.no_grad()
def saliency_siglip(img: Image.Image, device) -> tuple[np.ndarray, tuple]:
    if not hasattr(saliency_siglip, "_m"):
        from transformers import AutoModel, AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        m = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
        saliency_siglip._m = m.vision_model
        saliency_siglip._proc = proc
    m = saliency_siglip._m
    proc = saliency_siglip._proc
    x = proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
    out = m(pixel_values=x, output_hidden_states=False)
    h = F.normalize(out.last_hidden_state, p=2, dim=-1)   # (1, 196, D) — no CLS
    cls = F.normalize(out.pooler_output, p=2, dim=-1)     # (1, D)
    sal = (cls @ h.transpose(1, 2)).squeeze().cpu().numpy()
    sal = sal.reshape(14, 14)
    return sal, (224, 224)


@torch.no_grad()
def saliency_dinov2(img: Image.Image, device) -> tuple[np.ndarray, tuple]:
    import torchvision.transforms as T
    if not hasattr(saliency_dinov2, "_m"):
        saliency_dinov2._m = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", verbose=False).to(device).eval()
        saliency_dinov2._tf = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    m = saliency_dinov2._m
    x = saliency_dinov2._tf(img.convert("RGB")).unsqueeze(0).to(device)
    feats = m.forward_features(x)
    cls = F.normalize(feats["x_norm_clstoken"], p=2, dim=-1)            # (1, 768)
    pat = F.normalize(feats["x_norm_patchtokens"], p=2, dim=-1)         # (1, 256, 768)
    sal = (cls @ pat.transpose(1, 2)).squeeze().cpu().numpy()
    sal = sal.reshape(16, 16)
    return sal, (224, 224)


@torch.no_grad()
def saliency_ijepa(img: Image.Image, device) -> tuple[np.ndarray, tuple]:
    """I-JEPA target encoder has no CLS token; we use mean-token as the
    'CLS surrogate' since that is what the downstream pipeline mean-pools."""
    from transformers import AutoModel, AutoProcessor
    if not hasattr(saliency_ijepa, "_m"):
        proc = AutoProcessor.from_pretrained("facebook/ijepa_vith14_1k")
        m = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").to(device).eval()
        saliency_ijepa._m = m
        saliency_ijepa._proc = proc
    m = saliency_ijepa._m
    proc = saliency_ijepa._proc
    inputs = proc(images=img.convert("RGB"), return_tensors="pt")
    x = inputs["pixel_values"].to(device)
    out = m(pixel_values=x)
    h = F.normalize(out.last_hidden_state, p=2, dim=-1)   # (1, T, D); no CLS
    pooled = F.normalize(h.mean(dim=1, keepdim=True), p=2, dim=-1)  # (1, 1, D)
    sal = (pooled @ h.transpose(1, 2)).squeeze().cpu().numpy()
    T_tok = sal.shape[0]
    side = int(round(T_tok ** 0.5))
    if side * side != T_tok:
        return None, None  # unrecognised grid
    sal = sal.reshape(side, side)
    return sal, (224, 224)


SALIENCY_FNS = {
    "clip":   saliency_clip,
    "siglip": saliency_siglip,
    "dinov2": saliency_dinov2,
    "ijepa":  saliency_ijepa,
}


# ---------------------------------------------------------------------------
# Text-encoder per-word importance via BERT [CLS] last-layer attention.
# ---------------------------------------------------------------------------


@torch.no_grad()
def bert_word_importance(text: str, device, max_words: int = 256) -> dict[str, float]:
    """Returns {lowercased word: importance scalar}. Importance = mean over
    BERT heads of last-layer attention from [CLS] to that word's first
    sub-token."""
    if not hasattr(bert_word_importance, "_m"):
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        m = AutoModel.from_pretrained("bert-base-uncased",
                                       output_attentions=True).to(device).eval()
        bert_word_importance._m = m
        bert_word_importance._tok = tok
    tok = bert_word_importance._tok
    m = bert_word_importance._m

    words = text.split()[:max_words]
    if len(words) < 2:
        return {}
    enc = tok(words, return_tensors="pt", is_split_into_words=True,
              truncation=True, max_length=512).to(device)
    out = m(**enc)
    # last layer attention: (1, num_heads, T, T); attention from CLS (idx 0)
    attn_cls = out.attentions[-1][0, :, 0, :].mean(0).cpu().numpy()       # (T,)
    word_ids = enc.word_ids(0)
    imp: dict[str, float] = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is None or wid >= len(words):
            continue
        w = words[wid].lower()
        if w not in imp:
            imp[w] = float(attn_cls[tok_idx])
    return imp


# ---------------------------------------------------------------------------
# OCR + saliency-mass-in-bbox.
# ---------------------------------------------------------------------------


def ocr_words(img: Image.Image, reader) -> list[tuple[str, tuple[int, int, int, int]]]:
    """easyocr -> [(word, (x0, y0, x1, y1)), ...] in image coords."""
    arr = np.array(img.convert("RGB"))
    raw = reader.readtext(arr, detail=1)
    out = []
    for box, txt, _ in raw:
        if not txt.strip():
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        for w in txt.split():
            out.append((w, (min(xs), min(ys), max(xs), max(ys))))
    return out


def saliency_mass_in_bbox(sal_grid: np.ndarray, img_size: tuple[int, int],
                           bbox: tuple[int, int, int, int]) -> float:
    """Bilinearly resize sal_grid to img_size and integrate over bbox."""
    from PIL import Image as _I
    H_img, W_img = img_size
    sal_norm = sal_grid - sal_grid.min()
    sal_norm = sal_norm / max(sal_norm.max(), 1e-8)
    sal_img = _I.fromarray((sal_norm * 255).astype(np.uint8)).resize(
        (W_img, H_img), resample=_I.BILINEAR)
    arr = np.asarray(sal_img, dtype=np.float32) / 255.0
    x0, y0, x1, y1 = (max(0, int(c)) for c in bbox)
    x1 = min(W_img, x1); y1 = min(H_img, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = arr[y0:y1, x0:x1]
    return float(region.mean())


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    p.add_argument("--cache-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "wiki_ss")
    p.add_argument("--num-pages", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoders", nargs="+",
                   default=["clip", "siglip", "dinov2", "ijepa"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vis-pages", type=int, default=3,
                   help="number of side-by-side visualisations to render")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    manifest = M.read_manifest(args.cache_dir)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(manifest["num_rows"])
    rows = [manifest["rows"][i] for i in perm[: args.num_pages]]

    import easyocr
    print("loading easyocr...", flush=True)
    reader = easyocr.Reader(["en"], gpu=str(device).startswith("cuda"))

    per_encoder_rhos: dict[str, list[float]] = {e: [] for e in args.encoders}
    matched_word_counts: list[int] = []
    n_done = 0

    vis_payload: list[dict] = []
    t0 = time.time()
    for row_idx, row in enumerate(rows):
        page_path = args.cache_dir / row["image_path"]
        text_path = args.cache_dir / row["text_path"]
        try:
            img = Image.open(page_path).convert("RGB")
            text = text_path.read_text(errors="ignore")
        except Exception as e:
            print(f"  skip {row['idx']}: {e}", flush=True)
            continue
        W_img, H_img = img.size

        ocr = ocr_words(img, reader)
        if len(ocr) < 5:
            continue

        bert_imp = bert_word_importance(text, device)
        if not bert_imp:
            continue

        per_encoder_word_scores: dict[str, list[float]] = {}
        for enc in args.encoders:
            try:
                sal, in_size = SALIENCY_FNS[enc](img, device)
                if sal is None:
                    continue
            except Exception as e:
                print(f"  {enc} failed on {row['idx']}: {e}", flush=True)
                continue
            ws, ts, vs = [], [], []
            for w, bbox in ocr:
                wl = w.lower().strip(",.;:!?\"'()[]")
                if wl not in bert_imp:
                    continue
                v = saliency_mass_in_bbox(sal, (H_img, W_img), bbox)
                ws.append(wl); ts.append(bert_imp[wl]); vs.append(v)
            per_encoder_word_scores[enc] = (ws, ts, vs)
            if len(ws) >= 5:
                rho, _ = spearmanr(ts, vs)
                if not np.isnan(rho):
                    per_encoder_rhos[enc].append(float(rho))

        # Visualisations: keep heatmaps + matched words for the first few.
        if len(vis_payload) < args.vis_pages and per_encoder_word_scores:
            vis_payload.append({"row_idx": row["idx"],
                                "title": row["title"],
                                "image_path": str(page_path),
                                "matched_words": len(per_encoder_word_scores.get(args.encoders[0], ([], [], []))[0])})

        n_done += 1
        matched_word_counts.append(len(set(w.lower() for w, _ in ocr) & set(bert_imp.keys())))
        if (row_idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            mean_rhos = {e: float(np.mean(v)) if v else float("nan")
                         for e, v in per_encoder_rhos.items()}
            print(f"  page {row_idx+1}/{len(rows)} ({elapsed:.0f}s)  "
                  f"mean rho so far: {mean_rhos}", flush=True)

    summary = {
        "num_pages_evaluated": n_done,
        "encoders": args.encoders,
        "mean_rho": {e: float(np.mean(v)) if v else None for e, v in per_encoder_rhos.items()},
        "median_rho": {e: float(np.median(v)) if v else None for e, v in per_encoder_rhos.items()},
        "std_rho": {e: float(np.std(v)) if v else None for e, v in per_encoder_rhos.items()},
        "n_pages_with_rho": {e: len(v) for e, v in per_encoder_rhos.items()},
        "mean_matched_words_per_page": float(np.mean(matched_word_counts)) if matched_word_counts else 0,
        "vis_payload": vis_payload,
        "seed": args.seed,
        "num_pages_requested": args.num_pages,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""I-JEPA + linear text adapter (Q3 follow-up).

Hypothesis: I-JEPA's image-only predictive features are zero-shot
useless for visual document retrieval (R@10 = 0.015, below random
init), but a small linear adapter trained post-hoc on
(page-screenshot, page-title) pairs can recover CLIP-level
performance --- without ever using text supervision in the heavy
backbone. This is the document-screenshot analogue of Jose et al.\
2024 ``DINOv2 Meets Text''.

Pipeline:

  1. Sample N_train + N_eval pages (disjoint from the Track-A train).
  2. Encode each page-screenshot CENTER CROP with frozen I-JEPA
     ($ x_i \in R^{1280} $).
  3. Encode each page-title with a frozen text encoder
     (BERT-base by default, $ y_i \in R^{768} $).
  4. Fit linear adapter $ A \in R^{1280 \times d_t} $ via InfoNCE on
     L2-normalised projections, $ \tau = 0.07 $, AdamW, ~2k steps.
  5. Eval on held-out pages: text-to-image retrieval R@k where
     gallery = $ A x_j $ and queries = $ y_q $.

Outputs to ``runs/ijepa_adapter/``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from visword.data import manifest as M


def _sample_pages(cache_dir: Path, n_train: int, n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    manifest = M.read_manifest(cache_dir)
    rows = manifest["rows"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    # Pick (train + eval) pages from the first 2*N positions of the seed=42
    # shuffled deck used by zero-shot Protocol-A. Skip the first 2000 to
    # avoid overlap with that eval slice.
    skip = 3000
    picks = perm[skip : skip + n_train + n_eval]
    train_rows = [rows[i] for i in picks[:n_train]]
    eval_rows  = [rows[i] for i in picks[n_train:]]
    return train_rows, eval_rows


def _read_text(row: dict, cache_dir: Path, source: str = "title") -> str:
    """Pull the title (or title+body[:200]) for a row."""
    title = row.get("title", "")
    if source == "title":
        return title
    text_path = cache_dir / row.get("text_path", "")
    if text_path.exists():
        body = text_path.read_text()[:200]
        return f"{title}. {body}"
    return title


def _center_crop_pil(im: Image.Image, target: int = 224) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side)).resize(
        (target, target), Image.BILINEAR)


@torch.no_grad()
def encode_ijepa_images(rows: list[dict], cache_dir: Path,
                        device: torch.device, batch_size: int = 16) -> np.ndarray:
    from visword.models.zeroshot import ZeroShotIJepa
    from visword.data.light_dataset import default_transform
    enc = ZeroShotIJepa(cfg=None).to(device).eval()
    tf = default_transform()
    out: list[np.ndarray] = []
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i : i + batch_size]
        ts = []
        for r in batch_rows:
            with Image.open(cache_dir / r["image_path"]) as im:
                im = im.convert("RGB")
                im = _center_crop_pil(im, 224)
                ts.append(tf(im))
        x = torch.stack(ts).to(device)
        z = enc(x).cpu().numpy()
        out.append(z)
        if (i // batch_size) % 10 == 0:
            print(f"  ijepa encode: {i + len(batch_rows)}/{len(rows)}", flush=True)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def encode_bert_titles(rows: list[dict], cache_dir: Path,
                       device: torch.device, source: str = "title",
                       batch_size: int = 32) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased",
                                      use_safetensors=True).to(device).eval()
    out: list[np.ndarray] = []
    for i in range(0, len(rows), batch_size):
        texts = [_read_text(r, cache_dir, source) for r in rows[i : i + batch_size]]
        enc = tok(texts, padding="max_length", truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state            # (B, T, 768)
        # mean-pool with attention mask
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        v = F.normalize(v, dim=-1)
        out.append(v.cpu().numpy())
    return np.concatenate(out, axis=0)


def info_nce(z_v: torch.Tensor, z_t: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP-style)."""
    z_v = F.normalize(z_v, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    logits = z_v @ z_t.T / tau
    labels = torch.arange(z_v.size(0), device=z_v.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def train_adapter(x: torch.Tensor, y: torch.Tensor,
                  d_out: int, steps: int = 2000, batch: int = 256,
                  lr: float = 1e-3, tau: float = 0.07,
                  device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """x: (N, d_in) image features; y: (N, d_out) text features.
    Returns adapter A (d_in, d_out) as a torch tensor."""
    N, d_in = x.shape
    A = nn.Linear(d_in, d_out, bias=False).to(device)
    nn.init.xavier_uniform_(A.weight)
    opt = torch.optim.AdamW(A.parameters(), lr=lr, weight_decay=1e-4)
    x = x.to(device); y = y.to(device)
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), device=device)
        z_v = A(x[idx])
        z_t = y[idx]
        loss = info_nce(z_v, z_t, tau)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}  loss {float(loss):.4f}", flush=True)
    return A.weight.detach().cpu()  # (d_out, d_in) per nn.Linear convention


def text_to_image_retrieval(z_v_eval: torch.Tensor, z_t_eval: torch.Tensor,
                            k_values: list[int]) -> dict:
    """For each text query y_q, rank gallery vectors by cosine."""
    z_v = F.normalize(z_v_eval, dim=-1)
    z_t = F.normalize(z_t_eval, dim=-1)
    sim = z_t @ z_v.T                      # (Q, P) — query is text, gallery is image
    Q = sim.shape[0]
    correct = (sim.topk(max(k_values), dim=1).indices == torch.arange(Q).unsqueeze(1))
    out = {}
    for k in k_values:
        out[str(k)] = float(correct[:, :k].any(dim=1).float().mean())
    diag = sim.diag()
    off = (sim.sum(dim=1) - diag) / max(1, sim.shape[1] - 1)
    return {
        "recall": out,
        "sanity": {
            "diag_sim_mean": float(diag.mean()),
            "off_sim_mean": float(off.mean()),
            "gap": float(diag.mean() - off.mean()),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path,
                   default="/scratch/hkanpak21/VISWORD/data/wiki_ss")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-source", choices=["title", "title_body"], default="title")
    p.add_argument("--out-dir", type=Path,
                   default="/scratch/hkanpak21/VISWORD/runs/ijepa_adapter")
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_rows, eval_rows = _sample_pages(
        args.cache_dir, args.n_train, args.n_eval, args.seed)
    print(f"sampled: train={len(train_rows)}, eval={len(eval_rows)}", flush=True)

    cache_train = args.out_dir / f"feats_train_n{args.n_train}.npz"
    cache_eval  = args.out_dir / f"feats_eval_n{args.n_eval}.npz"

    if cache_train.exists():
        d = np.load(cache_train)
        x_train, y_train = d["x"], d["y"]
        print(f"loaded train cache: x={x_train.shape}, y={y_train.shape}")
    else:
        t0 = time.time()
        x_train = encode_ijepa_images(train_rows, args.cache_dir, device)
        print(f"  ijepa train: {x_train.shape} in {time.time()-t0:.1f}s")
        t0 = time.time()
        y_train = encode_bert_titles(train_rows, args.cache_dir, device,
                                     source=args.text_source)
        print(f"  bert train:  {y_train.shape} in {time.time()-t0:.1f}s")
        np.savez(cache_train, x=x_train, y=y_train)

    if cache_eval.exists():
        d = np.load(cache_eval)
        x_eval, y_eval = d["x"], d["y"]
        print(f"loaded eval cache: x={x_eval.shape}, y={y_eval.shape}")
    else:
        t0 = time.time()
        x_eval = encode_ijepa_images(eval_rows, args.cache_dir, device)
        print(f"  ijepa eval: {x_eval.shape} in {time.time()-t0:.1f}s")
        t0 = time.time()
        y_eval = encode_bert_titles(eval_rows, args.cache_dir, device,
                                    source=args.text_source)
        print(f"  bert eval:  {y_eval.shape} in {time.time()-t0:.1f}s")
        np.savez(cache_eval, x=x_eval, y=y_eval)

    # ---- baseline: no adapter (raw I-JEPA + BERT cosine) ----
    print("\n=== baseline: raw I-JEPA <-> BERT (no adapter) ===")
    # Project I-JEPA to BERT dim by truncation/pad — meaningless but a floor
    d_t = y_eval.shape[1]
    base_score = text_to_image_retrieval(
        torch.from_numpy(x_eval[:, :d_t].astype(np.float32)),
        torch.from_numpy(y_eval.astype(np.float32)),
        k_values=[1, 5, 10, 20])
    print(json.dumps(base_score, indent=2))

    # ---- train adapter ----
    print("\n=== training linear adapter ===")
    t0 = time.time()
    A = train_adapter(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
        d_out=d_t, steps=args.steps, device=device)
    print(f"adapter trained in {time.time()-t0:.1f}s, shape={tuple(A.shape)}")

    # ---- eval adapted ----
    print("\n=== adapted I-JEPA <-> BERT ===")
    z_v_adapted = torch.from_numpy(x_eval.astype(np.float32)) @ A.T
    adapted_score = text_to_image_retrieval(
        z_v_adapted,
        torch.from_numpy(y_eval.astype(np.float32)),
        k_values=[1, 5, 10, 20])
    print(json.dumps(adapted_score, indent=2))

    # ---- save report ----
    report = {
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "seed": args.seed,
        "steps": args.steps,
        "ijepa_dim": int(x_train.shape[1]),
        "text_dim": int(y_train.shape[1]),
        "baseline_no_adapter": base_score,
        "adapted": adapted_score,
        "improvement_R10": adapted_score["recall"]["10"] - base_score["recall"]["10"],
    }
    out_path = args.out_dir / f"adapter_report_seed{args.seed}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

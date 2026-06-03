#!/usr/bin/env python3
r"""Probe whether fine-tuned I-JEPA features are linearly decodable to text.

This script answers the user's core question: *"Can a vision encoder behave
like a translator from document images into language-like latent
representations?"*

Pipeline:

1. Load a fine-tuned I-JEPA checkpoint (or the frozen baseline).
2. Encode eval-set page screenshots → (B, 1280) mean-pooled features.
3. Encode the same pages' titles with frozen BERT → (B, 768).
4. Fit a linear regression: I-JEPA features → BERT embeddings.
5. Report:
   - **R²** of the linear map (1.0 = perfectly linearly decodable).
   - **CKA** alignment between the two representation spaces.
   - **Procrustes distance** (how close the spaces are up to rotation).
   - **Text-to-image retrieval R@k** through the linear map (practical
     measure of alignment quality).

Usage::

    # Frozen I-JEPA baseline (no checkpoint):
    python scripts/probe_ijepa_text.py --n-eval 1000

    # Fine-tuned I-JEPA:
    python scripts/probe_ijepa_text.py \
        --checkpoint runs/<run_dir>/checkpoints/best_phase1.pt \
        --config runs/<run_dir>/config.resolved.yaml \
        --n-eval 1000

Outputs to ``runs/ijepa_text_probes/`` by default.
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visword.data import manifest as M


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _center_crop_pil(im: Image.Image, target: int = 224) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side)).resize(
        (target, target), Image.BILINEAR)


@torch.no_grad()
def encode_ijepa_finetuned(
    rows: list[dict],
    cache_dir: Path,
    checkpoint: Path | None,
    config_path: Path | None,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract features from a fine-tuned or frozen I-JEPA."""
    from visword.data.light_dataset import default_transform

    if checkpoint is not None and config_path is not None:
        # Load fine-tuned model
        import yaml
        from visword.config import Config
        from visword.models.ijepa_finetune import IJepaFinetune

        raw = yaml.safe_load(config_path.read_text())
        cfg = Config.model_validate(raw)
        model = IJepaFinetune(cfg)
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()

        # Extract from the backbone (pre-head) for probing
        def _extract(x: torch.Tensor) -> torch.Tensor:
            _, mean_pool = model.backbone(x)
            return mean_pool
    else:
        # Frozen baseline
        from visword.models.ijepa_backbone import IJepaBackbone
        backbone = IJepaBackbone(num_trainable_blocks=0).to(device).eval()

        def _extract(x: torch.Tensor) -> torch.Tensor:
            _, mean_pool = backbone(x)
            return mean_pool

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
        z = _extract(x).cpu().numpy()
        out.append(z)
        if (i // batch_size) % 20 == 0:
            print(f"  ijepa encode: {i + len(batch_rows)}/{len(rows)}", flush=True)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def encode_bert_titles(
    rows: list[dict],
    cache_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode page titles with frozen BERT-base."""
    from visword.hf_dns_shim import install as _install_dns_shim
    _install_dns_shim()
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained(
        "bert-base-uncased", use_safetensors=True
    ).to(device).eval()
    out: list[np.ndarray] = []
    for i in range(0, len(rows), batch_size):
        texts = [r.get("title", "") for r in rows[i : i + batch_size]]
        enc = tok(
            texts, padding="max_length", truncation=True,
            max_length=64, return_tensors="pt"
        ).to(device)
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        v = F.normalize(v, dim=-1)
        out.append(v.cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Alignment metrics
# ---------------------------------------------------------------------------


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment (linear kernel) between two feature matrices.

    Kornblith et al., ICML 2019.
    """
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # HSIC estimator
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    hsic_xy = np.trace(XtX @ YtY)   # not actually HSIC, but proportional
    hsic_xx = np.trace(XtX @ XtX)
    hsic_yy = np.trace(YtY @ YtY)
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Orthogonal Procrustes distance: min ||X @ R - Y||_F over rotations R.

    Lower = more aligned. Returns normalised distance.
    """
    # Center and scale
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    X = X / (np.linalg.norm(X, 'fro') + 1e-10)
    Y = Y / (np.linalg.norm(Y, 'fro') + 1e-10)
    # Truncate to the smaller dimension
    d = min(X.shape[1], Y.shape[1])
    X, Y = X[:, :d], Y[:, :d]
    # SVD of Y^T X
    U, _, Vt = np.linalg.svd(Y.T @ X, full_matrices=False)
    return float(np.linalg.norm(X @ (Vt.T @ U.T) - Y, 'fro'))


def linear_regression_r2(X: np.ndarray, Y: np.ndarray) -> float:
    """R² of the linear map X → Y via least squares.

    R² = 1 means the I-JEPA features perfectly predict BERT embeddings
    through a linear transformation.
    """
    # Center
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    # Least squares: W = (X^T X)^{-1} X^T Y  →  Y_hat = X W
    W, _, _, _ = np.linalg.lstsq(X_c, Y_c, rcond=None)
    Y_hat = X_c @ W
    ss_res = np.sum((Y_c - Y_hat) ** 2)
    ss_tot = np.sum(Y_c ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def retrieval_via_linear_map(
    X: np.ndarray, Y: np.ndarray, k_values: list[int]
) -> dict:
    """Text-to-image retrieval through a learned linear map.

    Fit W on the full set (this is a probing metric, not a model selection metric),
    project X → X @ W, then do cosine retrieval.
    """
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    W, _, _, _ = np.linalg.lstsq(X_c, Y_c, rcond=None)
    X_proj = X_c @ W

    # L2-normalise
    X_proj = X_proj / (np.linalg.norm(X_proj, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)

    sim = Y_norm @ X_proj.T   # (Q, P) — text queries, image gallery
    Q = sim.shape[0]
    max_k = min(max(k_values), sim.shape[1])
    topk = np.argsort(-sim, axis=1)[:, :max_k]
    gt = np.arange(Q).reshape(-1, 1)
    recall = {}
    for k in k_values:
        hits = (topk[:, :min(k, max_k)] == gt).any(axis=1)
        recall[str(k)] = float(hits.mean())
    return recall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    p.add_argument("--cache-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "wiki_ss")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to fine-tuned I-JEPA checkpoint (omit for frozen baseline)")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to resolved YAML config for the checkpoint")
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "runs" / "ijepa_text_probes")
    p.add_argument("--label", type=str, default=None,
                   help="Label for this probe run (default: auto from checkpoint)")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = args.label or (
        args.checkpoint.parent.parent.name if args.checkpoint else "frozen_baseline"
    )
    print(f"=== I-JEPA text alignment probe: {label} ===")
    print(f"  device: {device}")
    print(f"  n_eval: {args.n_eval}")

    # Sample eval pages
    manifest = M.read_manifest(args.cache_dir)
    rows = manifest["rows"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(rows))
    eval_rows = [rows[i] for i in perm[3000 : 3000 + args.n_eval]]
    print(f"  sampled {len(eval_rows)} eval pages")

    # Encode
    t0 = time.time()
    X = encode_ijepa_finetuned(
        eval_rows, args.cache_dir, args.checkpoint, args.config,
        device, batch_size=args.batch_size)
    ijepa_time = time.time() - t0
    print(f"  I-JEPA features: {X.shape} in {ijepa_time:.1f}s")

    t0 = time.time()
    Y = encode_bert_titles(eval_rows, args.cache_dir, device)
    bert_time = time.time() - t0
    print(f"  BERT features:   {Y.shape} in {bert_time:.1f}s")

    # Compute alignment metrics
    print("\n=== Alignment Metrics ===")
    cka = linear_cka(X, Y)
    print(f"  Linear CKA:         {cka:.4f}")

    proc = procrustes_distance(X, Y)
    print(f"  Procrustes distance: {proc:.4f}")

    r2 = linear_regression_r2(X, Y)
    print(f"  Linear regression R²: {r2:.4f}")

    retrieval = retrieval_via_linear_map(X, Y, [1, 5, 10, 20])
    print(f"  Retrieval via linear map:")
    for k, v in retrieval.items():
        print(f"    R@{k}: {v:.4f}")

    # Save results
    report = {
        "label": label,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "n_eval": args.n_eval,
        "seed": args.seed,
        "ijepa_dim": int(X.shape[1]),
        "bert_dim": int(Y.shape[1]),
        "alignment": {
            "linear_cka": cka,
            "procrustes_distance": proc,
            "linear_regression_r2": r2,
        },
        "retrieval_via_linear_map": retrieval,
        "encode_time_s": {
            "ijepa": round(ijepa_time, 1),
            "bert": round(bert_time, 1),
        },
    }
    out_path = args.out_dir / f"probe_{label}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

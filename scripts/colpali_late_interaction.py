#!/usr/bin/env python3
"""ColPali-style late-interaction Protocol-A on a frozen image encoder.

Conventional retrieval pools per-patch features into one vector per
image. ColPali (Faysse et al.\ ICLR 2025) keeps every patch token
and scores via late interaction: for each query patch, take the max
cosine similarity against any document patch, then sum across query
patches. This skips the aggregation step entirely.

We test the zero-shot version: extract per-patch tokens from a
frozen CLIP-ViT-B/16 image encoder (the strongest zero-shot in our
grid), score Protocol-A with late interaction.

For each crop we get $T = 197$ tokens (1 CLS + 196 patches at
$224{\times}224$ with patch 16). Document = page; per-page tokens
are the union of all that page's crop tokens (so an N_crop=4 page
has $4 \times 196$ document patches). This is heavy: 2000 pages
$\times 4$ crops $\times 196$ patches $\approx$ 1.5M document
tokens; for each query crop (196 tokens), we compute a $196 \times
1.5M$ similarity matrix, take row-wise max, sum.

To keep memory bounded we batch document tokens across pages and
do max-sim incrementally.
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
from visword.data.light_dataset import default_transform


@torch.no_grad()
def encode_clip_patches(crops: list[Image.Image], device: torch.device,
                        batch_size: int = 8) -> torch.Tensor:
    """Return per-crop patch token features, (n_crops, T, D)."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai")
    model = model.to(device).eval()
    visual = model.visual
    tf = default_transform()

    # Inline ImageNet→CLIP renorm.
    in_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    in_std  = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    cl_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device).view(1, 3, 1, 1)
    cl_std  = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device).view(1, 3, 1, 1)

    out: list[torch.Tensor] = []
    for i in range(0, len(crops), batch_size):
        batch = torch.stack([tf(c) for c in crops[i : i + batch_size]]).to(device)
        x = (batch * in_std + in_mean - cl_mean) / cl_std
        x = visual.conv1(x)                                # (B, 768, 14, 14)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)       # (B, 196, 768)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(
            B, 1, C, dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                     # (B, 197, 768)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)                              # (T, B, D)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)                              # (B, T, D)
        x = visual.ln_post(x)
        x = F.normalize(x, p=2, dim=-1)                     # (B, T, D)
        out.append(x.cpu())
    return torch.cat(out, dim=0)                            # (N, T, D)


def late_interaction_score(query_tokens: torch.Tensor,    # (Tq, D)
                           doc_tokens: torch.Tensor,      # (Td, D)
                           ) -> float:
    """ColBERT-style: sum over query tokens of (max over document tokens
    of cosine similarity). Tokens are L2-normed."""
    sim = query_tokens @ doc_tokens.T                      # (Tq, Td)
    return float(sim.max(dim=1).values.sum())


def _load_eval_pages(cache_dir: Path, num_pages: int, seed: int) -> list[dict]:
    manifest = M.read_manifest(cache_dir)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(manifest["num_rows"])
    return [manifest["rows"][i] for i in perm[:num_pages]]


def _crop_pages(rows: list[dict], cache_dir: Path,
                cropper: NonOverlappingCropper) -> tuple[list[Image.Image], np.ndarray]:
    pils, page_ids = [], []
    for local_idx, row in enumerate(rows):
        with Image.open(cache_dir / row["image_path"]) as im:
            im = im.convert("RGB")
            crops = cropper(im)
        for c in crops:
            pils.append(c.copy()); page_ids.append(local_idx)
    return pils, np.asarray(page_ids)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path,
                   default="/scratch/hkanpak21/VISWORD/data/wiki_ss")
    p.add_argument("--num-pages", type=int, default=500,
                   help="smaller default than Protocol-A because late "
                        "interaction is O(N_doc * Tq * Td) per query")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--include-cls", action="store_true",
                   help="include CLS token; default is patch tokens only")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # Sample pages, crop, encode all crops with CLIP per-patch tokens.
    rows = _load_eval_pages(args.cache_dir, args.num_pages, args.seed)
    cropper = NonOverlappingCropper(
        crop_size=490, overlap=0.0, min_text_ratio=0.05, target_size=224)
    t0 = time.time()
    crops, page_ids = _crop_pages(rows, args.cache_dir, cropper)
    print(f"  cropped {len(crops)} crops from {len(rows)} pages "
          f"in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    tok = encode_clip_patches(crops, device)               # (N, T, D)
    print(f"  encoded {tok.shape} in {time.time() - t0:.1f}s", flush=True)

    # Optionally drop CLS token (index 0).
    if not args.include_cls:
        tok = tok[:, 1:, :]
    N, T, D = tok.shape
    page_ids_t = torch.from_numpy(page_ids).long()

    # Build per-page document token bank: concat all crop tokens for the
    # page. Different pages may have different numbers of crops, so we
    # store a list of (P, T_p, D) tensors.
    unique_pages, compact = torch.unique(page_ids_t, sorted=True, return_inverse=True)
    P = int(unique_pages.numel())
    page_tokens: list[torch.Tensor] = []
    for p_id in range(P):
        mask = (compact == p_id)
        page_tokens.append(tok[mask].reshape(-1, D))       # (T_p_total, D)

    # For each query crop, compute late-interaction score against all P
    # page-token banks. Build the gallery vector for that page on the fly
    # under leave-one-crop-out aggregation: for the query's own page,
    # exclude that crop's tokens from the document-token bank.
    print(f"  computing late-interaction scores for {N} queries x {P} galleries...", flush=True)
    t0 = time.time()
    sim_matrix = torch.zeros(N, P)
    for i in range(N):
        q = tok[i]                                          # (T, D)
        my_page = int(compact[i].item())
        for p_id in range(P):
            if p_id == my_page:
                # Leave-one-out: exclude this crop's tokens from page bank
                mask = (compact == p_id)
                idx_in_page = torch.where(mask)[0]
                idx_keep = idx_in_page[idx_in_page != i]
                if len(idx_keep) == 0:
                    sim_matrix[i, p_id] = float("-inf")
                    continue
                doc = tok[idx_keep].reshape(-1, D)
            else:
                doc = page_tokens[p_id]
            sim_matrix[i, p_id] = late_interaction_score(q, doc)
        if i % 200 == 0:
            print(f"    query {i}/{N} ({(time.time()-t0):.1f}s elapsed)", flush=True)
    print(f"  scoring done in {time.time() - t0:.1f}s", flush=True)

    # Recall@k: was the source page in the top k?
    eligible = torch.tensor([
        (compact == int(compact[i])).sum() >= 2 for i in range(N)
    ])
    k_values = [1, 5, 10, 20]
    recall = {}
    for k in k_values:
        k_eff = min(k, P)
        topk = sim_matrix.topk(k_eff, dim=1).indices
        correct = (topk == compact.unsqueeze(1)).any(dim=1)
        recall[str(k)] = float(correct[eligible].float().mean()) if eligible.any() else 0.0

    same = sim_matrix[torch.arange(N), compact]
    diff = (sim_matrix.sum(dim=1) - same) / max(1, P - 1)
    payload = {
        "encoder": "clip_image",
        "method": "late_interaction (ColPali-style, zero-shot)",
        "include_cls_token": args.include_cls,
        "num_pages": P,
        "num_crops": N,
        "tokens_per_crop": T,
        "embedding_dim": D,
        "recall": recall,
        "sanity": {
            "same_page_score_mean": float(same[eligible].mean()),
            "diff_page_score_mean": float(diff[eligible].mean()),
            "gap": float(same[eligible].mean() - diff[eligible].mean()),
        },
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

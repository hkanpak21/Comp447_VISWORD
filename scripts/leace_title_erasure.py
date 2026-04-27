#!/usr/bin/env python3
"""LEACE causal title-region erasure for zero-shot encoders.

We have, for each encoder, two descriptor caches at the same 2000
pages: one with original screenshots, one with the top 15% painted
white. The *title-region direction* in feature space is

    delta_i = phi(crop_i^orig) - phi(crop_i^blank)

stacked across all crops. We treat ``delta`` as a continuous protected
attribute and apply LEACE (Belrose 2023) to project it out of the
*original* descriptors. Re-running Protocol-A on the erased
descriptors then asks: is the title-region's effect purely linear?

- LEACE drops retrieval by roughly the manual-blank delta -> linear
  story holds, the title-region acts via a single direction.
- LEACE drops more -> linear projection sweeps up extra structure
  correlated with the title direction.
- LEACE drops less -> manual blanking changes more than just one
  linear direction (eg. it removes confusable boilerplate which
  happens to be off-axis).

Usage:
    python -m scripts.leace_title_erasure \
        --orig runs/_zeroshot_descr/clip_image_orig.npz \
        --blank runs/_zeroshot_descr/clip_image_blank15.npz \
        --out runs/_zeroshot_descr/clip_image_leace.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from visword.analysis.leace import fit_leace
from visword.eval_phase1_holdout import compute_protocol_a_recall


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig", type=Path, required=True,
                   help=".npz with emb (N, D) and page_ids (N,) for the "
                        "un-blanked encoding.")
    p.add_argument("--blank", type=Path, required=True,
                   help=".npz with emb (N, D) and page_ids (N,) for the "
                        "title-blanked encoding (must be the same crop set).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1,
                   help="rank of the protected attribute Z. Default 1 "
                        "projects per-crop title deltas onto their top "
                        "principal direction (one scalar per crop) so "
                        "LEACE erases a single feature-space axis. Use a "
                        "larger value to erase a low-dim subspace; using "
                        "Z = full delta matrix (k = d) over-erases the "
                        "entire feature space.")
    args = p.parse_args()

    orig = np.load(args.orig)
    blank = np.load(args.blank)
    X = torch.from_numpy(orig["emb"]).float()
    Xb = torch.from_numpy(blank["emb"]).float()
    pids = torch.from_numpy(orig["page_ids"]).long()
    if X.shape != Xb.shape:
        raise ValueError(f"shape mismatch: orig {X.shape} vs blank {Xb.shape}")
    if not np.array_equal(orig["page_ids"], blank["page_ids"]):
        raise ValueError("page_ids differ between orig and blank caches")
    print(f"loaded {X.shape[0]} crops at d={X.shape[1]}", flush=True)

    # Title-region delta per crop, projected to a low-rank Z to keep LEACE
    # from erasing the entire feature space (rank(Cxz) caps at min(d, k)).
    delta = X - Xb                                                  # (N, D)
    delta_c = delta - delta.mean(0, keepdim=True)
    U, S, _ = torch.linalg.svd(delta_c, full_matrices=False)
    Z = U[:, : args.rank] * S[: args.rank]                          # (N, rank)
    explained = float((S[: args.rank] ** 2).sum() / (S ** 2).sum())
    print(f"Z rank={args.rank}, captures {explained:.3f} of delta variance",
          flush=True)

    eraser = fit_leace(X, Z)
    X_erased = eraser.erase(X)
    X_erased = F.normalize(X_erased, p=2, dim=-1)

    # Sanity: how much of the title direction survives in X_erased? Project
    # onto the top-1 PCA direction of the deltas.
    z_dir = U[:, 0]
    z_dir = (delta_c.T @ z_dir)
    z_dir = z_dir / max(z_dir.norm().item(), 1e-8)
    proj_before = float((X.float() @ z_dir).abs().mean())
    proj_after = float((X_erased @ z_dir).abs().mean())

    R_orig = compute_protocol_a_recall(
        F.normalize(X, p=2, dim=-1), pids, k_values=[1, 5, 10, 20])
    R_blank = compute_protocol_a_recall(
        F.normalize(Xb, p=2, dim=-1), pids, k_values=[1, 5, 10, 20])
    R_leace = compute_protocol_a_recall(
        X_erased, pids, k_values=[1, 5, 10, 20])

    payload = {
        "orig_npz": str(args.orig),
        "blank_npz": str(args.blank),
        "num_crops": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "z_rank": int(args.rank),
        "z_explained_variance": explained,
        "leace_proj_norm_before": proj_before,
        "leace_proj_norm_after": proj_after,
        "recall_orig": R_orig["recall"],
        "recall_blank": R_blank["recall"],
        "recall_leace": R_leace["recall"],
        "delta_orig_blank_R10": R_blank["recall"]["10"] - R_orig["recall"]["10"],
        "delta_orig_leace_R10": R_leace["recall"]["10"] - R_orig["recall"]["10"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

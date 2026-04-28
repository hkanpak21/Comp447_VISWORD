"""LEACE rank sweep on a saved (orig, blank) descriptor pair."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from visword.analysis.leace import fit_leace
from visword.eval_phase1_holdout import compute_protocol_a_recall


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--orig", type=Path, required=True)
    p.add_argument("--blank", type=Path, required=True)
    p.add_argument("--ranks", nargs="+", type=int,
                   default=[1, 4, 16, 64, 256, 1024, 4000])
    args = p.parse_args()

    orig = np.load(args.orig); blank = np.load(args.blank)
    X = torch.from_numpy(orig["emb"]).float()
    Xb = torch.from_numpy(blank["emb"]).float()
    pids = torch.from_numpy(orig["page_ids"]).long()
    delta = X - Xb
    delta_c = delta - delta.mean(0, keepdim=True)
    U, S, _ = torch.linalg.svd(delta_c, full_matrices=False)

    R_orig = compute_protocol_a_recall(F.normalize(X, p=2, dim=-1), pids, [10])["recall"]["10"]
    R_blank = compute_protocol_a_recall(F.normalize(Xb, p=2, dim=-1), pids, [10])["recall"]["10"]
    print(f"orig  R@10 = {R_orig:.3f}")
    print(f"blank R@10 = {R_blank:.3f}  Delta = {R_blank - R_orig:+.3f}")
    print(f"{'rank':>5} {'expl_var':>9} {'R@10_LEACE':>10} {'Delta':>8}")
    for rank in args.ranks:
        rank = min(rank, U.shape[1])
        Z = U[:, :rank] * S[:rank]
        expl = float((S[:rank] ** 2).sum() / (S ** 2).sum())
        eraser = fit_leace(X, Z)
        Xe = F.normalize(eraser.erase(X), p=2, dim=-1)
        R = compute_protocol_a_recall(Xe, pids, [10])["recall"]["10"]
        print(f"{rank:>5} {expl:>9.3f} {R:>10.3f} {R - R_orig:>+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

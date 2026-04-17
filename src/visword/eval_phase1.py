"""Phase-1 recall: same-page non-overlapping-crop retrieval (PROJECT_SPEC.md §7).

Used two ways:
  * ``phase1_recall(model, dataset, k_values)`` — library entry called
    mid-training by ``visword.train`` for eval_every (Phase C).
  * ``main()`` — CLI that loads a checkpoint, runs the full eval against
    the Phase-1 held-out set, and writes ``phase1_recall.json`` into the
    run directory per the §7 schema (Phase D; skeleton here).

Protocol:
  * For each page in the eval set, generate all non-overlapping crops.
  * Encode every crop into the model's descriptor space.
  * For each crop as a query, rank every *other* crop by cosine similarity;
    the query is "recalled at K" iff at least one same-page crop lies in
    the top K. Mean over all queries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from visword.data.light_dataset import LightWikiScreenshotDataset


@torch.no_grad()
def _encode_all_crops(
    model: torch.nn.Module,
    dataset: LightWikiScreenshotDataset,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every crop in ``dataset`` (using ``iter_all_crops``).

    Returns:
        embeddings: ``(N_crops, D)`` L2-normed (the model already normalises).
        page_ids:   ``(N_crops,)`` page index (local to ``dataset.rows``).
    """
    model.eval()
    embs: list[torch.Tensor] = []
    page_ids: list[int] = []

    for local_idx, crops in dataset.iter_all_crops():
        if not crops:
            continue
        x = torch.stack(crops).to(device)
        z = model(x)
        embs.append(z.detach().cpu())
        page_ids.extend([local_idx] * x.shape[0])

    if not embs:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    return torch.cat(embs, dim=0), torch.as_tensor(page_ids, dtype=torch.long)


def compute_recall_at_k(
    embeddings: torch.Tensor,
    page_ids: torch.Tensor,
    k_values: list[int],
) -> dict[str, float | dict | int]:
    """Page-level recall@K for same-page retrieval.

    A query is "correct at K" iff any of its top-K non-self neighbours
    share its page id.
    """
    if embeddings.numel() == 0:
        return {
            "num_crops": 0,
            "num_pages": 0,
            "recall": {str(k): 0.0 for k in k_values},
            "sanity": {
                "same_page_sim_mean": 0.0,
                "diff_page_sim_mean": 0.0,
                "gap": 0.0,
                "monotonic": True,
            },
        }

    sim = embeddings @ embeddings.T
    n = sim.shape[0]
    sim.fill_diagonal_(float("-inf"))  # never retrieve self

    same_mask = page_ids.unsqueeze(0) == page_ids.unsqueeze(1)
    diff_mask = ~same_mask & ~torch.eye(n, dtype=torch.bool)

    # Sanity stats
    same_vals = sim[same_mask & ~torch.eye(n, dtype=torch.bool)]
    diff_vals = sim[diff_mask]
    same_pos = same_vals[torch.isfinite(same_vals)]
    same_mean = float(same_pos.mean()) if same_pos.numel() > 0 else float("nan")
    diff_mean = float(diff_vals.mean()) if diff_vals.numel() > 0 else float("nan")

    recall: dict[str, float] = {}
    for k in sorted(k_values):
        topk_idx = sim.topk(k, dim=1).indices                 # (N, k)
        topk_same = same_mask.gather(1, topk_idx)
        # Only count queries that have at least one non-self same-page neighbour.
        eligible = (same_mask & ~torch.eye(n, dtype=torch.bool)).any(dim=1)
        if eligible.any():
            recall[str(k)] = float(topk_same[eligible].any(dim=1).float().mean())
        else:
            recall[str(k)] = 0.0

    monotonic = all(recall[str(a)] <= recall[str(b)] + 1e-6
                    for a, b in zip(sorted(k_values), sorted(k_values)[1:]))

    return {
        "num_crops": n,
        "num_pages": int(page_ids.unique().numel()),
        "recall": recall,
        "sanity": {
            "same_page_sim_mean": same_mean,
            "diff_page_sim_mean": diff_mean,
            "gap": same_mean - diff_mean if (same_mean == same_mean and diff_mean == diff_mean) else 0.0,
            "monotonic": bool(monotonic),
        },
    }


def phase1_recall(
    model: torch.nn.Module,
    dataset: LightWikiScreenshotDataset,
    *,
    k_values: list[int],
    device: torch.device | str | None = None,
) -> dict:
    """One-shot Phase-1 recall (used by ``visword.train`` for eval_every)."""
    if device is None:
        device = next(model.parameters()).device
    embs, page_ids = _encode_all_crops(model, dataset, device)
    return compute_recall_at_k(embs, page_ids, k_values)


# ---------------------------------------------------------------------------
# CLI — fleshed out in Phase D; for Phase C we only need the library API.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default="best_phase1.pt",
                   help="filename under <run-dir>/checkpoints/")
    args = p.parse_args(argv)

    run_dir = args.run_dir.resolve()
    ckpt = run_dir / "checkpoints" / args.checkpoint
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}. Phase D will populate this.")

    # Full Phase-1 protocol lands in Phase D (ref to PROJECT_SPEC.md §7).
    # For now, fail loudly rather than fake a number — CONTEXT.md session-2 lesson.
    raise SystemExit(
        "Full eval_phase1 CLI arrives in Phase D. "
        "Phase C only exposes the phase1_recall(model, dataset, ...) library API."
    )


if __name__ == "__main__":
    raise SystemExit(main())

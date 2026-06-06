"""Ticket 02 — page-level same-page re-identification scoring (D13/D14).

The canonical public name for the retrieval metric: given crop/view embeddings and
their page ids, score recall@k where a query (a crop/view of a page) must retrieve
its OWN page from a gallery of per-page mean embeddings, with the query's own page
vector recomputed leaving the query out (forbids trivial self-match).

This is a thin, documented alias over the existing, tested implementation in
``eval_phase1_holdout`` — one source of truth, no duplicated logic.
"""
from __future__ import annotations

import torch

from visword.eval_phase1_holdout import compute_protocol_a_recall


def page_reid_recall(
    embeddings: torch.Tensor,           # (N, D), L2-normed crop/view embeddings
    page_ids: torch.Tensor,             # (N,) page id per row
    k_values: tuple[int, ...] = (1, 5, 10, 20),
) -> dict:
    """Page-level same-page re-identification recall@k (leave-one-out gallery).

    Returns the dict from :func:`compute_protocol_a_recall`:
    ``num_crops``, ``num_pages``, ``num_queries_eligible``, ``recall`` (``{str(k): float}``),
    and ``sanity`` (``same_page_sim_mean`` / ``diff_page_sim_mean`` / ``gap``).
    Pages with a single view are ineligible as queries (LOO undefined).
    """
    return compute_protocol_a_recall(embeddings, page_ids, list(k_values))


__all__ = ["page_reid_recall"]

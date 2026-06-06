"""Ticket 02 — page-level same-page re-identification scoring (TESTED module).

Pure-tensor tests on synthetic embeddings (no model load), so they run in the CPU
suite. Cover the PRD's scoring contract: recall monotonic in k; same-page similarity
above different-page; leave-one-out gallery is actually applied (forbids self-match);
single-view pages are ineligible queries.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from visword.page_reid import page_reid_recall


def _synthetic(P: int = 5, C: int = 4, D: int = 8, sep: float = 10.0, seed: int = 0):
    """P pages x C crops; tight per-page clusters separated by ``sep``."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(P, D, generator=g) * sep
    embs, pids = [], []
    for p in range(P):
        embs.append(centers[p] + torch.randn(C, D, generator=g) * 0.01)
        pids += [p] * C
    return F.normalize(torch.cat(embs), dim=1), torch.tensor(pids)


def test_recall_is_monotonic_and_same_above_diff() -> None:
    emb, pid = _synthetic()
    r = page_reid_recall(emb, pid, k_values=(1, 5, 10, 20))
    rec = r["recall"]
    assert rec["1"] <= rec["5"] <= rec["10"] <= rec["20"]
    assert r["sanity"]["same_page_sim_mean"] > r["sanity"]["diff_page_sim_mean"]


def test_well_separated_pages_give_perfect_recall_at_1() -> None:
    emb, pid = _synthetic(sep=50.0)
    r = page_reid_recall(emb, pid, k_values=(1,))
    assert r["recall"]["1"] == 1.0


def test_single_view_pages_are_ineligible_queries() -> None:
    emb, pid = _synthetic(P=2, C=3, sep=50.0, seed=1)          # 6 eligible crops
    single = F.normalize(torch.randn(1, emb.shape[1], generator=torch.Generator().manual_seed(9)), dim=1)
    emb = torch.cat([emb, single])
    pid = torch.cat([pid, torch.tensor([2])])                 # a singleton page
    r = page_reid_recall(emb, pid, k_values=(1,))
    assert r["num_queries_eligible"] == 6


def test_leave_one_out_is_applied_not_trivial_self_match() -> None:
    # Page 0 = {A, B} (orthogonal). Page 1 = {C} with C ~ A.
    # WITHOUT LOO, query B would match page-0 mean (which contains B) -> recall 0.5.
    # WITH LOO, B's page-0 gallery = A only (B excluded): B.A = 0 but B.C = 0.1 -> B
    # ranks page 1 first; A likewise misses -> recall@1 = 0.0. So 0.0 proves LOO.
    A = torch.tensor([1.0, 0, 0, 0]); B = torch.tensor([0.0, 1, 0, 0]); C = torch.tensor([0.9, 0.1, 0, 0])
    emb = F.normalize(torch.stack([A, B, C]), dim=1)
    pid = torch.tensor([0, 0, 1])
    r = page_reid_recall(emb, pid, k_values=(1,))
    assert r["num_queries_eligible"] == 2          # C's page is a singleton
    assert r["recall"]["1"] == 0.0                 # would be 0.5 without LOO

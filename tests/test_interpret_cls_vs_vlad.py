"""Phase E / E3 — CLS-vs-VLAD decomposition is exact (TESTS.md E3).

Does NOT need a GPU or a trained model: the maths is a pure tensor slice
+ inner product, so we test with synthetic L2-normed descriptors on CPU.
That's what TESTS.md specifies too (E3 is the only Phase-E item whose
correctness is deterministic).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from visword.interpret.cls_vs_vlad import (
    CLSVsVLADResult,
    pairwise_cos_by_half,
    plot_cls_vs_vlad,
)


def _fake_descriptors(B: int = 8, vlad_dim: int = 64, cls_dim: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(B, vlad_dim + cls_dim, generator=g)
    z = F.normalize(raw, p=2, dim=-1)
    # Two pages of 4 crops each.
    labels = torch.arange(2).repeat_interleave(B // 2)
    return z, labels, vlad_dim


def test_halves_sum_to_full():
    z, labels, vlad_dim = _fake_descriptors()
    full_sim = z @ z.T
    vlad_sim = z[:, :vlad_dim] @ z[:, :vlad_dim].T
    cls_sim = z[:, vlad_dim:] @ z[:, vlad_dim:].T
    assert torch.allclose(full_sim, vlad_sim + cls_sim, atol=1e-6)


def test_decomposition_matches_exact_similarity():
    z, labels, vlad_dim = _fake_descriptors()
    result = pairwise_cos_by_half(z, labels, vlad_dim)

    # The headline E3 check: same-page VLAD + same-page CLS ≈ same-page full.
    assert abs(
        result.same_page_full_cos
        - (result.same_page_vlad_cos + result.same_page_cls_cos)
    ) < 1e-4, result

    # Same invariant for diff-page.
    assert abs(
        result.diff_page_full_cos
        - (result.diff_page_vlad_cos + result.diff_page_cls_cos)
    ) < 1e-4, result


def test_plot_writes_non_empty_png(tmp_path):
    z, labels, vlad_dim = _fake_descriptors()
    result = pairwise_cos_by_half(z, labels, vlad_dim)
    out = plot_cls_vs_vlad(result, tmp_path / "cls_vs_vlad.png")
    assert out.exists() and out.stat().st_size > 0

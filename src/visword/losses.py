"""Contrastive losses used in training (PROJECT_SPEC.md §3.1 / §6).

All three operate on a batch of L2-normed embeddings ``z: (N, D)`` and
integer page labels ``labels: (N,)``; two samples with the same label are
positives, different labels are negatives. ``(N, D)`` is already the
flattened ``(B * K, D)`` of a multi-positive batch.

* ``InfoNCEMultiPositive`` — per-anchor cross-entropy over its positives,
  with a temperature ``tau``. Reduces to vanilla InfoNCE when K=2.
* ``MultiSimilarity`` — Wang et al. CVPR 2019 (default hparams alpha=2,
  beta=50, base=0.5). Sensitive to hard pairs; the paper's loss for
  screenshot retrieval (PROJECT_SPEC.md §3.1 default).
* ``Triplet`` — batch-hard triplet with margin.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------


def _pairwise_cosine(z: torch.Tensor) -> torch.Tensor:
    """``z`` assumed L2-normed → cosine similarity = ``z @ z.T``."""
    return z @ z.T


def _label_masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos_mask, self_mask)`` as bool tensors of shape ``(N, N)``.

    ``pos_mask[i, j]`` is True iff label[i] == label[j] and i != j.
    ``self_mask[i, j]`` is True iff i == j (diagonal).
    """
    self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    return pos_mask, self_mask


# ---------------------------------------------------------------------------
# InfoNCE multi-positive
# ---------------------------------------------------------------------------


class InfoNCEMultiPositive(nn.Module):
    """Per-anchor multi-positive InfoNCE with negatives-only denominator.

    For anchor ``i`` with positive set ``P_i`` and negative set ``N_i``::

        L_i = (1/|P_i|) * sum_{p in P_i}
              log( 1 + exp(log_sum_exp_neg_i - s_{i,p} / tau) )
            = (1/|P_i|) * sum_{p in P_i}
              -log( exp(s_{i,p}/tau)
                    / ( exp(s_{i,p}/tau) + sum_{n in N_i} exp(s_{i,n}/tau) ) )

    With labels-clustered embeddings (same-class sims ≈ 1, diff-class ≈ -1)
    and a reasonable temperature, this loss drives to 0. Contrast with the
    SupCon formulation which includes positives in the denominator and has
    a floor of ``log |P_i|``.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = _pairwise_cosine(z) / self.temperature
        pos_mask, self_mask = _label_masks(labels)
        neg_mask = ~pos_mask & ~self_mask

        # log-sum-exp of negatives only, per anchor row.
        neg_only = sim.masked_fill(~neg_mask, float("-inf"))
        neg_lse = torch.logsumexp(neg_only, dim=1)                 # (N,)

        # log(1 + exp(neg_lse - sim)) broadcast over all (i, j) pairs.
        per_pair_ce = F.softplus(neg_lse.unsqueeze(1) - sim)       # (N, N)

        # Mean per-anchor over positives only.
        pos_mask_f = pos_mask.to(per_pair_ce.dtype)
        num_pos = pos_mask_f.sum(dim=1)
        per_anchor = (per_pair_ce * pos_mask_f).sum(dim=1) / num_pos.clamp(min=1.0)

        has_pos = num_pos > 0
        if not has_pos.any():
            return torch.zeros((), device=z.device, dtype=z.dtype)
        return per_anchor[has_pos].mean()


# ---------------------------------------------------------------------------
# Multi-Similarity loss (Wang et al., CVPR 2019)
# ---------------------------------------------------------------------------


class MultiSimilarity(nn.Module):
    """Classic MS loss — hard-pair aware, the paper's default for screenshots.

    Args:
        alpha: positive-term slope (default 2.0).
        beta:  negative-term slope (default 50.0 per Wang et al.).
        base:  similarity offset aka lambda (default 0.5).
        margin: pair-mining margin ε; positives below (min_neg + margin) and
            negatives above (max_pos - margin) are kept. ``0.0`` keeps all.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 50.0,
        base: float = 0.5,
        margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.margin = margin

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = _pairwise_cosine(z)
        pos_mask, self_mask = _label_masks(labels)
        neg_mask = ~pos_mask & ~self_mask

        # Per-row hardest positive / negative for mining.
        pos_sim = sim.masked_fill(~pos_mask, float("inf"))
        neg_sim = sim.masked_fill(~neg_mask, float("-inf"))
        min_pos = pos_sim.min(dim=1).values          # hardest positive (lowest sim)
        max_neg = neg_sim.max(dim=1).values          # hardest negative (highest sim)

        # Mining: keep positives with sim < max_neg + margin, negatives with sim > min_pos - margin.
        kept_pos = pos_mask & (sim < (max_neg.unsqueeze(1) + self.margin))
        kept_neg = neg_mask & (sim > (min_pos.unsqueeze(1) - self.margin))

        losses: list[torch.Tensor] = []
        for i in range(z.shape[0]):
            pos_i = sim[i][kept_pos[i]]
            neg_i = sim[i][kept_neg[i]]
            if pos_i.numel() == 0 or neg_i.numel() == 0:
                continue
            pos_term = (1.0 / self.alpha) * torch.log1p(
                torch.exp(-self.alpha * (pos_i - self.base)).sum()
            )
            neg_term = (1.0 / self.beta) * torch.log1p(
                torch.exp(self.beta * (neg_i - self.base)).sum()
            )
            losses.append(pos_term + neg_term)

        if not losses:
            return torch.zeros((), device=z.device, dtype=z.dtype, requires_grad=True)
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Triplet (batch-hard)
# ---------------------------------------------------------------------------


class Triplet(nn.Module):
    """Batch-hard triplet: hardest positive - hardest negative + margin."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = _pairwise_cosine(z)
        pos_mask, self_mask = _label_masks(labels)
        neg_mask = ~pos_mask & ~self_mask

        pos_sim = sim.masked_fill(~pos_mask, float("inf"))
        neg_sim = sim.masked_fill(~neg_mask, float("-inf"))
        hardest_pos = pos_sim.min(dim=1).values
        hardest_neg = neg_sim.max(dim=1).values

        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return torch.zeros((), device=z.device, dtype=z.dtype, requires_grad=True)
        d_pos = 1.0 - hardest_pos[valid]   # cosine distance
        d_neg = 1.0 - hardest_neg[valid]
        return F.relu(d_pos - d_neg + self.margin).mean()


# ---------------------------------------------------------------------------

LossName = Literal["infonce", "multisim", "triplet"]


def build_loss(name: LossName, **kwargs) -> nn.Module:
    name = name.lower()  # type: ignore[assignment]
    if name == "infonce":
        return InfoNCEMultiPositive(**kwargs)
    if name == "multisim":
        return MultiSimilarity(**kwargs)
    if name == "triplet":
        return Triplet(**kwargs)
    raise ValueError(f"unknown loss {name!r}")

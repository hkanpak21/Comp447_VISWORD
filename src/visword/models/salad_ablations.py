"""SALAD ablation variants — isolate which methodological piece earns the gain.

Subclasses :class:`OfficialSALAD` and overrides ``forward`` per ablation mode.
Adds **no new parameters** (state_dict is identical to the parent), so a "full"
mode forward is byte-equivalent to the vendored implementation; checkpoints
trained with the vendored class load cleanly here.

Modes:
  * ``full``           — vendored behaviour (token + Sinkhorn-VLAD, 8448-d).
  * ``token_only``     — only the ``token_features`` MLP branch (256-d).
  * ``vlad_only``      — only the Sinkhorn-VLAD branch (K*cluster_dim, 8192-d).
  * ``softmax_assign`` — same shape as ``full`` (8448-d), but the per-patch
    cluster assignment is a plain softmax over clusters instead of the
    Sinkhorn / dustbin OT solver. Tests whether the OT structure (balanced
    marginals + dustbin slack) is what carries the win.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from visword.models.salad_bridge import OfficialSALAD


_VALID_ABLATIONS = {"full", "token_only", "vlad_only", "softmax_assign"}


class AblatedSALAD(OfficialSALAD):
    def __init__(self, *args, ablation: str = "full", sinkhorn_iters: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; expected one of {_VALID_ABLATIONS}")
        self.ablation = ablation
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, x):
        feats, cls = x

        if self.ablation == "token_only":
            return F.normalize(self.token_features(cls), p=2, dim=-1)

        s = self.score(feats).flatten(2)        # (B, K, N)
        f = self.cluster_features(feats).flatten(2)  # (B, cluster_dim, N)

        if self.ablation == "softmax_assign":
            p = F.softmax(s, dim=1)             # (B, K, N), per-patch normalised
        else:
            from models.aggregators.salad import get_matching_probs
            log_p = get_matching_probs(s, self.dust_bin, self.sinkhorn_iters)
            p = torch.exp(log_p)[:, :-1, :]     # drop dustbin row

        # VLAD: per-cluster sum of weighted local features → (B, K, cluster_dim)
        p_e = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f_e = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        vlad = F.normalize((f_e * p_e).sum(dim=-1), p=2, dim=1).flatten(1)

        if self.ablation == "vlad_only":
            return F.normalize(vlad, p=2, dim=-1)

        t = F.normalize(self.token_features(cls), p=2, dim=-1)
        return F.normalize(torch.cat([t, vlad], dim=-1), p=2, dim=-1)


def descriptor_dim_for(ablation: str, num_clusters: int, cluster_dim: int, token_dim: int) -> int:
    if ablation == "token_only":
        return token_dim
    if ablation == "vlad_only":
        return num_clusters * cluster_dim
    return num_clusters * cluster_dim + token_dim


__all__ = ["AblatedSALAD", "descriptor_dim_for"]

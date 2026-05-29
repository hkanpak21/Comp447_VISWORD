"""Adapter head factory for frozen-backbone experiments.

Three capacity levels for the ablation grid:
  - ``linear``:     Linear(d_in → d_out)               ~197k params (768→256)
  - ``mlp``:        Linear → GELU → Linear              ~524k params
  - ``bottleneck``: Linear(d_in → 64) → GELU → Linear   ~65k params

All adapters output ``d_out``-dim vectors (not L2-normed here — the caller
normalises after the adapter so gradient flow is clean).
"""
from __future__ import annotations

import torch.nn as nn


def build_adapter(kind: str, d_in: int, d_out: int = 256) -> nn.Module:
    """Return a trainable adapter head.

    Parameters
    ----------
    kind : {"linear", "mlp", "bottleneck"}
    d_in : int
        Input dimension from the frozen backbone.
    d_out : int
        Output descriptor dimension.
    """
    if kind == "linear":
        return nn.Linear(d_in, d_out)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(d_in, 512),
            nn.GELU(),
            nn.Linear(512, d_out),
        )
    if kind == "bottleneck":
        return nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, d_out),
        )
    raise ValueError(f"unknown adapter kind {kind!r}; expected linear/mlp/bottleneck")

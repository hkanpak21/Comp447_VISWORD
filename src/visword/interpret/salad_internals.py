"""Hook-based SALAD aggregator internals (PROJECT_SPEC.md §8).

The official aggregator is a black box from the outside: given
``(patches, cls_token)`` it returns one 8448-d descriptor and consumes
its intermediate tensors on the way. To inspect its behaviour we
register forward hooks on the **three MLP-producing submodules** inside
SALAD::

    aggregator.score              (B, num_channels, H, W) -> (B, num_clusters, H, W)
    aggregator.cluster_features   (B, num_channels, H, W) -> (B, cluster_dim, H, W)
    aggregator.token_features     (B, num_channels)       -> (B, token_dim)

Discovery is by **output shape**, not by name: we run one forward with
known-shape dummy inputs, record each leaf submodule's output shape,
and pick those matching the expected channel count. The resolved
submodule names are cached in ``<run_dir>/interpret/salad_hooks.json``
so repeated calls are deterministic.

Downstream:
  * The Sinkhorn OT computation from ``salad.get_matching_probs`` can
    then be re-run on the captured ``score`` tensor — this gives the
    doubly-stochastic assignment matrix + dustbin row, which §8 uses
    for cluster-map overlays and dustbin-mass analyses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from visword.models.salad_bridge import OfficialSALAD


# ---------------------------------------------------------------------------


@dataclass
class SaladHooks:
    """Names of the resolved submodules inside an OfficialSALAD aggregator."""
    score: str
    cluster_features: str
    token_features: str

    def to_dict(self) -> dict:
        return {"score": self.score, "cluster_features": self.cluster_features,
                "token_features": self.token_features}


def discover_salad_submodules(
    aggregator: OfficialSALAD,
    *,
    num_channels: int,
    H: int = 16,
    W: int = 16,
    device: torch.device | str = "cpu",
) -> SaladHooks:
    """Run one forward on dummy inputs and identify the three submodules by shape.

    Does not require a GPU. The aggregator weights are used as-is; we only
    care about per-submodule output shapes.
    """
    aggregator = aggregator.to(device).eval()
    num_clusters = aggregator.num_clusters
    cluster_dim = aggregator.cluster_dim
    token_dim = aggregator.token_dim

    patches = torch.zeros(1, num_channels, H, W, device=device)
    cls = torch.zeros(1, num_channels, device=device)

    captured: dict[str, tuple[int, ...]] = {}

    def make_hook(name: str):
        def _h(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                captured[name] = tuple(out.shape)
        return _h

    handles = []
    for name, module in aggregator.named_modules():
        if not name or any(True for _ in module.children()):
            continue   # skip container / root modules
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            _ = aggregator((patches, cls))
    finally:
        for h in handles:
            h.remove()

    # Identify by shape signatures:
    #   score            -> (1, num_clusters, H, W)
    #   cluster_features -> (1, cluster_dim,  H, W)
    #   token_features   -> (1, token_dim)
    want_score = (1, num_clusters, H, W)
    want_cluster = (1, cluster_dim, H, W)
    want_token = (1, token_dim)

    def _find(want: tuple[int, ...]) -> str:
        candidates = [name for name, shape in captured.items() if shape == want]
        if not candidates:
            raise RuntimeError(
                f"no submodule produced shape {want}; got: {captured}"
            )
        # When multiple leaf modules share the same output shape (e.g. a final
        # Conv2d followed by an identity), return the deepest (longest name)
        # as that's the one actually shaping the output.
        return max(candidates, key=len)

    return SaladHooks(
        score=_find(want_score),
        cluster_features=_find(want_cluster),
        token_features=_find(want_token),
    )


def save_hooks_json(hooks: SaladHooks, run_dir: Path) -> Path:
    out = Path(run_dir) / "interpret" / "salad_hooks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(hooks.to_dict(), indent=2))
    return out


def load_hooks_json(run_dir: Path) -> SaladHooks:
    data = json.loads((Path(run_dir) / "interpret" / "salad_hooks.json").read_text())
    return SaladHooks(**data)


# ---------------------------------------------------------------------------
# Capture score tensor + recompute Sinkhorn assignments (for dustbin / clusters)
# ---------------------------------------------------------------------------


def capture_score_tensor(
    aggregator: OfficialSALAD,
    patches: torch.Tensor,
    cls_token: torch.Tensor,
    score_submodule_name: str,
) -> torch.Tensor:
    """Forward the aggregator once; return its ``score`` submodule output.

    Shape: ``(B, num_clusters, H, W)``. This is the *logit* tensor that
    feeds into Sinkhorn; the normalised assignment requires recomputing
    ``salad.get_matching_probs`` on top.
    """
    target = dict(aggregator.named_modules())[score_submodule_name]
    captured: dict[str, torch.Tensor] = {}

    def _h(_m, _inp, out):
        captured["out"] = out.detach()

    handle = target.register_forward_hook(_h)
    try:
        with torch.no_grad():
            _ = aggregator((patches, cls_token))
    finally:
        handle.remove()
    return captured["out"]


def sinkhorn_assignment(
    score_tensor: torch.Tensor,
    dust_bin: torch.Tensor,
    *,
    num_iters: int = 3,
) -> torch.Tensor:
    """Run the §8 Sinkhorn normalisation on a captured score tensor.

    Args:
        score_tensor: ``(B, num_clusters, H, W)`` logits from
            ``aggregator.score``.
        dust_bin: ``aggregator.dust_bin`` scalar parameter.

    Returns:
        ``(B, num_clusters + 1, H*W)`` probabilities. The last row is the
        dustbin channel; rows 0..num_clusters-1 are the per-cluster masses.
    """
    from models.aggregators.salad import get_matching_probs   # from vendored repo

    flat = score_tensor.flatten(2)                            # (B, num_clusters, H*W)
    log_p = get_matching_probs(flat, dust_bin, num_iters=num_iters)
    return log_p.exp()


def dustbin_mass_fraction(assignment: torch.Tensor) -> float:
    """Fraction of total assignment mass that went to the dustbin row."""
    total = assignment.sum()
    if total <= 0:
        return float("nan")
    return float(assignment[:, -1, :].sum() / total)


__all__ = [
    "SaladHooks",
    "capture_score_tensor",
    "discover_salad_submodules",
    "dustbin_mass_fraction",
    "load_hooks_json",
    "save_hooks_json",
    "sinkhorn_assignment",
]

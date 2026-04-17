"""Phase B / B3 tests for contrastive losses (TESTS.md B3).

All three tests run on CPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from visword.losses import (
    InfoNCEMultiPositive,
    MultiSimilarity,
    Triplet,
    build_loss,
)


def _rand_normed(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_infonce_multi_positive_is_reasonable():
    # B=32 random embeddings with 4 positives each (8 classes).
    z = _rand_normed(32, 128, seed=0)
    labels = torch.arange(8).repeat_interleave(4)

    loss = InfoNCEMultiPositive(temperature=0.07)(z, labels)
    loss = float(loss)
    assert 1.0 <= loss <= 10.0, f"random-init InfoNCE out of range: {loss}"


def test_infonce_zero_when_perfect():
    # 4 classes, 4 exemplars each; same-class points identical, classes well-separated.
    class_protos = _rand_normed(4, 128, seed=1) * 10.0   # amplify separation
    z = class_protos.repeat_interleave(4, dim=0)
    z = F.normalize(z, dim=-1)
    labels = torch.arange(4).repeat_interleave(4)

    loss = float(InfoNCEMultiPositive(temperature=0.07)(z, labels))
    assert loss < 1e-3, f"perfect clustering should drive loss ≈ 0, got {loss}"


def test_multisim_runs_forward_backward():
    z = _rand_normed(16, 64, seed=2).requires_grad_(True)
    labels = torch.arange(4).repeat_interleave(4)
    loss = MultiSimilarity()(z, labels)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(loss)
    assert torch.isfinite(z.grad).all()


def test_triplet_basic():
    z = _rand_normed(16, 32, seed=3).requires_grad_(True)
    labels = torch.arange(4).repeat_interleave(4)
    loss = Triplet(margin=0.2)(z, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_build_loss_dispatch():
    for name, cls in [("infonce", InfoNCEMultiPositive),
                      ("multisim", MultiSimilarity),
                      ("triplet", Triplet)]:
        loss = build_loss(name)
        assert isinstance(loss, cls)

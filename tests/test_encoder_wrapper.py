"""Ticket 02 — shared model-wrapper (TESTED module).

Per-encoder contract: a forward gives the expected embedding size, unit length, and
is deterministic (same input -> same output) so the cross-model grid stays fair.
Marked ``integration`` (each encoder loads pretrained weights from the offline cache);
run on the cluster, not in the bare CPU collection.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from visword.config import Config
from visword.models.encoder_wrapper import ENCODER_NAMES, build_encoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_cfg() -> Config:
    # Validate the default config without env-expansion (we only read cfg.backbone.*).
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text())
    return Config.model_validate(raw)


@pytest.mark.integration
def test_mae_wrapper_dim_unitnorm_determinism() -> None:
    enc = build_encoder("mae").eval()
    assert enc.descriptor_dim == 768
    x = torch.randn(2, 3, 224, 224)
    z1, z2 = enc(x), enc(x)
    assert z1.shape == (2, 768)
    assert torch.allclose(z1.norm(p=2, dim=-1), torch.ones(2), atol=1e-4)
    assert torch.allclose(z1, z2, atol=1e-5), "mask_ratio=0 must make MAE deterministic"


@pytest.mark.integration
@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_each_encoder_dim_unitnorm_determinism(name: str) -> None:
    enc = build_encoder(name, _default_cfg()).eval()
    x = torch.randn(2, 3, 224, 224)
    z1, z2 = enc(x), enc(x)
    assert z1.shape == (2, enc.descriptor_dim)
    norms = z1.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), f"{name}: not unit-norm"
    assert torch.allclose(z1, z2, atol=1e-3), f"{name}: non-deterministic forward"

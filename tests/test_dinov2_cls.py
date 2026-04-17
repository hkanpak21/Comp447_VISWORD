"""Phase B / B2 tests for the DINOv2 CLS-only baseline (TESTS.md B2)."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


from visword.config import (   # noqa: E402
    BackboneConfig,
    Config,
    SaladConfig,
)
from visword.models.dinov2_cls import DINOv2CLS  # noqa: E402


def _default_cfg(tmp_path):
    return Config.model_validate({
        "experiment_name": "b2-test",
        "model_kind": "cls",
        "data": {
            "wiki_ss_cache_dir": str(tmp_path / "wiki_ss"),
            "anchors_cache_dir": str(tmp_path / "wiki_ss_anchors"),
        },
        "backbone": BackboneConfig().model_dump(),
        "salad": SaladConfig().model_dump(),
    })


def test_cls_baseline_forward_shape(tmp_path):
    cfg = _default_cfg(tmp_path)
    model = DINOv2CLS(cfg).cuda().eval()
    x = torch.randn(3, 3, 224, 224, device="cuda")
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3, DINOv2CLS.OUTPUT_DIM)
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

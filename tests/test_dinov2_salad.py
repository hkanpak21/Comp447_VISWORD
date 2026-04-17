"""Phase B / B1 tests for the DINOv2 + SALAD wrapper (TESTS.md B1).

All @pytest.mark.gpu — skipped when CUDA is unavailable (login node). Run
them on a T4 via ``srun --pty --partition=t4_ai --qos=comx29 --account=comx29
--gres=gpu:tesla_t4:1 --cpus-per-task=4 --mem=16G --time=00:30:00 bash`` then
``PYTHONPATH=src pytest tests/test_dinov2_salad.py -m gpu -v``.

Note: the first run on any env downloads DINOv2 weights via ``torch.hub``
(~100 MB). Cached under ``~/.cache/torch/hub/`` for later runs.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


from visword.config import (   # noqa: E402
    BackboneConfig,
    Config,
    DataConfig,
    SaladConfig,
)
from visword.models.dinov2_salad import DINOv2SALAD  # noqa: E402


def _default_cfg(tmp_path) -> Config:
    return Config.model_validate({
        "experiment_name": "b1-test",
        "data": {
            "wiki_ss_cache_dir": str(tmp_path / "wiki_ss"),
            "anchors_cache_dir": str(tmp_path / "wiki_ss_anchors"),
        },
        "backbone": BackboneConfig().model_dump(),
        "salad": SaladConfig().model_dump(),
    })


def test_model_forward_shape(tmp_path):
    cfg = _default_cfg(tmp_path)
    model = DINOv2SALAD(cfg).cuda()
    model.eval()

    x = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, model.descriptor_dim), out.shape
    assert out.shape[1] == cfg.salad.num_clusters * cfg.salad.cluster_dim + cfg.salad.token_dim
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_trainable_param_count(tmp_path):
    """Last-4 blocks + head trainable (~35-45% of total at default config)."""
    cfg = _default_cfg(tmp_path)
    model = DINOv2SALAD(cfg).cuda()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frac = trainable / total
    assert 0.30 < frac < 0.60, f"unexpected trainable fraction {frac:.3f}"


def test_model_state_dict_roundtrip(tmp_path):
    cfg = _default_cfg(tmp_path)
    model1 = DINOv2SALAD(cfg).cuda().eval()
    ckpt = tmp_path / "sd.pt"
    torch.save(model1.state_dict(), ckpt)

    model2 = DINOv2SALAD(cfg).cuda().eval()
    model2.load_state_dict(torch.load(ckpt, map_location="cuda"))

    x = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.no_grad():
        y1, y2 = model1(x), model2(x)
    assert torch.equal(y1, y2)

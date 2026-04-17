"""Phase E / E1 — last-block CLS→patch attention heatmaps (TESTS.md E1).

``@pytest.mark.gpu`` + ``@pytest.mark.integration`` because running a
DINOv2 forward pass on a real crop is what we actually want to inspect.
The test runs via the sbatch test runner; on the login node it skips.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_attention_overlay_is_nontrivial_png(debug_run):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from PIL import Image
    from visword.config import Config
    from visword.data.light_dataset import default_transform
    from visword.eval_phase1 import _build_model_from_cfg, _load_checkpoint, _load_cfg
    from visword.interpret.attention import (
        compute_cls_to_patch_attention,
        render_overlay,
    )

    cfg: Config = _load_cfg(debug_run)
    model = _build_model_from_cfg(cfg)
    _load_checkpoint(model, debug_run / "checkpoints" / "last.pt",
                     device=torch.device("cuda"))

    # Build one sample crop from the anchors cache (any image will do).
    anchors_img_dir = Path(cfg.data.anchors_cache_dir) / "images"
    jpg = next(iter(sorted(anchors_img_dir.glob("*.jpg"))), None)
    if jpg is None:
        pytest.skip(f"no jpgs in {anchors_img_dir}")

    target = cfg.cropper.target_size
    with Image.open(jpg) as im:
        im = im.convert("RGB").resize((target, target), Image.BILINEAR)
        transformed = default_transform()(im).unsqueeze(0).cuda()

    captures = compute_cls_to_patch_attention(model.backbone, transformed)
    assert len(captures) == 1
    cap = captures[0]
    assert cap.side > 0
    # Attention weights are probs → sum ≈ 1 per sample.
    assert 0.95 <= cap.weights.sum() <= 1.05, cap.weights.sum()

    out_path = debug_run / "interpret" / "attention_sample0.png"
    render_overlay(im, cap, out_path)
    sidecar = out_path.with_suffix(".json")

    assert out_path.exists() and out_path.stat().st_size > 0
    data = json.loads(sidecar.read_text())
    assert {"side", "num_heads", "mean", "max", "entropy"} <= data.keys()
    assert data["max"] > 0.0

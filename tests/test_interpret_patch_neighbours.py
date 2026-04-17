"""Phase E / E4 — patch-level anchor/pos/neg visualiser (TESTS.md E4).

Asserts each produced PNG is a readable image with non-trivial pixel
variance (rules out an all-white failure case).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _open_resized(path: Path, size: int):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB").resize((size, size), Image.BILINEAR)


def test_patch_neighbours_produces_nontrivial_pngs(debug_run):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    import json
    from PIL import Image
    from visword.config import Config
    from visword.data.light_dataset import default_transform
    from visword.eval_phase1 import _build_model_from_cfg, _load_checkpoint, _load_cfg
    from visword.interpret.patch_neighbours import find_patch_matches, render_matches

    cfg: Config = _load_cfg(debug_run)
    model = _build_model_from_cfg(cfg)
    _load_checkpoint(model, debug_run / "checkpoints" / "last.pt",
                     device=torch.device("cuda"))

    # Pick a (anchor, positive, negative) from the anchors cache's val triplets.
    triplets_path = Path(cfg.data.anchors_cache_dir) / "triplets_val.jsonl"
    if not triplets_path.exists():
        pytest.skip(f"no triplets at {triplets_path}")
    img_root = Path(cfg.data.anchors_cache_dir) / "images"

    triplet = None
    for line in triplets_path.read_text().splitlines():
        row = json.loads(line)
        anc = img_root / row["anchor"]
        pos_candidates = [img_root / p for p in row.get("positives", [])]
        neg_candidates = [img_root / n for n in row.get("negatives", [])]
        pos_existing = [p for p in pos_candidates if p.exists()]
        neg_existing = [n for n in neg_candidates if n.exists()]
        if anc.exists() and pos_existing and neg_existing:
            triplet = (anc, pos_existing[0], neg_existing[0])
            break

    if triplet is None:
        pytest.skip("no triplet with all three files present on disk")

    size = cfg.cropper.target_size
    pil_anc, pil_pos, pil_neg = (_open_resized(p, size) for p in triplet)
    transform = default_transform()
    t_anc = transform(pil_anc).cuda()
    t_pos = transform(pil_pos).cuda()
    t_neg = transform(pil_neg).cuda()

    matches = find_patch_matches(
        model.backbone, t_anc, t_pos, t_neg, k_examples=2,
    )
    assert len(matches) == 2

    # side = target_size / 14 (DINOv2 patch size)
    side = size // 14
    out_dir = debug_run / "interpret"
    outs = render_matches(pil_anc, pil_pos, pil_neg, matches,
                         side=side, image_size=size, out_dir=out_dir)
    assert len(outs) == 2
    for p in outs:
        assert p.exists() and p.stat().st_size > 0
        arr = np.array(Image.open(p))
        # Not an all-one-colour image — pixel stddev above a trivial floor.
        assert arr.std() > 5, f"{p} has trivial variance ({arr.std():.2f})"

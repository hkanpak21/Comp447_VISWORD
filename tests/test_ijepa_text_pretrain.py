import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    import pytest
    has_pytest = True
except ImportError:
    has_pytest = False

# Pytest markers if pytest is available
if has_pytest:
    pytestmark = [pytest.mark.integration, pytest.mark.gpu]

from visword.config import Config
from visword.data.cropper import NonOverlappingCropper
from visword.data.light_dataset import LightWikiScreenshotDataset
from visword.models.ijepa_text_predictor import VisionTransformerTextPredictor


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_dataset_text_returns():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    data_dir = os.environ.get("DATA_DIR") or str(PROJECT_ROOT / "data")
    cache_dir = Path(data_dir) / "wiki_ss"
    
    if not (cache_dir / "manifest.json").exists():
        if has_pytest:
            pytest.skip(f"no cache at {cache_dir}")
        else:
            print(f"Skipping test_dataset_text_returns: no cache at {cache_dir}")
            return

    cropper = NonOverlappingCropper(crop_size=490, target_size=224)
    
    # Test title source
    ds_title = LightWikiScreenshotDataset(
        cache_dir,
        indices=[0, 1],
        cropper=cropper,
        k_per_page=1,
        return_text=True,
        text_source="title"
    )
    
    item = ds_title[0]
    assert len(item) == 3
    tensors, text, idx = item
    assert isinstance(tensors, torch.Tensor)
    assert isinstance(text, str)
    assert len(text) > 0
    assert text == ds_title.rows[0]["title"]

    # Test full text source
    ds_text = LightWikiScreenshotDataset(
        cache_dir,
        indices=[0, 1],
        cropper=cropper,
        k_per_page=1,
        return_text=True,
        text_source="text"
    )
    
    tensors, text_content, idx = ds_text[0]
    assert isinstance(text_content, str)
    assert len(text_content) > 0


def test_ijepa_text_predictor():
    predictor = VisionTransformerTextPredictor(
        num_patches=256,
        max_text_tokens=64,
        embed_dim=1280,
        predictor_embed_dim=384,
        depth=2,
        num_heads=4,
        target_dim=768
    )
    
    B = 2
    min_keep_enc = 100
    nenc = 1
    T = 32
    
    # x context representations: (B * nenc, min_keep_enc, D)
    x = torch.zeros(B * nenc, min_keep_enc, 1280)
    # masks_x (context masks): list of nenc tensors of shape (B, min_keep_enc)
    masks_x = [torch.randint(0, 256, (B, min_keep_enc)) for _ in range(nenc)]
    
    out = predictor(x, masks_x, target_len=T)
    # Expected output: (B * nenc, T, target_dim)
    assert out.shape == (B * nenc, T, 768)


# ---------------------------------------------------------------------------
# Integration Tests (Requires GPU and data cache)
# ---------------------------------------------------------------------------

def test_ijepa_text_pretrain_integration(tmp_path):
    if not torch.cuda.is_available():
        if has_pytest:
            pytest.skip("CUDA required for integration test")
        else:
            print("Skipping integration test: CUDA not available")
            return
        
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    data_dir = os.environ.get("DATA_DIR") or str(PROJECT_ROOT / "data")
    cache_manifest = Path(data_dir) / "wiki_ss" / "manifest.json"
    if not cache_manifest.exists():
        if has_pytest:
            pytest.skip(f"no cache at {cache_manifest}")
        else:
            print(f"Skipping integration test: no cache at {cache_manifest}")
            return

    runs_root = Path(tmp_path) / "runs"
    runs_root.mkdir(exist_ok=True, parents=True)
    
    # Run train_ijepa_text.py for 1 step
    env = {**os.environ, "DATA_DIR": data_dir, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    cmd = [
        sys.executable, "-m", "visword.train_ijepa_text",
        "--config", "configs/ijepa_text_target.yaml",
        "--runs-root", str(runs_root),
        "--run-name", "test-text-pretrain",
        "--set", "data.num_train_samples=8",
        "--set", "data.num_eval_samples=4",
        "--set", "train.batch_size=2",
        "--set", "train.epochs=1",
        "--set", "train.eval_every_steps=2",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"train_ijepa_text failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    
    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    
    # Check that required files exist
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "checkpoints" / "best_phase1.pt").exists()
    
    # Read metrics.jsonl
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    assert len(metrics) >= 1
    
    # Validate fields in training step log
    train_logs = [m for m in metrics if "loss" in m]
    assert train_logs
    for key in ("step", "epoch", "loss", "lr_bb", "lr_pred", "gpu_mem_gb", "wall_time_s"):
        assert key in train_logs[0]
        
    # Validate eval log
    eval_logs = [m for m in metrics if "eval_step" in m]
    assert eval_logs
    assert "phase1_recall@10" in eval_logs[0]


if __name__ == "__main__":
    print("=== RUNNING I-JEPA TEXT-TARGET PRETRAIN UNIT TESTS ===")
    test_dataset_text_returns()
    print("  test_dataset_text_returns passed.")
    test_ijepa_text_predictor()
    print("  test_ijepa_text_predictor passed.")
    print("All unit tests passed successfully!")
    
    if torch.cuda.is_available():
        print("\n=== RUNNING I-JEPA TEXT-TARGET PRETRAIN INTEGRATION TEST ===")
        import tempfile
        import shutil
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            test_ijepa_text_pretrain_integration(tmp_dir)
            print("Integration test passed successfully!")
        finally:
            shutil.rmtree(tmp_dir)
    else:
        print("\nCUDA not available; skipping integration test.")

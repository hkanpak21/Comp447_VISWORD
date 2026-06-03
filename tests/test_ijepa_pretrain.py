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
from visword.models.ijepa_masks import MaskCollator, JepaMaskCollator
from visword.models.ijepa_predictor import VisionTransformerPredictor


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_mask_collator():
    # 224x224 input with patch_size 14 -> 16x16 grid (256 patches)
    collator = MaskCollator(
        input_size=(224, 224),
        patch_size=14,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False
    )
    
    # Batch of 2 dummy images
    batch = [torch.zeros(3, 224, 224), torch.zeros(3, 224, 224)]
    collated, masks_enc, masks_pred = collator(batch)
    
    assert collated.shape == (2, 3, 224, 224)
    # masks_enc is a list of length nenc containing tensors of shape (B, min_keep_enc)
    # masks_pred is a list of length npred containing tensors of shape (B, min_keep_pred)
    assert isinstance(masks_enc, list)
    assert len(masks_enc) == 1
    assert masks_enc[0].ndim == 2
    assert masks_enc[0].shape[0] == 2
    
    assert isinstance(masks_pred, list)
    assert len(masks_pred) == 4
    assert masks_pred[0].ndim == 2
    assert masks_pred[0].shape[0] == 2
    
    # Verify no overlap between context and target if allow_overlap=False
    for b in range(2):
        enc_indices = set(masks_enc[0][b].tolist())
        for p in range(4):
            pred_indices = set(masks_pred[p][b].tolist())
            overlap = enc_indices.intersection(pred_indices)
            assert not overlap, f"Overlap detected between context mask and target mask {p} for batch {b}"


def test_jepa_mask_collator_wrapper():
    collator = MaskCollator(
        input_size=(224, 224),
        patch_size=14,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        nenc=1,
        npred=2,
        min_keep=10
    )
    jepa_collator = JepaMaskCollator(collator)
    
    # Mock data loader item format: (crop_tensor: (1, 3, H, W), page_idx)
    batch = [
        (torch.zeros(1, 3, 224, 224), 0),
        (torch.zeros(1, 3, 224, 224), 1)
    ]
    collated, masks_enc, masks_pred = jepa_collator(batch)
    
    assert collated.shape == (2, 3, 224, 224)
    assert isinstance(masks_enc, list)
    assert masks_enc[0].shape[0] == 2
    assert isinstance(masks_pred, list)
    assert masks_pred[0].shape[0] == 2


def test_ijepa_predictor():
    predictor = VisionTransformerPredictor(
        num_patches=256,
        embed_dim=1280,
        predictor_embed_dim=384,
        depth=2,
        num_heads=4
    )
    
    B = 2
    min_keep_enc = 100
    min_keep_pred = 40
    nenc = 1
    npred = 2
    
    # x context representations: (B * nenc, min_keep_enc, D)
    x = torch.zeros(B * nenc, min_keep_enc, 1280)
    # masks_x (context masks): list of nenc tensors of shape (B, min_keep_enc)
    masks_x = [torch.randint(0, 256, (B, min_keep_enc)) for _ in range(nenc)]
    # masks (target masks): list of npred tensors of shape (B, min_keep_pred)
    masks = [torch.randint(0, 256, (B, min_keep_pred)) for _ in range(npred)]
    
    out = predictor(x, masks_x, masks)
    # Expected output: (B * npred * nenc, min_keep_pred, 1280)
    assert out.shape == (B * npred * nenc, min_keep_pred, 1280)


# ---------------------------------------------------------------------------
# Integration Tests (Requires GPU and data cache)
# ---------------------------------------------------------------------------

def test_ijepa_pretrain_integration(tmp_path):
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
    
    # Run train_ijepa.py for 1 step
    env = {**os.environ, "DATA_DIR": data_dir, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    cmd = [
        sys.executable, "-m", "visword.train_ijepa",
        "--config", "configs/ijepa_pretrain_2blocks.yaml",
        "--runs-root", str(runs_root),
        "--run-name", "test-pretrain",
        "--set", "data.num_train_samples=8",
        "--set", "data.num_eval_samples=4",
        "--set", "train.batch_size=2",
        "--set", "train.epochs=1",
        "--set", "train.eval_every_steps=2",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"train_ijepa failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    
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
    for key in ("step", "epoch", "loss", "lr_bb", "lr_pred", "ema_decay", "gpu_mem_gb", "wall_time_s"):
        assert key in train_logs[0]
        
    # Validate eval log
    eval_logs = [m for m in metrics if "eval_step" in m]
    assert eval_logs
    assert "phase1_recall@10" in eval_logs[0]

    # Test that eval_phase1 script can load and run on the resulting checkpoint
    eval_cmd = [
        sys.executable, "-m", "visword.eval_phase1",
        "--run-dir", str(run_dir),
        "--checkpoint", "last.pt",
    ]
    eval_res = subprocess.run(eval_cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert eval_res.returncode == 0, f"eval_phase1 failed on I-JEPA pretrain checkpoint:\nstdout:\n{eval_res.stdout}\nstderr:\n{eval_res.stderr}"
    assert (run_dir / "phase1_recall.json").exists()
    
    # Read phase1_recall.json and assert fields
    recall_data = json.loads((run_dir / "phase1_recall.json").read_text())
    assert "recall" in recall_data
    assert "sanity" in recall_data


# ---------------------------------------------------------------------------
# Direct Script Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== RUNNING I-JEPA PRETRAIN UNIT TESTS ===")
    test_mask_collator()
    print("  test_mask_collator passed.")
    test_jepa_mask_collator_wrapper()
    print("  test_jepa_mask_collator_wrapper passed.")
    test_ijepa_predictor()
    print("  test_ijepa_predictor passed.")
    print("All unit tests passed successfully!")
    
    if torch.cuda.is_available():
        print("\n=== RUNNING I-JEPA PRETRAIN INTEGRATION TEST ===")
        import tempfile
        import shutil
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            test_ijepa_pretrain_integration(tmp_dir)
            print("Integration test passed successfully!")
        finally:
            shutil.rmtree(tmp_dir)
    else:
        print("\nCUDA not available; skipping integration test.")

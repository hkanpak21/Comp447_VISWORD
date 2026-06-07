#!/usr/bin/env bash
#SBATCH --job-name=vw-eval-custom
#SBATCH --output=runs/_slurm/eval_custom_%j.out
#SBATCH --error=runs/_slurm/eval_custom_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
source scripts/_env.sh

python "$PROJECT_ROOT/scripts/ensure_dinov2_hub.py" || true

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

MLP_RUN="runs/2026-06-04_131101_a1627a72_visword-ijepa-text-target-mlp-30_b195"
SALAD_RUN="runs/2026-06-05_102116_a1627a72_visword-ijepa-text-target-salad-_9f04"

echo "========================================================================"
echo "EVALUATING MLP MODEL"
echo "========================================================================"
echo "--- 1. Protocol A Standard ---"
python -u -m visword.eval_phase1_holdout --run-dir "$MLP_RUN"

echo "--- 2. Protocol A Blank Top 15% ---"
python -u -m visword.eval_phase1_holdout --run-dir "$MLP_RUN" --blank-page-top-frac 0.15

echo "========================================================================"
echo "EVALUATING SALAD MODEL"
echo "========================================================================"
echo "--- 1. Phase 1 (Crop-vs-Crop) ---"
python -u -m visword.eval_phase1 --run-dir "$SALAD_RUN"

echo "--- 2. Phase 2 (Anchor-Pool) ---"
python -u -m visword.eval_phase2 --run-dir "$SALAD_RUN"

echo "--- 3. Protocol A Standard ---"
python -u -m visword.eval_phase1_holdout --run-dir "$SALAD_RUN"

echo "--- 4. Protocol A Blank Top 15% ---"
python -u -m visword.eval_phase1_holdout --run-dir "$SALAD_RUN" --blank-page-top-frac 0.15

echo "--- 5. Phase 2 Title Blanked 15% ---"
python -u -m visword.eval_phase2_titleblanked --run-dir "$SALAD_RUN" --blank-top-frac 0.15

echo "--- 6. Phase 2 Title Blanked 30% ---"
python -u -m visword.eval_phase2_titleblanked --run-dir "$SALAD_RUN" --blank-top-frac 0.30 || true

echo "--- 7. DINOv2 family / Interpret ---"
python -u -m visword.interpret --run-dir "$SALAD_RUN" --k 4 || true

echo "========================================================================"
echo "EVALUATIONS COMPLETED"
echo "========================================================================"

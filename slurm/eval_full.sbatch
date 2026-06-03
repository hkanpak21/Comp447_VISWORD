#!/usr/bin/env bash
#SBATCH --job-name=vw-eval-full
#SBATCH --output=runs/_slurm/evalfull_%j.out
#SBATCH --error=runs/_slurm/evalfull_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00

# Full eval suite for one trained-checkpoint run dir:
#   * Phase-1 legacy (crop-vs-crop)
#   * Phase-1 Protocol A (leak-free crop->page)
#   * Phase-2 (anchor pool)
#   * Phase-2 title-blanked (CLIP-OCR hypothesis test)
#   * Interpret artifacts (DINOv2 family only)
#
# Usage: RUN_DIR=/path/to/run sbatch slurm/eval_full.sbatch
# Or: RUN_DIR=latest (defaults to most recent run dir)

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
    source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    python "${SLURM_SUBMIT_DIR}/scripts/ensure_dinov2_hub.py" || true
else
    python "$(dirname "${BASH_SOURCE[0]}")/../scripts/ensure_dinov2_hub.py" || true
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

RUN_DIR_IN="${RUN_DIR:-latest}"
if [ "$RUN_DIR_IN" = "latest" ]; then
    RUN_DIR="$(ls -dt "$PROJECT_ROOT"/runs/*/ 2>/dev/null \
               | grep -v '/_slurm/' | grep -v '/_dedup_logs/' | head -1 | sed 's:/*$::')"
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: RUN_DIR=latest but no run directory found." >&2
        exit 2
    fi
else
    RUN_DIR="$RUN_DIR_IN"
fi
echo "--- RUN_DIR=$RUN_DIR ---"

CHECKPOINT="${CHECKPOINT:-best_phase1.pt}"

# Legacy phase-1 + phase-2 (existing artifacts compatibility).
python -u -m visword.eval_phase1 --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" || true
python -u -m visword.eval_phase2 --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" || true

# New: leak-free Protocol-A crop->page.
python -u -m visword.eval_phase1_holdout --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT"

# New: title-blanked Phase-2 (CLIP-OCR hypothesis test). Two blank fractions.
python -u -m visword.eval_phase2_titleblanked --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" --blank-top-frac 0.15
python -u -m visword.eval_phase2_titleblanked --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" --blank-top-frac 0.30 || true

# Interpret (DINOv2 family only; CLIP variants are gated inside the module).
python -u -m visword.interpret --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" --k 4 || true

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    python "${SLURM_SUBMIT_DIR}/scripts/summarise_run.py" "$RUN_DIR" || true
else
    python "$(dirname "${BASH_SOURCE[0]}")/../scripts/summarise_run.py" "$RUN_DIR" || true
fi

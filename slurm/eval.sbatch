#!/usr/bin/env bash
#SBATCH --job-name=vw-eval
#SBATCH --output=runs/_slurm/eval_%j.out
#SBATCH --error=runs/_slurm/eval_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source scripts/_env.sh

# Safety: the DINOv2 hub cache needs to be pinned at a Py3.9-compatible
# commit (e1277af), otherwise model-building crashes with "float | None"
# PEP 604 errors on our env. Re-runs idempotent.
python "$PROJECT_ROOT/scripts/ensure_dinov2_hub.py" || true

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

RUN_DIR_IN="${RUN_DIR:-latest}"
if [ "$RUN_DIR_IN" = "latest" ]; then
    RUN_DIR="$(ls -dt "$PROJECT_ROOT"/runs/*/ 2>/dev/null \
               | grep -v '/_slurm/' | head -1 | sed 's:/*$::')"
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: RUN_DIR=latest but no run directory found under runs/" >&2
        exit 2
    fi
else
    RUN_DIR="$RUN_DIR_IN"
fi
echo "--- RUN_DIR=$RUN_DIR ---"

CHECKPOINT="${CHECKPOINT:-best_phase1.pt}"

python -u -m visword.eval_phase1 --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT"
python -u -m visword.eval_phase2 --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT"
python -u -m visword.interpret   --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" --k 4

# Print a human summary at the end so `sacct -o stdout` is useful.
python "$PROJECT_ROOT/scripts/summarise_run.py" "$RUN_DIR"

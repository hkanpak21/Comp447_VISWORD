#!/usr/bin/env bash
# Generic eval-suite sbatch (for any cluster user).
#
# Required env-vars:
#   RUN_DIR    — path to a completed run dir (under YOUR runs root)
#   REPO_ROOT  — path to the visword repo on YOUR scratch
#
# Optional:
#   CONDA_ENV  — defaults to the shared visword env
#
# Submit example:
#   sbatch \
#     --partition=t4_ai \
#     --time=02:00:00 \
#     --gres=gpu:tesla_t4:1 \
#     --mem=24G \
#     --cpus-per-task=4 \
#     --output=/scratch/alice/visword/_slurm/eval_%j.out \
#     --error=/scratch/alice/visword/_slurm/eval_%j.err \
#     --export=ALL,RUN_DIR=/scratch/alice/visword/runs/<run-name>,REPO_ROOT=/scratch/alice/visword \
#     /scratch/alice/visword/slurm/eval_full_user.sbatch

set -euo pipefail

: "${RUN_DIR:?RUN_DIR env var required}"
: "${REPO_ROOT:?REPO_ROOT env var required}"

CONDA_ENV="${CONDA_ENV:-/scratch/hkanpak21/conda_envs/visword}"

cd "$REPO_ROOT"

if [ -f "$CONDA_ENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_ENV/bin/activate"
fi

export PYTHONPATH=src

# Run all four evals. Failures of one don't prevent the others.
echo "=== eval_phase1 (legacy crop-vs-crop) ==="
python -u -m visword.eval_phase1 --run-dir "$RUN_DIR" || true

echo "=== eval_phase1_holdout (Protocol-A: leak-free crop->page) ==="
python -u -m visword.eval_phase1_holdout --run-dir "$RUN_DIR" || true

echo "=== eval_phase2 (anchor pool) ==="
python -u -m visword.eval_phase2 --run-dir "$RUN_DIR" || true

echo "=== eval_phase2_titleblanked-15 (CLIP OCR test) ==="
python -u -m visword.eval_phase2_titleblanked --run-dir "$RUN_DIR" --blank-top-frac 0.15 || true

echo "=== outputs in $RUN_DIR ==="
ls -la "$RUN_DIR"/*.json 2>/dev/null | head -10

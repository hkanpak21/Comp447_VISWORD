#!/usr/bin/env bash
#SBATCH --job-name=vw-test-encoders
#SBATCH --output=runs/_slurm/test_encoders_%j.out
#SBATCH --error=runs/_slurm/test_encoders_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#
# Ticket 02: per-encoder model-wrapper integration tests (size / unit-length /
# determinism) for all 6 grid encoders + MAE. Submit from the dev worktree so
# SLURM_SUBMIT_DIR (== that worktree) supplies the right src + third_party/salad.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/${USER}/VISWORD_v1}"
# shellcheck source=scripts/_env.sh
source scripts/_env.sh

python scripts/ensure_dinov2_hub.py || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

PYTHONPATH=src python -m pytest tests/test_encoder_wrapper.py -m integration -v

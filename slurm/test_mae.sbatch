#!/usr/bin/env bash
#SBATCH --job-name=vw-test-mae
#SBATCH --output=runs/_slurm/test_mae_%j.out
#SBATCH --error=runs/_slurm/test_mae_%j.err
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
# Ticket 04 smoke: MAE reader contract + a one-step body-target training run finishes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/${USER}/VISWORD_v1}"
# shellcheck source=scripts/_env.sh
source scripts/_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# The worktree has no data/; point the integration test at the real cache so it RUNS
# (not skips) the one-step training path.
export DATA_DIR=/scratch/hkanpak21/VISWORD/data
PYTHONPATH=src python -m pytest tests/test_mae_text_pretrain.py -m integration -v

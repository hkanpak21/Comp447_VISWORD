#!/usr/bin/env bash
#SBATCH --job-name=vw-perfect-text
#SBATCH --output=runs/_slurm/perfect_text_%j.out
#SBATCH --error=runs/_slurm/perfect_text_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#
# Ticket 05: perfect-text upper bound (ground-truth text -> frozen BERT -> page-level
# retrieval) on the seed-42 eval slice. Submit from the dev worktree.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/${USER}/VISWORD_v1}"
# shellcheck source=scripts/_env.sh
source scripts/_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHONPATH=src python scripts/perfect_text_bound.py \
    --cache-dir /scratch/hkanpak21/VISWORD/data/wiki_ss \
    --out "${PT_OUT:-runs/perfect_text_v1}" \
    --num-eval 2000 --chunks 4 --max-text-tokens 256

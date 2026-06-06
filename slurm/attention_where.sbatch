#!/usr/bin/env bash
#SBATCH --job-name=vw-attention
#SBATCH --output=runs/_slurm/attention_%j.out
#SBATCH --error=runs/_slurm/attention_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#
# Ticket 06: attention-on-text for frozen MAE vs the trained reader. Submit from worktree.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/${USER}/VISWORD_v1}"
# shellcheck source=scripts/_env.sh
source scripts/_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHONPATH=src python scripts/attention_where.py \
    --cache-dir /scratch/hkanpak21/VISWORD/data/wiki_ss \
    --out "${ATTN_OUT:-runs/attention_v1}" \
    --reader-ckpt runs/mae_reader_v1/checkpoints/best.pt \
    --pages 8 --max-crops 8

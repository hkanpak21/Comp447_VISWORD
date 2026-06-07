#!/usr/bin/env bash
#SBATCH --job-name=vw-title-erasure
#SBATCH --output=runs/_slurm/title_erasure_%j.out
#SBATCH --error=runs/_slurm/title_erasure_%j.err
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
# Ticket 07: title-erasure check on a reader checkpoint (normal vs top-blanked recall).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/${USER}/VISWORD_v1}"
# shellcheck source=scripts/_env.sh
source scripts/_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHONPATH=src python scripts/eval_reader_blanked.py \
    --cache-dir /scratch/hkanpak21/VISWORD/data/wiki_ss \
    --ckpt "${CKPT:?set CKPT}" \
    --out "${TE_OUT:?set TE_OUT}" \
    --num-eval 2000 --eval-pages "${EVAL_PAGES:-2000}" --blank-frac 0.25

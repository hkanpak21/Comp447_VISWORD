#!/usr/bin/env bash
#SBATCH --job-name=eval-ijepa-text-target
#SBATCH --output=runs/_slurm/eval_ijepa_text_target_%j.out
#SBATCH --error=runs/_slurm/eval_ijepa_text_target_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/${USER}/VISWORD}"
# shellcheck source=scripts/_env.sh
source scripts/_env.sh

python scripts/ensure_dinov2_hub.py || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

PYTHONPATH=src python scripts/rebaseline_grid.py \
    --cache-dir /scratch/bbakay22/VISWORD/data/wiki_ss \
    --out runs/rebaseline_ijepa_text_target \
    --encoders ijepa_text_target \
    --num-eval 2000 \
    --num-pages 2000 \
    --batch-size 64

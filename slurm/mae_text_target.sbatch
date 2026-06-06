#!/usr/bin/env bash
#SBATCH --job-name=vw-mae-reader
#SBATCH --output=runs/_slurm/mae_reader_%j.out
#SBATCH --error=runs/_slurm/mae_reader_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#
# Ticket 04: train the MAE body-target reader. Resumable (fixed --out auto-resumes from
# checkpoints/last.pt), so re-submitting continues across the QOS=1 / <=8h-wall regime.
# Submit from the dev worktree (SLURM_SUBMIT_DIR -> right src + third_party/salad).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/${USER}/VISWORD_v1}"
# shellcheck source=scripts/_env.sh
source scripts/_env.sh

python scripts/ensure_dinov2_hub.py || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

OUT="${MAE_OUT:-runs/mae_reader_v1}"
PYTHONPATH=src python -m visword.train_mae_text \
    --cache-dir /scratch/hkanpak21/VISWORD/data/wiki_ss \
    --out "$OUT" \
    --num-train "${NUM_TRAIN:-20000}" \
    --num-eval 2000 \
    --eval-pages "${EVAL_PAGES:-500}" \
    --max-crops-per-page "${MAX_CROPS:-8}" \
    --epochs "${EPOCHS:-3}" \
    --batch-pages "${BATCH_PAGES:-8}" \
    --num-trainable-blocks "${TRAINABLE_BLOCKS:-4}" \
    --max-text-tokens "${MAX_TEXT_TOKENS:-256}" \
    --eval-every-steps "${EVAL_EVERY:-200}"

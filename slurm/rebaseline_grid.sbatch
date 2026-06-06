#!/usr/bin/env bash
#SBATCH --job-name=vw-rebaseline-grid
#SBATCH --output=runs/_slurm/rebaseline_%j.out
#SBATCH --error=runs/_slurm/rebaseline_%j.err
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
# Ticket 02 re-baseline grid at legible (native-224) resolution. Submit from the dev
# worktree (SLURM_SUBMIT_DIR == worktree -> right src + third_party/salad symlink).
# Resumable: a fixed --out dir keeps per-encoder results, so a re-submit continues.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/${USER}/VISWORD_v1}"
# shellcheck source=scripts/_env.sh
source scripts/_env.sh

python scripts/ensure_dinov2_hub.py || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

OUT="${GRID_OUT:-runs/rebaseline_grid_v1}"
PYTHONPATH=src python scripts/rebaseline_grid.py \
    --cache-dir /scratch/hkanpak21/VISWORD/data/wiki_ss \
    --out "$OUT" \
    --num-eval 2000 \
    --num-pages "${NUM_PAGES:-2000}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --blank-top-frac "${BLANK_TOP_FRAC:-0.0}"

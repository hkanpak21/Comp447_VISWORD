#!/usr/bin/env bash
# Single A40 (48GB), 48-hour fine-tune for the image-only I-JEPA + SALAD head at 490×490.
# Fills the last TBD cell in the I-JEPA heads comparison table.
#
# Root cause of the previous crash (step 69, job 002521): the earlier job used a
# 4-hour time limit (slurm/train_a40_3h.sbatch). batch_size=8 at 490px uses only
# ~3.4 GB GPU RAM on T4 16GB — no OOM. 48h is sufficient (text-target SALAD took 23.8h).
#
# Submit from the repo root on Valar:
#   sbatch slurm/ijepa_imageonly_salad_4xt4.sbatch
#
#SBATCH --job-name=vw-io-salad
#SBATCH --output=runs/_slurm/io_salad_%j.out
#SBATCH --error=runs/_slurm/io_salad_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:ampere_a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

# Establish the repo root from the submission directory (SLURM sets SLURM_SUBMIT_DIR)
cd "${SLURM_SUBMIT_DIR}"
source scripts/_env.sh

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

CONFIG="${CONFIG:-configs/ijepa_imageonly_salad_30k_490_t4.yaml}"
RUN_NAME="${RUN_NAME:-}"

EXTRA_ARGS=()
if [ -n "$RUN_NAME" ]; then
    EXTRA_ARGS+=(--run-name "$RUN_NAME")
fi

mkdir -p runs/_slurm

cd "$PROJECT_ROOT"
PYTHONPATH=src python -u -m visword.train --config "$CONFIG" "${EXTRA_ARGS[@]}"

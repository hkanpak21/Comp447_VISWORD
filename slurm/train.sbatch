#!/usr/bin/env bash
#SBATCH --job-name=vw-train
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/train_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/train_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:55:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

# Pin the DINOv2 hub cache to the Py3.9-compatible commit (idempotent).
python /scratch/hkanpak21/VISWORD/scripts/ensure_dinov2_hub.py || true

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

CONFIG="${CONFIG:-configs/salad_main.yaml}"
RUN_NAME="${RUN_NAME:-}"

EXTRA_ARGS=()
if [ -n "$RUN_NAME" ]; then
    EXTRA_ARGS+=(--run-name "$RUN_NAME")
fi

cd /scratch/hkanpak21/VISWORD
PYTHONPATH=src python -u -m visword.train --config "$CONFIG" "${EXTRA_ARGS[@]}" "$@"

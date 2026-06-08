#!/usr/bin/env bash
#SBATCH --job-name=vw-train-a40
#SBATCH --output=runs/_slurm/train_%j.out
#SBATCH --error=runs/_slurm/train_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:ampere_a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source scripts/_env.sh

# Pin the DINOv2 hub cache to the Py3.9-compatible commit (idempotent).
python scripts/ensure_dinov2_hub.py || true

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

CONFIG="${CONFIG:-configs/salad_main.yaml}"
RUN_NAME="${RUN_NAME:-}"

EXTRA_ARGS=()
if [ -n "$RUN_NAME" ]; then
    EXTRA_ARGS+=(--run-name "$RUN_NAME")
fi

cd "$PROJECT_ROOT"
PYTHONPATH=src python -u -m visword.train --config "$CONFIG" "${EXTRA_ARGS[@]}" "$@"

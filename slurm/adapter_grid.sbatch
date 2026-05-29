#!/usr/bin/env bash
#SBATCH --job-name=vw-adapter
#SBATCH --output=runs/_slurm/adapter_%j.out
#SBATCH --error=runs/_slurm/adapter_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:ampere_a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:55:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

# Pin the DINOv2 hub cache to the Py3.9-compatible commit (idempotent).
python "$(dirname "${BASH_SOURCE[0]}")/../scripts/ensure_dinov2_hub.py" || true

# I-JEPA and CLIP need HF access for model weights on first run.
# After that, they are cached under ~/.cache/huggingface/hub.
# Uncomment the next line if running on a node WITHOUT internet:
# export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

CONFIG="${CONFIG:-configs/adapter_dinov2_linear.yaml}"
RUN_NAME="${RUN_NAME:-}"

EXTRA_ARGS=()
if [ -n "$RUN_NAME" ]; then
    EXTRA_ARGS+=(--run-name "$RUN_NAME")
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHONPATH=src python -u -m visword.train --config "$CONFIG" "${EXTRA_ARGS[@]}" "$@"

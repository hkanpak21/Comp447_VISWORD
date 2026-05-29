#!/usr/bin/env bash
#SBATCH --job-name=vw-text
#SBATCH --output=runs/_slurm/eval_text_%j.out
#SBATCH --error=runs/_slurm/eval_text_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
python "$(dirname "${BASH_SOURCE[0]}")/../scripts/ensure_dinov2_hub.py" || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0

: "${CONFIG:?need CONFIG}"
: "${RUN_NAME:?need RUN_NAME}"
: "${ENCODER:?need ENCODER}"

PYTHONPATH=src python -u -m visword.eval_text \
    --config "$CONFIG" --run-name "$RUN_NAME" --encoder "$ENCODER"

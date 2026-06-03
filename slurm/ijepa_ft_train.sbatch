#!/usr/bin/env bash
#SBATCH --job-name=vw-ijepa-ft
#SBATCH --output=runs/_slurm/ijepa_ft_%j.out
#SBATCH --error=runs/_slurm/ijepa_ft_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

CONFIG="${CONFIG:-configs/ijepa_ft_2blocks.yaml}"
RUN_NAME="${RUN_NAME:-}"

EXTRA_ARGS=()
if [ -n "$RUN_NAME" ]; then
    EXTRA_ARGS+=(--run-name "$RUN_NAME")
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHONPATH=src python -u -m visword.train --config "$CONFIG" "${EXTRA_ARGS[@]}" "$@"

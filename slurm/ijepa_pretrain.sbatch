#!/usr/bin/env bash
#SBATCH --job-name=vw-ijepa-pretrain
#SBATCH --output=runs/_slurm/ijepa_pretrain_%j.out
#SBATCH --error=runs/_slurm/ijepa_pretrain_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
    source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 SKIP_FINGERPRINT_VERIFICATION=1

CONFIG="${CONFIG:-configs/ijepa_pretrain_all_blocks_full_res.yaml}"
RUN_NAME="${RUN_NAME:-}"

EXTRA_ARGS=()
if [ -n "$RUN_NAME" ]; then
    EXTRA_ARGS+=(--run-name "$RUN_NAME")
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi
MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
PYTHONPATH=src torchrun --nproc_per_node=4 --master_port=$MASTER_PORT -m visword.train_ijepa --config "$CONFIG" "${EXTRA_ARGS[@]}" "$@"

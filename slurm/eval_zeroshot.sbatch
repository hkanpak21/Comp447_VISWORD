#!/usr/bin/env bash
#SBATCH --job-name=vw-zs
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/eval_zeroshot_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/eval_zeroshot_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

# Pin the DINOv2 hub cache (Py3.9-compatible commit).
python /scratch/hkanpak21/VISWORD/scripts/ensure_dinov2_hub.py || true

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

CONFIG="${CONFIG:?need CONFIG=configs/...}"
RUN_NAME="${RUN_NAME:?need RUN_NAME}"

cd /scratch/hkanpak21/VISWORD
PYTHONPATH=src python -u -m visword.eval_zeroshot --config "$CONFIG" --run-name "$RUN_NAME"

#!/usr/bin/env bash
#SBATCH --job-name=vw-attn-corr
#SBATCH --output=runs/_slurm/attncorr_%j.out
#SBATCH --error=runs/_slurm/attncorr_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
python "$(dirname "${BASH_SOURCE[0]}")/../scripts/ensure_dinov2_hub.py" || true

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs/_attn_corr

PYTHONPATH=src python -u -m scripts.attention_text_correlation \
    --num-pages 200 --seed 42 \
    --encoders clip siglip dinov2 ijepa \
    --out runs/_attn_corr/n200.json

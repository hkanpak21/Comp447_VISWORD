#!/usr/bin/env bash
#SBATCH --job-name=vw-clip-sweep
#SBATCH --output=runs/_slurm/clip_sweep_%j.out
#SBATCH --error=runs/_slurm/clip_sweep_%j.err
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
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHONPATH=src python -u scripts/backbone_text_adapter_sweep.py \
    --backbone clip \
    --sweep \
    --n-train 4000 \
    --n-eval 1000 \
    --steps 3000 \
    --out-dir "$PROJECT_ROOT"/runs/backbone_text_adapter_sweep

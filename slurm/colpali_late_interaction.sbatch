#!/usr/bin/env bash
#SBATCH --job-name=vw-colpali
#SBATCH --output=runs/_slurm/colpali_%j.out
#SBATCH --error=runs/_slurm/colpali_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs/_colpali

PYTHONPATH=src python -u -m scripts.colpali_late_interaction \
    --num-pages 500 \
    --seed 42 \
    --out runs/_colpali/clip_lateint_n500.json

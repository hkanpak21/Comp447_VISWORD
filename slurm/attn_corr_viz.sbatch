#!/usr/bin/env bash
#SBATCH --job-name=vw-attn-viz
#SBATCH --output=runs/_slurm/attnviz_%j.out
#SBATCH --error=runs/_slurm/attnviz_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
python "$(dirname "${BASH_SOURCE[0]}")/../scripts/ensure_dinov2_hub.py" || true

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHONPATH=src python -u -m scripts.attn_corr_viz --n 3

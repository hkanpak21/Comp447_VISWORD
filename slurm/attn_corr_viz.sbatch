#!/usr/bin/env bash
#SBATCH --job-name=vw-attn-viz
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/attnviz_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/attnviz_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -uo pipefail
source /scratch/hkanpak21/VISWORD/scripts/_env.sh
python /scratch/hkanpak21/VISWORD/scripts/ensure_dinov2_hub.py || true

cd /scratch/hkanpak21/VISWORD
PYTHONPATH=src python -u -m scripts.attn_corr_viz --n 3

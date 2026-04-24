#!/usr/bin/env bash
#SBATCH --job-name=vw-analysis
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/analysis_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/analysis_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

# CPU-only: no --gres=gpu and force CUDA hidden so torch.hub doesn't try.
export CUDA_VISIBLE_DEVICES=""

: "${MODULE:?need MODULE (e.g. visword.analysis.platonic_trained)}"
PYTHONPATH=src python -u -m "$MODULE"

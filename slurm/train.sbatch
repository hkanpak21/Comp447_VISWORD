#!/usr/bin/env bash
#SBATCH --job-name=vw-train
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/train_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/train_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:55:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CONFIG="${CONFIG:-configs/salad_main.yaml}"

# Training entry-point lands in Phase B. This stub exists so Phase A's layout
# is complete and `submit.sh train` is wire-able today.
echo "TODO(PhaseB): python -u -m visword.train --config $CONFIG $*"
echo "SLURM script OK; exiting 0 until Phase B lands."

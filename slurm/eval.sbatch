#!/usr/bin/env bash
#SBATCH --job-name=vw-eval
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/eval_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/eval_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN_DIR="${RUN_DIR:-latest}"

echo "TODO(PhaseD): python -u -m visword.eval_phase1 --run-dir $RUN_DIR"
echo "TODO(PhaseD): python -u -m visword.eval_phase2 --run-dir $RUN_DIR"
echo "SLURM script OK; exiting 0 until Phase D lands."

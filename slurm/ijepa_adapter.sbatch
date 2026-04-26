#!/usr/bin/env bash
#SBATCH --job-name=vw-ijepa-adapt
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/ijepa_adapt_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/ijepa_adapt_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00

set -euo pipefail
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

cd /scratch/hkanpak21/VISWORD
PYTHONPATH=src python -u -m scripts.ijepa_text_adapter \
    --n-train 4000 --n-eval 1000 --steps 3000 --seed 42

echo
echo "=== summary ==="
cat /scratch/hkanpak21/VISWORD/runs/ijepa_adapter/adapter_report_seed42.json

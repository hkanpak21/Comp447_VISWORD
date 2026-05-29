#!/usr/bin/env bash
#SBATCH --job-name=vw-ijepa-adapt
#SBATCH --output=runs/_slurm/ijepa_adapt_%j.out
#SBATCH --error=runs/_slurm/ijepa_adapt_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHONPATH=src python -u -m scripts.ijepa_text_adapter \
    --adapter-kind linear \
    --n-train 4000 --n-eval 1000 --steps 3000 --seed 42 \
    --out-dir "$PROJECT_ROOT"/runs/ijepa_text_adapter_sweep

echo
echo "=== summary ==="
cat "$PROJECT_ROOT"/runs/ijepa_text_adapter_sweep/summary.json

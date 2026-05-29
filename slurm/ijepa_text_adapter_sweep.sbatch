#!/usr/bin/env bash
#SBATCH --job-name=vw-ijepa-text-sweep
#SBATCH --output=runs/_slurm/ijepa_text_sweep_%j.out
#SBATCH --error=runs/_slurm/ijepa_text_sweep_%j.err
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
# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHONPATH=src python -u -m scripts.ijepa_text_adapter \
    --sweep \
    --n-train "${N_TRAIN:-4000}" \
    --n-eval "${N_EVAL:-1000}" \
    --steps "${STEPS:-3000}" \
    --seed "${SEED:-42}" \
    --out-dir "${OUT_DIR:-"$PROJECT_ROOT"/runs/ijepa_text_adapter_sweep}"

echo
echo "=== summary ==="
cat "${OUT_DIR:-"$PROJECT_ROOT"/runs/ijepa_text_adapter_sweep}/summary.json"

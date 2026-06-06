#!/usr/bin/env bash
#SBATCH --job-name=vw-ablations
#SBATCH --output=runs/_slurm/eval_ablations_%j.out
#SBATCH --error=runs/_slurm/eval_ablations_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
source scripts/_env.sh

RUN_DIR="runs/2026-06-03_070502_acfc947c_ijepa-text-target-all-blocks-ful_e27e"

echo "=== Running 15% Top Blanking Ablation ==="
python -u -m visword.eval_phase1_holdout --run-dir "$RUN_DIR" --blank-page-top-frac 0.15

echo "=== Running 30% Right Blanking Ablation ==="
python -u -m visword.eval_phase1_holdout --run-dir "$RUN_DIR" --blank-page-right-frac 0.30

echo "=== Done! ==="

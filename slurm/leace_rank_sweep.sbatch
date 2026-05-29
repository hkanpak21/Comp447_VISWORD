#!/usr/bin/env bash
#SBATCH --job-name=vw-leace-sweep
#SBATCH --output=runs/_slurm/leace_sweep_%j.out
#SBATCH --error=runs/_slurm/leace_sweep_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== trained DINOv2-SALAD-50k rank sweep ==="
PYTHONPATH=src python -u scripts/leace_rank_sweep.py \
    --orig runs/_trained_descr/dinov2_salad_50k_orig_mtr0.npz \
    --blank runs/_trained_descr/dinov2_salad_50k_blank15_mtr0.npz \
    --ranks 1 4 16 64 256 1024 4000 \
    > runs/_trained_descr/dinov2_salad_50k_rank_sweep.txt
cat runs/_trained_descr/dinov2_salad_50k_rank_sweep.txt

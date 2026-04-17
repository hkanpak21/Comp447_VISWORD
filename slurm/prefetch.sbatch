#!/usr/bin/env bash
#SBATCH --job-name=vw-prefetch
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/prefetch_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/prefetch_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

TARGET_ROWS="${TARGET_ROWS:-21000}"

python -u /scratch/hkanpak21/VISWORD/scripts/prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss \
    --target-rows "$TARGET_ROWS" \
    --resume

# Also pull the anchors dataset (tiny — ~30 MB).
python -u /scratch/hkanpak21/VISWORD/scripts/prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss-anchors \
    --resume

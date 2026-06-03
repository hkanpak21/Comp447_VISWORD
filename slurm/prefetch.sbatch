#!/usr/bin/env bash
#SBATCH --job-name=vw-prefetch
#SBATCH --output=runs/_slurm/prefetch_%j.out
#SBATCH --error=runs/_slurm/prefetch_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=07:55:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source scripts/_env.sh

TARGET_ROWS="${TARGET_ROWS:-21000}"

python -u scripts/prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss \
    --target-rows "$TARGET_ROWS" \
    --resume

# Also pull the anchors dataset (tiny — ~30 MB).
python -u scripts/prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss-anchors \
    --resume

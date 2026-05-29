#!/usr/bin/env bash
#SBATCH --job-name=vw-pf-arrow
#SBATCH --output=runs/_slurm/prefetch_arrow_%j.out
#SBATCH --error=runs/_slurm/prefetch_arrow_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:30:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

ARROW_FILE_START="${ARROW_FILE_START:?need ARROW_FILE_START}"
ARROW_FILE_END="${ARROW_FILE_END:?need ARROW_FILE_END}"
ARROW_IDX_BASE="${ARROW_IDX_BASE:?need ARROW_IDX_BASE}"
TARGET_ROWS="${TARGET_ROWS:?need TARGET_ROWS}"

python -u "$(dirname "${BASH_SOURCE[0]}")/../scripts/"prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss \
    --arrow-file-start "$ARROW_FILE_START" \
    --arrow-file-end   "$ARROW_FILE_END" \
    --arrow-idx-base   "$ARROW_IDX_BASE" \
    --target-rows      "$TARGET_ROWS" \
    --resume

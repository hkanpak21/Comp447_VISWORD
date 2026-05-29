#!/usr/bin/env bash
#SBATCH --job-name=vw-pf-split
#SBATCH --output=runs/_slurm/prefetch_split_%j.out
#SBATCH --error=runs/_slurm/prefetch_split_%j.err
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
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

SPLIT_START="${SPLIT_START:?need SPLIT_START}"
SPLIT_END="${SPLIT_END:?need SPLIT_END}"
SPLIT_IDX_BASE="${SPLIT_IDX_BASE:?need SPLIT_IDX_BASE}"
TARGET_ROWS="${TARGET_ROWS:?need TARGET_ROWS}"

python -u "$(dirname "${BASH_SOURCE[0]}")/../scripts/"prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss \
    --split-start "$SPLIT_START" \
    --split-end   "$SPLIT_END" \
    --split-idx-base "$SPLIT_IDX_BASE" \
    --target-rows "$TARGET_ROWS"

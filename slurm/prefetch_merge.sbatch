#!/usr/bin/env bash
#SBATCH --job-name=vw-pf-merge
#SBATCH --output=runs/_slurm/prefetch_merge_%j.out
#SBATCH --error=runs/_slurm/prefetch_merge_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

python -u "$(dirname "${BASH_SOURCE[0]}")/../scripts/"prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss \
    --merge-shards

#!/usr/bin/env bash
#SBATCH --job-name=vw-inspect
#SBATCH --output=runs/_slurm/inspect_parquet_%j.out
#SBATCH --error=runs/_slurm/inspect_parquet_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
python -u "$(dirname "${BASH_SOURCE[0]}")/../scripts/"inspect_wiki_ss_arrow.py \
    > "$PROJECT_ROOT"/data/wiki_ss_arrow_index.json
echo "wrote "$PROJECT_ROOT"/data/wiki_ss_arrow_index.json"

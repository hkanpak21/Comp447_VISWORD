#!/usr/bin/env bash
#SBATCH --job-name=vw-pf-merge
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/prefetch_merge_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/prefetch_merge_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

python -u /scratch/hkanpak21/VISWORD/scripts/prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss \
    --merge-shards

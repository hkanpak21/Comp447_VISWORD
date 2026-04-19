#!/usr/bin/env bash
#SBATCH --job-name=vw-pf-shard
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/prefetch_shard_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/prefetch_shard_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=07:55:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

START_IDX="${START_IDX:?need START_IDX}"
END_IDX="${END_IDX:?need END_IDX}"

python -u /scratch/hkanpak21/VISWORD/scripts/prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    --resume

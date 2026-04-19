#!/usr/bin/env bash
#SBATCH --job-name=vw-inspect
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/inspect_parquet_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/inspect_parquet_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
source /scratch/hkanpak21/VISWORD/scripts/_env.sh
python -u /scratch/hkanpak21/VISWORD/scripts/inspect_wiki_ss_arrow.py \
    > /scratch/hkanpak21/VISWORD/data/wiki_ss_arrow_index.json
echo "wrote /scratch/hkanpak21/VISWORD/data/wiki_ss_arrow_index.json"

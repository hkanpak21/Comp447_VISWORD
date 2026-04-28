#!/usr/bin/env bash
#SBATCH --job-name=vw-paper-build
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/paper_build_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/paper_build_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -uo pipefail
cd /scratch/hkanpak21/VISWORD/paper/report_template
./build.sh

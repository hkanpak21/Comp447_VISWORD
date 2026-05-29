#!/usr/bin/env bash
#SBATCH --job-name=vw-paper-build
#SBATCH --output=runs/_slurm/paper_build_%j.out
#SBATCH --error=runs/_slurm/paper_build_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."/paper/report_template
./build.sh

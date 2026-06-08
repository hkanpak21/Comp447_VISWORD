#!/usr/bin/env bash
#SBATCH --job-name=vw-doc-eval
#SBATCH --output=runs/_slurm/doc_eval_%j.out
#SBATCH --error=runs/_slurm/doc_eval_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:ampere_a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail
source scripts/_env.sh

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Doc-family encoders to eval (ticket 03). Resumable: existing result files are skipped.
ENCODERS="${ENCODERS:-pix2struct donut nougat}"
OUT="${OUT:-runs/doc_family_eval}"

cd "$PROJECT_ROOT"
PYTHONPATH=src python -u scripts/rebaseline_grid.py \
    --cache-dir "${DATA_DIR}/wiki_ss" \
    --encoders $ENCODERS \
    --out "$OUT" \
    --crop-size 224 \
    --num-eval 2000 \
    --num-pages 2000 \
    --batch-size 16 \
    --seed 42

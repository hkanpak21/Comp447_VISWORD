#!/usr/bin/env bash
# 3-hour A40 head fine-tune for the I-JEPA variants table (tab:native_legible_ijepa).
# Fills the TBD cells: {image-only, text-target} x {MLP, SALAD} at native 490x490.
#
#   sbatch --export=ALL,CONFIG=configs/ijepa_texttarget_salad_30k_490.yaml slurm/train_a40_3h.sbatch
#
# REQUIRES (neither is available under the comx29/t4_ai association as of 2026-06-13):
#   1. A40 access  — partition `ai` needs AllowAccounts=ai,avg / AllowQos=ai,ai_temp;
#                    comx29's qos is capped to gres/gpu:tesla_t4=1 (no A40, 1 GPU).
#   2. Read access to the pretrained backbones in /scratch/bbakay22 (currently mode 700).
# Until both clear, this job will pend forever (no eligible GPU) or fail loading the backbone.
#
#SBATCH --job-name=vw-ijepa-3h
#SBATCH --output=runs/_slurm/ijepa3h_%j.out
#SBATCH --error=runs/_slurm/ijepa3h_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:ampere_a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/_env.sh

python scripts/ensure_dinov2_hub.py || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

CONFIG="${CONFIG:?usage: --export=ALL,CONFIG=configs/<ijepa head cfg>.yaml}"
RUN_NAME="${RUN_NAME:-}"
EXTRA_ARGS=()
[ -n "$RUN_NAME" ] && EXTRA_ARGS+=(--run-name "$RUN_NAME")

cd "$PROJECT_ROOT"
# eval_every_steps in the config writes periodic R@k to the run dir, so a 3h cap
# still yields a best-so-far number even if 30k x 3ep does not fully converge.
PYTHONPATH=src python -u -m visword.train --config "$CONFIG" "${EXTRA_ARGS[@]}" "$@"

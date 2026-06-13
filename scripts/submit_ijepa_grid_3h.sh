#!/usr/bin/env bash
# Queue the four I-JEPA head fine-tunes (BERT/text-target + image-only/normal),
# each with a 3-hour cap, chained afterany so they run one-at-a-time in priority
# order (SALAD head-to-head first — that pair answers Q5 — then the MLP pair).
#
# Run from the repo root on Valar:  bash scripts/submit_ijepa_grid_3h.sh
#
# PRECONDITIONS (verified missing 2026-06-13 — see slurm/train_a40_3h.sbatch):
#   - A40 access (comx29 cannot reach partition `ai`; qos capped to 1 T4).
#   - Read access to /scratch/bbakay22 backbones (currently mode 700).
# This script intentionally does NOT auto-run; submitting before the above are
# fixed only creates jobs that pend forever or die on backbone load.
set -euo pipefail

SBATCH_FILE="slurm/train_a40_3h.sbatch"

# Priority order: text-target+SALAD and image-only+SALAD first (the Q5 head-to-head),
# then the two MLP variants. BERT (text-target) and normal (image-only) balanced.
CONFIGS=(
  configs/ijepa_texttarget_salad_30k_490.yaml
  configs/ijepa_imageonly_salad_30k_490.yaml
  configs/ijepa_texttarget_mlp_30k_490.yaml
  configs/ijepa_imageonly_mlp_30k_490.yaml
)

prev=""
for cfg in "${CONFIGS[@]}"; do
  dep=()
  [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
  jid=$(sbatch --parsable "${dep[@]}" --export=ALL,CONFIG="$cfg" "$SBATCH_FILE")
  echo "submitted $jid  <-  $cfg  ${dep[*]:-}"
  prev="$jid"
done
echo "queued ${#CONFIGS[@]} jobs (3h cap each, sequential via afterany)."

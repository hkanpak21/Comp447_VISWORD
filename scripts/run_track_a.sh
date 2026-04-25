#!/usr/bin/env bash
# Track-A submitter: queue the 4 single-seed encoder-grid fine-tunes.
#
# Each cell submits as: train -> eval_full chained on afterany.
# afterany (not afterok) so a failed/truncated eval doesn't kill the
# next training run (lesson from CLIP-low-LR chain 1027565).
#
# Usage: scripts/run_track_a.sh [grid_subset]
#   grid_subset: comma-separated subset of {dinov2_salad, dinov2_mlp, clip_salad, clip_mlp}
#   defaults to all 4.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBSET="${1:-dinov2_salad,dinov2_mlp,clip_salad,clip_mlp}"
IFS=',' read -ra CELLS <<< "$SUBSET"

declare -A CONFIGS=(
    [dinov2_salad]="configs/grid_dinov2_salad_30k.yaml"
    [dinov2_mlp]="configs/grid_dinov2_mlp_30k.yaml"
    [clip_salad]="configs/grid_clip_salad_30k.yaml"
    [clip_mlp]="configs/grid_clip_mlp_30k.yaml"
)

cd "$HERE"
echo "Track-A submission for cells: ${CELLS[*]}"

for cell in "${CELLS[@]}"; do
    cfg="${CONFIGS[$cell]:-}"
    if [ -z "$cfg" ]; then
        echo "unknown cell '$cell' (valid: ${!CONFIGS[*]})" >&2
        continue
    fi
    echo
    echo "=== $cell -> $cfg ==="
    TR_ID=$(sbatch --parsable --export=ALL,CONFIG="$cfg" \
              "$HERE/slurm/train.sbatch")
    echo "  train: $TR_ID"
    EV_ID=$(sbatch --parsable --dependency=afterany:"$TR_ID" \
              --export=ALL,RUN_DIR=latest "$HERE/slurm/eval_full.sbatch")
    echo "  eval:  $EV_ID  (afterany:$TR_ID)"
done

echo
echo "Submitted. Check: squeue -u $USER"

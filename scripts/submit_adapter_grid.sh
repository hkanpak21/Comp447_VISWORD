#!/usr/bin/env bash
# Submit the 3×3 adapter ablation grid (DINOv2/CLIP/I-JEPA × Linear/MLP/Bottleneck)
# as a chain of SLURM jobs.  Each job depends on the previous one finishing
# (afterany) so the 1-GPU QoS cap is respected automatically.
#
# Usage:
#   bash scripts/submit_adapter_grid.sh

set -euo pipefail

SBATCH=slurm/adapter_grid.sbatch
PREV_JID=""

CONFIGS=(
    configs/adapter_dinov2_linear.yaml
    configs/adapter_dinov2_mlp.yaml
    configs/adapter_dinov2_bottleneck.yaml
    configs/adapter_clip_linear.yaml
    configs/adapter_clip_mlp.yaml
    configs/adapter_clip_bottleneck.yaml
    configs/adapter_ijepa_linear.yaml
    configs/adapter_ijepa_mlp.yaml
    configs/adapter_ijepa_bottleneck.yaml
)

echo "=== Adapter ablation grid: ${#CONFIGS[@]} jobs ==="
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    DEP_FLAG=""
    if [ -n "$PREV_JID" ]; then
        DEP_FLAG="--dependency=afterany:${PREV_JID}"
    fi
    JID=$(sbatch --parsable $DEP_FLAG \
          --export=ALL,CONFIG="$cfg",RUN_NAME="$name" \
          "$SBATCH")
    echo "  $name → job $JID"
    PREV_JID="$JID"
done
echo "=== All ${#CONFIGS[@]} jobs submitted. Last JID: $PREV_JID ==="

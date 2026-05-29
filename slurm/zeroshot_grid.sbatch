#!/usr/bin/env bash
#SBATCH --job-name=vw-zs-grid
#SBATCH --output=runs/_slurm/zsgrid_%j.out
#SBATCH --error=runs/_slurm/zsgrid_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Track-C: zero-shot Protocol-A retrieval for the 6-encoder grid + Platonic
# alignment heatmap. ~3 GPU-hours total.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

python "$(dirname "${BASH_SOURCE[0]}")/../scripts/ensure_dinov2_hub.py" || true

# DO NOT set offline mode — SigLIP / I-JEPA need to load from local HF cache,
# which the cache_dir env in _env.sh should already point to.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs/_zeroshot

NUM_PAGES="${NUM_PAGES:-2000}"
SEED="${SEED:-42}"

for ENC in dinov2_cls clip_image siglip_image imagenet_vit plain_vit ijepa; do
    OUT="runs/_zeroshot/${ENC}_protocolA_n${NUM_PAGES}.json"
    if [ -f "$OUT" ]; then
        echo "skip ${ENC} (already at $OUT)"
        continue
    fi
    echo "=== zero-shot Protocol-A: ${ENC} ==="
    PYTHONPATH=src python -u -m scripts.zeroshot_protocol_a \
        --encoder "$ENC" --num-pages "$NUM_PAGES" --seed "$SEED" \
        --out "$OUT" || echo "FAILED: $ENC"
done

# Platonic alignment grid across all image + text encoders.
echo "=== Platonic alignment grid ==="
PYTHONPATH=src python -u -m visword.analysis.platonic_alignment \
    --n-samples "$NUM_PAGES" \
    > runs/_zeroshot/platonic_grid_n${NUM_PAGES}.log 2>&1

echo "Done. Results in runs/_zeroshot/"
ls -la runs/_zeroshot/

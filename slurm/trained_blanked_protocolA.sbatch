#!/usr/bin/env bash
#SBATCH --job-name=vw-tr-blank
#SBATCH --output=runs/_slurm/trblank_%j.out
#SBATCH --error=runs/_slurm/trblank_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
python "$(dirname "${BASH_SOURCE[0]}")/../scripts/ensure_dinov2_hub.py" || true

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Trained-checkpoint Protocol-A H-OCR ablation: re-run Protocol-A on
# every Track-A trained checkpoint with the page top region painted
# white. Mirrors the Phase-2 title-blanked sweep, but on Protocol-A
# (the headline metric) — gives the cleanest trained-model H-OCR test.
RUNS=(
    "runs/2026-04-25_113011_6a91c434_visword-grid-dinov2-salad-30k_5b84"
    "runs/2026-04-25_160409_8ed70e75_visword-grid-dinov2-mlp-30k_bbfc"
    "runs/2026-04-25_201550_ab12d8a1_visword-grid-clip-salad-30k_3fcf"
    "runs/2026-04-25_235004_ab12d8a1_visword-grid-clip-mlp-30k_cc42"
    "runs/2026-04-26_041724_ab12d8a1_visword-saturation-dinov2-salad-_af3f"
)
for D in "${RUNS[@]}"; do
    if [ ! -d "$D" ]; then
        echo "skip missing $D"; continue
    fi
    if [ -f "$D/phase1_holdout_blank15.json" ]; then
        echo "skip $D (blank15 already done)"
        continue
    fi
    echo "=== $D blank=0.15 ==="
    PYTHONPATH=src python -u -m visword.eval_phase1_holdout \
        --run-dir "$D" --blank-page-top-frac 0.15
done

echo "=== summary ==="
for D in "${RUNS[@]}"; do
    if [ -f "$D/phase1_holdout.json" ] && [ -f "$D/phase1_holdout_blank15.json" ]; then
        python -c "
import json
o = json.load(open('$D/phase1_holdout.json'))
b = json.load(open('$D/phase1_holdout_blank15.json'))
import os
name = os.path.basename('$D')
print(f'  {name[:60]:<60} orig R@10={o[\"recall\"][\"10\"]:.3f} blanked R@10={b[\"recall\"][\"10\"]:.3f} Δ={b[\"recall\"][\"10\"]-o[\"recall\"][\"10\"]:+.3f}')
"
    fi
done

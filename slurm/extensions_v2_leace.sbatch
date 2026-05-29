#!/usr/bin/env bash
#SBATCH --job-name=vw-ext-leace
#SBATCH --output=runs/_slurm/extleace_%j.out
#SBATCH --error=runs/_slurm/extleace_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
python "$(dirname "${BASH_SOURCE[0]}")/../scripts/ensure_dinov2_hub.py" || true

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs/_zeroshot_descr

# --min-text-ratio 0.0 keeps every grid cell, so orig and blank15 produce the
# same number of crops and pair index-by-index. Required for the LEACE
# direction (orig - blank) to be well-defined per-crop.
for ENC in clip_image dinov2_cls; do
    echo "=== $ENC: encode orig P=2000 (min_text_ratio=0) ==="
    PYTHONPATH=src python -u -m scripts.zeroshot_protocol_a \
        --encoder "$ENC" --num-pages 2000 --seed 42 \
        --blank-page-top-frac 0.0 --min-text-ratio 0.0 \
        --out "runs/_zeroshot/${ENC}_protocolA_n2000_mtr0.json" \
        --save-descriptors "runs/_zeroshot_descr/${ENC}_orig_mtr0.npz" \
        || echo "[$ENC orig FAILED]"

    echo "=== $ENC: encode blank15 P=2000 (min_text_ratio=0) ==="
    PYTHONPATH=src python -u -m scripts.zeroshot_protocol_a \
        --encoder "$ENC" --num-pages 2000 --seed 42 \
        --blank-page-top-frac 0.15 --min-text-ratio 0.0 \
        --out "runs/_zeroshot/${ENC}_protocolA_n2000_blank15_mtr0.json" \
        --save-descriptors "runs/_zeroshot_descr/${ENC}_blank15_mtr0.npz" \
        || echo "[$ENC blank FAILED]"

    echo "=== $ENC: LEACE causal title-erasure ==="
    PYTHONPATH=src python -u -m scripts.leace_title_erasure \
        --orig "runs/_zeroshot_descr/${ENC}_orig_mtr0.npz" \
        --blank "runs/_zeroshot_descr/${ENC}_blank15_mtr0.npz" \
        --out "runs/_zeroshot_descr/${ENC}_leace.json" \
        || echo "[$ENC LEACE FAILED]"
done

echo "=== summary ==="
python -c "
import json
from pathlib import Path
print()
print('LEACE causal title-erasure (P=2000 zero-shot, min_text_ratio=0):')
for enc in ['clip_image', 'dinov2_cls']:
    p = f'runs/_zeroshot_descr/{enc}_leace.json'
    if Path(p).is_file():
        d = json.load(open(p))
        ro = d['recall_orig']['10']; rb = d['recall_blank']['10']; rl = d['recall_leace']['10']
        ro1 = d['recall_orig']['1']; rb1 = d['recall_blank']['1']; rl1 = d['recall_leace']['1']
        print(f'  {enc:<14} R@10  orig={ro:.3f}  blank={rb:.3f}  LEACE={rl:.3f}'
              f'  Δ_blank={rb-ro:+.3f}  Δ_LEACE={rl-ro:+.3f}')
        print(f'  {enc:<14} R@1   orig={ro1:.3f}  blank={rb1:.3f}  LEACE={rl1:.3f}')
"

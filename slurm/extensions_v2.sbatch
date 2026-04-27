#!/usr/bin/env bash
#SBATCH --job-name=vw-ext-v2
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/extv2_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/extv2_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source /scratch/hkanpak21/VISWORD/scripts/_env.sh
python /scratch/hkanpak21/VISWORD/scripts/ensure_dinov2_hub.py || true

cd /scratch/hkanpak21/VISWORD
mkdir -p runs/_zeroshot runs/_zeroshot_descr runs/_colpali

# ---------------------------------------------------------------------------
# 1. Matched-pool comparisons (close the open caveat from §4.6).
# ---------------------------------------------------------------------------
echo "=== single-vector CLIP at P=500 (matched ColPali pool) ==="
PYTHONPATH=src python -u -m scripts.zeroshot_protocol_a \
    --encoder clip_image --num-pages 500 --seed 42 \
    --out runs/_zeroshot/clip_image_protocolA_n500.json

echo "=== late-interaction DINOv2 at P=500 ==="
PYTHONPATH=src python -u -m scripts.colpali_late_interaction \
    --encoder dinov2 --num-pages 500 --seed 42 \
    --out runs/_colpali/dinov2_lateint_n500.json

# ---------------------------------------------------------------------------
# 2. LEACE causal title-region erasure on CLIP and DINOv2 zero-shot.
# Encode 2000-page eval slice twice (orig + blanked), save descriptors,
# fit LEACE on the (orig - blank) direction, re-rank.
# ---------------------------------------------------------------------------
for ENC in clip_image dinov2_cls; do
    echo "=== $ENC: encode orig P=2000 ==="
    PYTHONPATH=src python -u -m scripts.zeroshot_protocol_a \
        --encoder "$ENC" --num-pages 2000 --seed 42 \
        --blank-page-top-frac 0.0 \
        --out "runs/_zeroshot/${ENC}_protocolA_n2000_for_leace.json" \
        --save-descriptors "runs/_zeroshot_descr/${ENC}_orig.npz"

    echo "=== $ENC: encode blank15 P=2000 ==="
    PYTHONPATH=src python -u -m scripts.zeroshot_protocol_a \
        --encoder "$ENC" --num-pages 2000 --seed 42 \
        --blank-page-top-frac 0.15 \
        --out "runs/_zeroshot/${ENC}_protocolA_n2000_blank15_for_leace.json" \
        --save-descriptors "runs/_zeroshot_descr/${ENC}_blank15.npz"

    echo "=== $ENC: LEACE causal title-erasure ==="
    PYTHONPATH=src python -u -m scripts.leace_title_erasure \
        --orig "runs/_zeroshot_descr/${ENC}_orig.npz" \
        --blank "runs/_zeroshot_descr/${ENC}_blank15.npz" \
        --out "runs/_zeroshot_descr/${ENC}_leace.json"
done

echo "=== summary ==="
python -c "
import json
from pathlib import Path

print()
print('matched-pool ColPali test:')
for p in ['runs/_zeroshot/clip_image_protocolA_n500.json',
         'runs/_colpali/clip_lateint_n500.json',
         'runs/_colpali/dinov2_lateint_n500.json']:
    if Path(p).is_file():
        d = json.load(open(p))
        r = d['recall']
        print(f'  {p:<55} R@1={r[\"1\"]:.3f} R@5={r[\"5\"]:.3f} R@10={r[\"10\"]:.3f}')

print()
print('LEACE causal title-erasure (P=2000 zero-shot):')
for enc in ['clip_image', 'dinov2_cls']:
    p = f'runs/_zeroshot_descr/{enc}_leace.json'
    if Path(p).is_file():
        d = json.load(open(p))
        ro = d['recall_orig']['10']; rb = d['recall_blank']['10']; rl = d['recall_leace']['10']
        print(f'  {enc:<14} R@10  orig={ro:.3f}  blank={rb:.3f}  LEACE={rl:.3f}'
              f'  Δ_blank={rb-ro:+.3f}  Δ_LEACE={rl-ro:+.3f}')
"

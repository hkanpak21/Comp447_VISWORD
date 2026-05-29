#!/usr/bin/env bash
#SBATCH --job-name=vw-ext-v3
#SBATCH --output=runs/_slurm/extv3_%j.out
#SBATCH --error=runs/_slurm/extv3_%j.err
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
mkdir -p runs/_zeroshot_descr runs/_trained_descr

# ---------------------------------------------------------------------------
# 1. SigLIP zero-shot LEACE causal title-erasure (extends §4.7 from CLIP +
# DINOv2 to the third image-text contrastive encoder).
# ---------------------------------------------------------------------------
ENC=siglip_image
echo "=== $ENC: encode orig P=2000 (mtr=0) ==="
PYTHONPATH=src python -u -m scripts.zeroshot_protocol_a \
    --encoder "$ENC" --num-pages 2000 --seed 42 \
    --blank-page-top-frac 0.0 --min-text-ratio 0.0 \
    --out "runs/_zeroshot/${ENC}_protocolA_n2000_mtr0.json" \
    --save-descriptors "runs/_zeroshot_descr/${ENC}_orig_mtr0.npz" \
    || echo "[$ENC orig FAILED]"

echo "=== $ENC: encode blank15 P=2000 (mtr=0) ==="
PYTHONPATH=src python -u -m scripts.zeroshot_protocol_a \
    --encoder "$ENC" --num-pages 2000 --seed 42 \
    --blank-page-top-frac 0.15 --min-text-ratio 0.0 \
    --out "runs/_zeroshot/${ENC}_protocolA_n2000_blank15_mtr0.json" \
    --save-descriptors "runs/_zeroshot_descr/${ENC}_blank15_mtr0.npz" \
    || echo "[$ENC blank FAILED]"

echo "=== $ENC: LEACE rank-1 ==="
PYTHONPATH=src python -u -m scripts.leace_title_erasure \
    --orig "runs/_zeroshot_descr/${ENC}_orig_mtr0.npz" \
    --blank "runs/_zeroshot_descr/${ENC}_blank15_mtr0.npz" \
    --out "runs/_zeroshot_descr/${ENC}_leace.json" --rank 1 \
    || echo "[$ENC LEACE FAILED]"

# ---------------------------------------------------------------------------
# 2. Trained DINOv2-SALAD-50k LEACE causal title-erasure. The trained model
# loses 0.242 R@10 under pixel-blanking; LEACE asks whether the title-region
# dependency is purely a single linear axis or a richer subspace.
# ---------------------------------------------------------------------------
RUN_DIR=runs/2026-04-26_041724_ab12d8a1_visword-saturation-dinov2-salad-_af3f
TAG=dinov2_salad_50k

echo "=== $TAG: encode orig P=2000 (mtr=0) ==="
PYTHONPATH=src python -u -m visword.eval_phase1_holdout \
    --run-dir "$RUN_DIR" --blank-page-top-frac 0.0 \
    --min-text-ratio 0.0 \
    --save-descriptors "runs/_trained_descr/${TAG}_orig_mtr0.npz" \
    || echo "[$TAG orig FAILED]"

echo "=== $TAG: encode blank15 P=2000 (mtr=0) ==="
PYTHONPATH=src python -u -m visword.eval_phase1_holdout \
    --run-dir "$RUN_DIR" --blank-page-top-frac 0.15 \
    --min-text-ratio 0.0 \
    --save-descriptors "runs/_trained_descr/${TAG}_blank15_mtr0.npz" \
    || echo "[$TAG blank FAILED]"

echo "=== $TAG: LEACE rank-1 ==="
PYTHONPATH=src python -u -m scripts.leace_title_erasure \
    --orig "runs/_trained_descr/${TAG}_orig_mtr0.npz" \
    --blank "runs/_trained_descr/${TAG}_blank15_mtr0.npz" \
    --out "runs/_trained_descr/${TAG}_leace.json" --rank 1 \
    || echo "[$TAG LEACE FAILED]"

echo "=== summary ==="
python -c "
import json
from pathlib import Path

print()
print('LEACE causal title-erasure (rank-1 PCA Z, P=2000):')
print(f'  {\"encoder\":<22}  R@10 orig  R@10 blank  R@10 LEACE  expl_var')
for label, p in [
    ('siglip_image (zero-shot)', 'runs/_zeroshot_descr/siglip_image_leace.json'),
    ('clip_image  (zero-shot)',  'runs/_zeroshot_descr/clip_image_leace.json'),
    ('dinov2_cls  (zero-shot)',  'runs/_zeroshot_descr/dinov2_cls_leace.json'),
    ('dinov2-SALAD-50k trained',  'runs/_trained_descr/dinov2_salad_50k_leace.json'),
]:
    if Path(p).is_file():
        d = json.load(open(p))
        ro = d['recall_orig']['10']; rb = d['recall_blank']['10']; rl = d['recall_leace']['10']
        ev = d.get('z_explained_variance', float('nan'))
        print(f'  {label:<22}  {ro:>9.3f}  {rb:>10.3f}  {rl:>10.3f}  {ev:>8.3f}')
"

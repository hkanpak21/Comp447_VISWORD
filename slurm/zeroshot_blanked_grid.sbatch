#!/usr/bin/env bash
#SBATCH --job-name=vw-zs-blank
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/zsblank_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/zsblank_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail
source /scratch/hkanpak21/VISWORD/scripts/_env.sh
python /scratch/hkanpak21/VISWORD/scripts/ensure_dinov2_hub.py || true

cd /scratch/hkanpak21/VISWORD
mkdir -p runs/_zeroshot_blanked

# Run each of the six zero-shot encoders with the title region of every
# page painted white before cropping. Frac = 0.15 (matches the
# eval_phase2_titleblanked_15.json runs we already have for trained
# models).
FRAC=0.15
for ENC in clip_image siglip_image dinov2_cls imagenet_vit plain_vit ijepa; do
    OUT="runs/_zeroshot_blanked/${ENC}_protocolA_blank${FRAC}_n2000.json"
    if [ -f "$OUT" ]; then
        echo "skip $ENC (already done at $OUT)"
        continue
    fi
    echo "=== encoder: $ENC, blank-top-frac=$FRAC ==="
    PYTHONPATH=src python -u -m scripts.zeroshot_protocol_a \
        --encoder "$ENC" \
        --num-pages 2000 \
        --seed 42 \
        --blank-page-top-frac "$FRAC" \
        --out "$OUT"
done

echo "=== summary ==="
for ENC in clip_image siglip_image dinov2_cls imagenet_vit plain_vit ijepa; do
    OUT="runs/_zeroshot_blanked/${ENC}_protocolA_blank0.15_n2000.json"
    if [ -f "$OUT" ]; then
        python -c "
import json
d = json.load(open('$OUT'))
r = d['recall']
print(f'  $ENC: blank R@10 = {r[\"10\"]:.3f}, R@1 = {r[\"1\"]:.3f}, gap = {d[\"sanity\"][\"gap\"]:+.3f}')
"
    fi
done

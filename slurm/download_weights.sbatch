#!/usr/bin/env bash
#SBATCH --job-name=vw-download
#SBATCH --output=runs/_slurm/download_%j.out
#SBATCH --error=runs/_slurm/download_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

# Do NOT set offline flags here. We want to connect to HF Hub and OpenAI.
export HF_HUB_ENABLE_HF_TRANSFER=1

python -u -c "
import open_clip
print('Downloading OpenCLIP ViT-B-16 weights...')
open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
print('OpenCLIP downloaded successfully.')

from transformers import AutoModel
print('Downloading I-JEPA ViT-H/14 weights...')
AutoModel.from_pretrained('facebook/ijepa_vith14_1k')
print('I-JEPA downloaded successfully.')
"

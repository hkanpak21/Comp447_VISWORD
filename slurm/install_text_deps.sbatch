#!/usr/bin/env bash
#SBATCH --job-name=vw-pip
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/install_text_deps_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/install_text_deps_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

# One-shot dependency install for method-ladder text/CLIP baselines.
# Uses --no-deps to avoid the cu121 torch ABI trap (VALAR_GUIDE gotcha):
# transformers/sentence-transformers pull in compatible tokenizers+huggingface
# packages we want, but they'd also want to upgrade torch which would break
# everything. We install transitive deps explicitly and freeze torch.

set -euo pipefail
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Install in order of dependency depth.
pip install --no-deps \
    tokenizers \
    safetensors \
    regex \
    filelock \
    fsspec \
    packaging \
    tqdm \
    requests

pip install --no-deps \
    transformers==4.40.2

pip install --no-deps \
    sentence-transformers==2.7.0

# Verify torch is untouched.
python -c "import torch; assert torch.__version__.startswith('2.3'), torch.__version__; print('torch OK', torch.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "from transformers import CLIPModel, AutoTokenizer, AutoModel; print('CLIP + AutoModel OK')"
python -c "import sentence_transformers; print('sentence-transformers', sentence_transformers.__version__)"

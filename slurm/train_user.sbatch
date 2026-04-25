#!/usr/bin/env bash
# Generic training sbatch for any cluster user (not pinned to hkanpak21).
#
# Required env-vars (set via `--export=ALL,VAR=value,...`):
#   CONFIG     — path to a configs/*.yaml file (relative to repo root)
#   REPO_ROOT  — path to the visword repo on YOUR scratch (writable)
#   DATA_CACHE — path to a wiki_ss/ cache directory (read-only is fine)
#   RUNS_ROOT  — directory under YOUR scratch where the run dir is created
#
# Optional:
#   QOS        — your SLURM QoS (defaults to no --qos flag, uses partition default)
#   ACCOUNT    — your SLURM account (defaults to no --account flag)
#   CONDA_ENV  — path to a writable conda env (defaults to the shared visword env)
#
# Submit example (assume you SSH'd in as alice; visword repo at /scratch/alice/visword):
#
#   sbatch \
#     --partition=t4_ai \
#     --time=07:00:00 \
#     --gres=gpu:tesla_t4:1 \
#     --mem=32G \
#     --cpus-per-task=4 \
#     --output=/scratch/alice/visword/_slurm/train_%j.out \
#     --error=/scratch/alice/visword/_slurm/train_%j.err \
#     --export=ALL,CONFIG=configs/grid_dinov2_mlp_30k.yaml,REPO_ROOT=/scratch/alice/visword,DATA_CACHE=/scratch/hkanpak21/VISWORD/data/wiki_ss,RUNS_ROOT=/scratch/alice/visword/runs \
#     /scratch/alice/visword/slurm/train_user.sbatch

set -euo pipefail

: "${CONFIG:?CONFIG env var required}"
: "${REPO_ROOT:?REPO_ROOT env var required}"
: "${DATA_CACHE:?DATA_CACHE env var required}"
: "${RUNS_ROOT:?RUNS_ROOT env var required}"

CONDA_ENV="${CONDA_ENV:-/scratch/hkanpak21/conda_envs/visword}"

cd "$REPO_ROOT"
mkdir -p "$RUNS_ROOT" "$REPO_ROOT/_slurm"

# Activate Python env.
if [ -f "$CONDA_ENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_ENV/bin/activate"
elif [ -d "$CONDA_ENV" ]; then
    export PATH="$CONDA_ENV/bin:$PATH"
else
    echo "ERROR: CONDA_ENV $CONDA_ENV doesn't exist." >&2
    exit 2
fi

# Pin DINOv2 hub cache (no-op if already done).
python scripts/ensure_dinov2_hub.py || true

# Vendor SALAD if missing (read access to a vendored copy works too,
# but we re-clone here to be self-sufficient).
if [ ! -d "$REPO_ROOT/third_party/salad" ]; then
    bash scripts/vendor_salad.sh
fi

# Run training. We override the cache path AND runs-root so they live
# under the submitting user's scratch — never write to hkanpak21's tree.
export PYTHONPATH=src
python -u -m visword.train \
    --config "$CONFIG" \
    --runs-root "$RUNS_ROOT" \
    --set "data.wiki_ss_cache_dir=$DATA_CACHE" \
    "$@"

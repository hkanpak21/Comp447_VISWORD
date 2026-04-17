#!/usr/bin/env bash
# Source-able env setup. Never executed directly.
# Loaded by every SLURM script and by local interactive sessions.

set -euo pipefail

# Load the git module (not on Valar's default PATH).
if command -v module >/dev/null 2>&1; then
    module load git/2.9.5 2>/dev/null || true
fi

export PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Activate the project conda env.
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate /scratch/hkanpak21/conda_envs/visword

mkdir -p "$PROJECT_ROOT/runs/_slurm"

cat <<EOF
--- env ---
host      : $(hostname)
slurm_job : ${SLURM_JOB_ID:-none}
partition : ${SLURM_JOB_PARTITION:-none}
project   : $PROJECT_ROOT
data_dir  : $DATA_DIR
python    : $(which python) ($(python -V 2>&1))
gpus      : $(nvidia-smi -L 2>/dev/null | tr '\n' '|' | sed 's/|$//' || echo none)
-----------
EOF

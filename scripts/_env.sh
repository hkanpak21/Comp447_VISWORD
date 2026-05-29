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
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/${USER}/conda_envs/visword}"
conda activate "$CONDA_ENV_PATH"

mkdir -p "$PROJECT_ROOT/runs/_slurm"

# Load HF_TOKEN from .env if present (used to lift the unauthenticated rate
# limit on huggingface_hub bulk downloads). Tolerates `key = "value"`,
# `key=value`, optional quotes, and either upper- or lower-case key.
if [ -f "$PROJECT_ROOT/.env" ]; then
    _hf_line=$(grep -iE '^[[:space:]]*hf_token[[:space:]]*=' "$PROJECT_ROOT/.env" | head -1 || true)
    if [ -n "$_hf_line" ]; then
        _hf_val=$(echo "$_hf_line" | sed -E 's/^[[:space:]]*[Hh][Ff]_[Tt][Oo][Kk][Ee][Nn][[:space:]]*=[[:space:]]*//' | sed -E 's/^"//; s/"$//; s/^'\''//; s/'\''$//')
        export HF_TOKEN="$_hf_val"
        export HUGGING_FACE_HUB_TOKEN="$_hf_val"
    fi
    unset _hf_line _hf_val
fi

cat <<EOF
--- env ---
host      : $(hostname)
slurm_job : ${SLURM_JOB_ID:-none}
partition : ${SLURM_JOB_PARTITION:-none}
project   : $PROJECT_ROOT
data_dir  : $DATA_DIR
python    : $(which python) ($(python -V 2>&1))
gpus      : $(nvidia-smi -L 2>/dev/null | tr '\n' '|' | sed 's/|$//' || echo none)
hf_token  : $([ -n "${HF_TOKEN:-}" ] && echo "set (${#HF_TOKEN} chars)" || echo "unset")
-----------
EOF

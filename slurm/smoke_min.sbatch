#!/usr/bin/env bash
#SBATCH --job-name=vw-smoke
#SBATCH --output=runs/_slurm/smoke_%j.out
#SBATCH --error=runs/_slurm/smoke_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#
# Ticket 00 step 4 "one tiny SLURM job completes": a fast GPU smoke test proving the
# env + offline model loading work on a compute node. Runs only the quick gpu/integration
# tests (the heavier eval-suite is excluded; it overran the 30-min full-suite job).

set -euo pipefail
# Slurm spools the batch script, so BASH_SOURCE points into /var/spool; use the
# submission dir (the repo root, since we sbatch from there) like prefetch.sbatch does.
cd "${SLURM_SUBMIT_DIR:-/scratch/${USER}/VISWORD}"
# shellcheck source=scripts/_env.sh
source scripts/_env.sh

python scripts/ensure_dinov2_hub.py || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Only the fast forward/param GPU tests — NOT the training-based ones
# (test_determinism / eval-suite run real training and overran on a shared T4).
PYTHONPATH=src python -m pytest -v tests/test_dinov2_cls.py tests/test_dinov2_salad.py

#!/usr/bin/env bash
#SBATCH --job-name=vw-test-gpu
#SBATCH --output=/scratch/hkanpak21/VISWORD/runs/_slurm/test_gpu_%j.out
#SBATCH --error=/scratch/hkanpak21/VISWORD/runs/_slurm/test_gpu_%j.err
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
# shellcheck source=scripts/_env.sh
source /scratch/hkanpak21/VISWORD/scripts/_env.sh

# Make sure the DINOv2 hub cache is still pinned — some code paths
# (e.g. accidental force_reload, or another user's work) can re-fetch
# main which uses PEP 604 syntax and breaks Py 3.9.
python /scratch/hkanpak21/VISWORD/scripts/ensure_dinov2_hub.py || true

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Use an array so spaces inside the -m expression survive word-splitting.
# Override with PYTEST_MARK="fancier expression" on the sbatch command line.
PYTEST_MARK="${PYTEST_MARK:-gpu or integration}"
PYTEST_EXTRA_ARGS=("${PYTEST_EXTRA_ARGS[@]:-}")
if [ -z "${PYTEST_EXTRA_ARGS[0]}" ]; then
    PYTEST_EXTRA_ARGS=("-v")
fi

echo "--- test selector: -m '$PYTEST_MARK' ---"
cd /scratch/hkanpak21/VISWORD
PYTHONPATH=src python -m pytest -m "$PYTEST_MARK" "${PYTEST_EXTRA_ARGS[@]}"

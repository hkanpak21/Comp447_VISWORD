#!/usr/bin/env bash
# Canonical SLURM entry point (PROJECT_SPEC.md §10.4).
#
# Usage:
#   scripts/submit.sh prefetch [target_rows]
#   scripts/submit.sh train configs/salad_main.yaml
#   scripts/submit.sh full  configs/salad_main.yaml [target_rows]
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmd="${1:-}"

case "$cmd" in
  prefetch)
    rows="${2:-21000}"
    JID=$(sbatch --parsable --export=ALL,TARGET_ROWS="$rows" "$HERE/slurm/prefetch.sbatch")
    echo "$JID"
    ;;
  train)
    cfg="${2:?usage: submit.sh train <config.yaml>}"
    JID=$(sbatch --parsable --export=ALL,CONFIG="$cfg" "$HERE/slurm/train.sbatch")
    echo "$JID"
    ;;
  full)
    cfg="${2:?usage: submit.sh full <config.yaml> [target_rows]}"
    rows="${3:-21000}"
    PF_ID=$(sbatch --parsable --export=ALL,TARGET_ROWS="$rows" "$HERE/slurm/prefetch.sbatch")
    echo "prefetch job id: $PF_ID" >&2
    TR_ID=$(sbatch --parsable --dependency=afterok:"$PF_ID" \
              --export=ALL,CONFIG="$cfg" "$HERE/slurm/train.sbatch")
    echo "train job id: $TR_ID" >&2
    EV_ID=$(sbatch --parsable --dependency=afterok:"$TR_ID" \
              --export=ALL,RUN_DIR=latest "$HERE/slurm/eval.sbatch")
    echo "eval job id: $EV_ID" >&2
    echo "$PF_ID $TR_ID $EV_ID"
    ;;
  ""|-h|--help)
    sed -n '2,7p' "${BASH_SOURCE[0]}"
    exit 0
    ;;
  *)
    echo "unknown subcommand: $cmd" >&2
    sed -n '2,7p' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac

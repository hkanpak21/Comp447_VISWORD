#!/usr/bin/env bash
# Submit 8 parallel split-slice prefetch workers + a merge-shards step
# that runs after all 8 complete (via afterok dependency chain).
#
# Usage:
#   scripts/submit_parallel_prefetch.sh
#
# Worker layout (disjoint HF split slices, disjoint idx_base ranges):
#   W1: train[ 15000: 75000)  idx_base 100000
#   W2: train[ 75000:135000)  idx_base 200000
#   ... 8 workers × 60k rows each = ~480k new rows on top of existing 15k.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

JIDS=()
W_ARGS=(
  # split_start split_end idx_base target_rows
  "15000  75000  100000 60000"
  "75000  135000 200000 60000"
  "135000 195000 300000 60000"
  "195000 255000 400000 60000"
  "255000 315000 500000 60000"
  "315000 375000 600000 60000"
  "375000 435000 700000 60000"
  "435000 495000 800000 60000"
)

for i in "${!W_ARGS[@]}"; do
  read -r SS SE IB TR <<<"${W_ARGS[$i]}"
  JID=$(sbatch --parsable \
          --export=ALL,SPLIT_START="$SS",SPLIT_END="$SE",SPLIT_IDX_BASE="$IB",TARGET_ROWS="$TR" \
          "$HERE/slurm/prefetch_split.sbatch")
  echo "W$((i+1)) JID=$JID  split[$SS:$SE) idx_base=$IB target=$TR"
  JIDS+=("$JID")
done

DEP=$(IFS=: ; echo "afterok:${JIDS[*]}")
MERGE_JID=$(sbatch --parsable --dependency="$DEP" "$HERE/slurm/prefetch_merge.sbatch")
echo "merge JID=$MERGE_JID (runs after all 8 workers complete)"
echo "all JIDs: ${JIDS[*]} $MERGE_JID"

#!/usr/bin/env bash
#
# Pin the torch.hub cache of facebookresearch/dinov2 to commit e1277af.
#
# Why this script exists: the DINOv2 main branch migrated to PEP 604 union
# syntax (``float | None``) in late 2024, which fails at import on Python
# 3.9. Our project env is Python 3.9 (cloned from he_ofl; see
# AGENTS/PROJECT_SPEC.md amendment in commit 688d5ae). Commit e1277af is the
# most recent DINOv2 commit before that migration and still supports Python
# 3.9.
#
# torch.hub.load('facebookresearch/dinov2', ...) caches under
# ``~/.cache/torch/hub/facebookresearch_dinov2_main`` and, if the directory
# exists, will reuse it without hitting the network. This script pre-populates
# that cache with the pinned commit. Safe to run repeatedly.
#
# Must run on a machine with internet access (the Valar login node).
set -euo pipefail

PIN_COMMIT="e1277af2ca75c7b07b0e0c6f4ee9180c82ada1b3"
CACHE_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub/facebookresearch_dinov2_main"

if command -v module >/dev/null 2>&1; then
    module load git/2.9.5 2>/dev/null || true
fi

if [ -d "$CACHE_DIR" ] && [ -f "$CACHE_DIR/.visword_pin" ] \
   && [ "$(cat "$CACHE_DIR/.visword_pin")" = "$PIN_COMMIT" ]; then
    echo "DINOv2 hub cache already pinned at $PIN_COMMIT"
    exit 0
fi

rm -rf "$CACHE_DIR"
mkdir -p "$(dirname "$CACHE_DIR")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --quiet https://github.com/facebookresearch/dinov2 "$TMP/dinov2"
( cd "$TMP/dinov2" && git checkout --quiet "$PIN_COMMIT" && rm -rf .git )
mv "$TMP/dinov2" "$CACHE_DIR"
echo "$PIN_COMMIT" > "$CACHE_DIR/.visword_pin"

echo "Pinned DINOv2 hub cache @ $PIN_COMMIT in $CACHE_DIR"

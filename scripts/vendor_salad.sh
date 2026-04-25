#!/usr/bin/env bash
#
# Vendor the official SALAD repo into third_party/salad/ at a pinned commit.
# Per PROJECT_SPEC.md §1.2 the vendored tree is READ-ONLY: never edit files
# under third_party/salad/. To upgrade, delete the directory and re-run this
# script with a new SALAD_COMMIT.
#
# Run from the Valar login node (compute nodes' internet is slow).
#
set -euo pipefail

SALAD_COMMIT="6aede13a3f6c25750bf7fde10209c06cb73060bb"  # serizba/salad HEAD on 2026-04-17
SALAD_REMOTE="https://github.com/serizba/salad"

# Make `module` available so we can load git on Valar (RHEL8 + Lmod).
if [ -z "${LMOD_CMD:-}" ] && [ -f /etc/profile.d/lmod.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/lmod.sh
fi

# Ensure git is on PATH (Valar's default PATH does not include git).
if ! command -v git >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load git/2.9.5
    fi
fi
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git not found on PATH and 'module load git/2.9.5' did not help." >&2
    echo "       Load git manually then re-run this script." >&2
    exit 1
fi

# Resolve project root from the script location (no git dep).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# `third_party/` is empty in the repo and gitignored, so a fresh clone
# may not have it. Create it on demand so this script works on a
# vanilla `git clone` (Colab, colleague's laptop, etc.).
mkdir -p "$PROJECT_ROOT/third_party"
cd "$PROJECT_ROOT/third_party"

if [ -d salad ]; then
    echo "third_party/salad already exists. Delete it to re-vendor." >&2
    exit 1
fi

git clone "$SALAD_REMOTE" salad
cd salad
git checkout "$SALAD_COMMIT"
rm -rf .git

cat > SETUP.md <<EOF
# SALAD vendoring record

Vendored from ${SALAD_REMOTE} at commit \`${SALAD_COMMIT}\`
on $(date -u +%Y-%m-%dT%H:%M:%SZ).

Do not edit files under this directory. To update, delete the directory and
re-run \`scripts/vendor_salad.sh\` with a new commit.

Imported by \`src/visword/models/salad_bridge.py\` via a sys.path manipulation.
EOF

echo "Vendored serizba/salad @ ${SALAD_COMMIT} into third_party/salad/"

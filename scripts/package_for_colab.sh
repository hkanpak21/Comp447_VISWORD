#!/usr/bin/env bash
# Pack the VISWORD source tree into a single zip so it can be uploaded to
# Google Drive and unzipped by the Colab notebook. Keeps third_party/salad
# (vendored) but excludes data caches, checkpoints, run dirs, and __pycache__.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${HERE}/VISWORD_src.zip"
rm -f "$OUT"

cd "$HERE"
zip -r -q "$OUT" \
    src/ \
    configs/ \
    scripts/ \
    slurm/ \
    tests/ \
    notebooks/ \
    third_party/salad/ \
    pyproject.toml \
    .python-version \
    README.md \
    -x '*/__pycache__/*' '*.pyc' '*/.DS_Store' '*/.pytest_cache/*'

SIZE=$(du -h "$OUT" | awk '{print $1}')
echo
echo "wrote $OUT ($SIZE)"
echo
echo "Next steps:"
echo "  1) scp off VALAR:     scp \$USER@valar:\$OUT ."
echo "  2) upload to Drive at: MyDrive/VISWORD/VISWORD_src.zip"
echo "     (the notebook's Part 0 will extract it automatically)"

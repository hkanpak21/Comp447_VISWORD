#!/usr/bin/env bash
# Build the paper on this cluster's TeX Live install.
# The system TeX module's env vars are incomplete (kpathsea finds nothing
# without explicit per-format paths); this script sets them.
#
# Usage: ./build.sh [paper_basename]
#   paper_basename defaults to visword_report.

set -euo pipefail

PAPER="${1:-visword_report}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load TeX Live module if not already in PATH.
if ! command -v pdflatex >/dev/null 2>&1; then
    module load latex/2025 2>/dev/null || true
fi
if ! command -v pdflatex >/dev/null 2>&1; then
    export PATH=/opt/ohpc/pub/apps/latex/2025/bin:$PATH
fi

TEXMFROOT=/opt/ohpc/pub/apps/latex/2025

# Reset module env (it sets TEXINPUTS to a path that doesn't include the
# real tree, which then masks kpathsea defaults).
unset TEXMF TEXMFCNF
export TEXFORMATS="$TEXMFROOT/texmf-var/web2c//"
export TEXINPUTS=".:$TEXMFROOT/texmf-dist/tex//:"
export TFMFONTS=".:$TEXMFROOT/texmf-dist/fonts/tfm//:"
export VFFONTS=".:$TEXMFROOT/texmf-dist/fonts/vf//:"
export ENCFONTS=".:$TEXMFROOT/texmf-dist/fonts/enc//:"
export T1FONTS=".:$TEXMFROOT/texmf-dist/fonts/type1//:"
export TEXFONTMAPS=".:$TEXMFROOT/texmf-var/fonts/map//:$TEXMFROOT/texmf-dist/fonts/map//:"
export BIBINPUTS=".:$TEXMFROOT/texmf-dist/bibtex/bib//:"
export BSTINPUTS=".:$TEXMFROOT/texmf-dist/bibtex/bst//:"
export TEXMFCNF="$TEXMFROOT/texmf-dist/web2c"

cd "$HERE"

echo "=== pdflatex pass 1 ==="
pdflatex -interaction=nonstopmode "$PAPER.tex" >/dev/null
echo "=== bibtex ==="
bibtex "$PAPER" >/dev/null
echo "=== pdflatex pass 2 ==="
pdflatex -interaction=nonstopmode "$PAPER.tex" >/dev/null
echo "=== pdflatex pass 3 ==="
pdflatex -interaction=nonstopmode "$PAPER.tex" >/dev/null

if [ -f "$PAPER.pdf" ]; then
    PAGES=$(grep -oE 'Output written on .* \([0-9]+ pages' "$PAPER.log" | tail -1 \
            | grep -oE '[0-9]+ pages' | head -1)
    echo
    echo "Built $PAPER.pdf — $PAGES"
    echo "Path: $HERE/$PAPER.pdf"
else
    echo "Build failed; see $HERE/$PAPER.log"
    exit 1
fi

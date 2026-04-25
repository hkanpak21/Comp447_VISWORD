# VISWORD report

Submission file: `visword_report.tex`. Bibliography: `visword_report.bib`.
Paper template originally from COMP547 (Koç University) — built on the
`comp547.sty` style file (an ICML-2021 derivative).

## Building

On the Valar cluster:

```
./build.sh
```

`build.sh` sets the per-format kpathsea env vars the system `latex/2025`
module forgets to set (the module's defaults silence kpathsea's standard
search paths so nothing is found). After the script: `visword_report.pdf`
is in this directory.

Off-cluster (Overleaf, MacTeX, etc.) the standard four-pass build is:

```
pdflatex visword_report
bibtex   visword_report
pdflatex visword_report
pdflatex visword_report
```

The `\TODO{...}` placeholders highlight numbers and figures that get
populated by `scripts/aggregate_demo_results.py` once the queued SLURM
jobs (Track A / B / C) finish. After those land, the workflow is:

```
PYTHONPATH=src python -m scripts.aggregate_demo_results
# then transcribe runs/_demo/*.md into the table cells of visword_report.tex
```

## Three positioning angles (per project plan)

The paper is structured around three questions — see the introduction:

1. **Q1.** When and why does a visual encoder beat a text encoder
   (BERT-style) on the same documents?
2. **Q2.** Can a JEPA-style image-only encoder reach CLIP-level
   multimodal alignment via a small post-hoc text adapter?
3. **Q3.** Does the Platonic Representation Hypothesis explain the
   cross-encoder ordering, and does LEACE causal erasure
   disambiguate encoded vs. causally used features?

# issue-06 — Attention "where it reads"

**Owner:** US · **Slice:** E8 · **Status:** needs-triage

## What to build
For each grid encoder and our MAE reader, produce an attention heatmap over a page
showing where the model looks, plus a single number for how much attention mass lands on
text regions (vs title bar / whitespace / figures). This makes the "it reads the body"
claim visual and quantitative.

## Acceptance criteria
- [ ] Heatmap images saved per model on a few example pages.
- [ ] An "attention-on-text" score computed per model.
- [ ] A short comparison (does the reader attend to body vs title?) recorded; numbers in RESULTS.md.

## Blocked by
- issue-02 (encoders available through the wrapper).

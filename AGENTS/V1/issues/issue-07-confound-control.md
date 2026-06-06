# issue-07 — Confound control (random title-region masking)

**Owner:** US · **Slice:** E9 · **Status:** needs-triage

## What to build
Stop the model from identifying a page by its title bar instead of reading. Add random
title/header-region masking during the reader's fine-tuning, then re-run the title-erasure
check (paint out the top of every eval page) on the resulting model. Confirm accuracy no
longer collapses when the title is removed — i.e. the layout shortcut is gone at legible
resolution.

## Acceptance criteria
- [ ] Reader fine-tuned with random title-region masking.
- [ ] Title-erasure delta near zero for the masked-trained reader (vs the large drop seen before).
- [ ] Comparison (masked-trained vs unmasked) recorded; numbers in RESULTS.md.

## Blocked by
- issue-04 (our reader).

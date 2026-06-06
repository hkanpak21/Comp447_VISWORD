# issue-05 — Perfect-text upper bound

**Owner:** US · **Slice:** E5 · **Status:** needs-triage

## What to build
The "if reading were perfect" reference line for the visual pipeline. Feed the dataset's
ground-truth `text` through the frozen language model (BERT) and retrieve with that, at
the page level, on the same legible eval slice. Produce both a title-text and a body-text
version. No OCR engine is built (we have ground-truth text). This bounds how far the
visual encoders and our MAE reader sit from an ideal reader.

## Acceptance criteria
- [ ] Perfect-text recall@k (title and body) computed on the legible eval slice.
- [ ] Reported alongside the visual encoders and the reader for direct comparison.
- [ ] Numbers appended to RESULTS.md.

## Blocked by
- issue-02 (the legible eval slice + scoring).

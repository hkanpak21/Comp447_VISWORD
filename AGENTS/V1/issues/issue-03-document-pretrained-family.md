# issue-03 — Add the document/text-pretrained model family

**Owner:** US · **Slice:** E3 · **Status:** needs-triage

## What to build
Extend the legible comparison grid with models already trained to read rendered
text/documents: Pix2Struct, Donut, Nougat, and ColPali/ColQwen2. Extract a comparable
pooled embedding from each vision tower; route ColPali's multi-vector output to the
existing patch-level (late-interaction) scoring rather than forcing a single vector.
Present this as a labeled reference/ceiling family (different architecture class), with
params + throughput, not as size-matched competitors.

## Acceptance criteria
- [ ] Each model produces a comparable page embedding (or late-interaction score for ColPali).
- [ ] Grid table extended with the document family; family clearly labeled.
- [ ] Params/throughput reported; numbers appended to RESULTS.md.

## Blocked by
- issue-02 (the grid + wrapper); issue-00 (models prefetched offline).

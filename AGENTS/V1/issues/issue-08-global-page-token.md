# issue-08 — Global-page token / two-stream (optional)

**Owner:** US · **Slice:** E6 (optional) · **Status:** needs-triage

## What to build
Native tiles read text well but lose the page's overall layout (a tile doesn't know it's
the title vs a body paragraph). Add a low-resolution whole-page "gist" alongside the
legible local tiles. **Literature note (see [../LITERATURE.md](../LITERATURE.md)):
GlobalDoc shows the global stream should *re-weight / condition* the local features, not
just be concatenated as an extra token — "synergistic global-local fusion is
significantly more effective than simple global vector inclusion."** So implement fusion
where the global gist modulates the local tile embeddings (e.g. element-wise
conditioning / gating), and compare it against naive concatenation as an ablation.
Measure the change in page-level retrieval.

## Acceptance criteria
- [ ] Page embedding combines a global gist with local tiles via *conditioning/re-weighting* (not just concat).
- [ ] Recall@k delta reported for: no-global vs concat-global vs conditioned-global; numbers in RESULTS.md.

## Blocked by
- issue-02. Optional / lower priority within P1.

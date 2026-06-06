# issue-08 — Global-page token / two-stream (optional)

**Owner:** US · **Slice:** E6 (optional) · **Status:** needs-triage

## What to build
Native tiles read text well but lose the page's overall layout (a tile doesn't know it's
the title vs a body paragraph). Add a low-resolution whole-page "gist" token alongside
the legible local tiles and fuse them into the page embedding. Measure the change in
page-level retrieval.

## Acceptance criteria
- [ ] Page embedding combines a global gist with local tiles.
- [ ] Recall@k delta (with vs without the global token) reported; numbers in RESULTS.md.

## Blocked by
- issue-02. Optional / lower priority within P1.

# issue-11 — I-JEPA body-target variant (optional, parity with our MAE reader)

**Owner:** BARIŞ · **Slice:** E4 (Barış's lane, optional) · **Status:** needs-triage

## What to build
Optional parity experiment: re-pretrain (or continue-train) the I-JEPA Text-Target with
the **body** text as the target (`text_source = body`) instead of the title, mirroring
our MAE body-target reader (issue-04). This enables a clean I-JEPA-vs-MAE comparison both
trained on the same body-text objective. Barış's lane because it uses his I-JEPA pretrain
pipeline and checkpoint/GPU access.

## Acceptance criteria
- [ ] An I-JEPA body-target reader trained and evaluated at page level.
- [ ] Compared to the title-target version (issue-10) and, jointly, to our MAE reader (issue-04).
- [ ] Numbers appended to RESULTS.md.

## Blocked by
- None (Barış's pipeline). Optional / parity.

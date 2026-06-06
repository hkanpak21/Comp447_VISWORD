# issue-09 — High-frequency autoencoder arm (optional)

**Owner:** US · **Slice:** E7 (optional) · **Status:** needs-triage

## What to build
Test whether preserving the high-frequency detail of glyphs helps reading. Use an
autoencoder front-end (this folds naturally into the MAE line — MAE is itself a masked
autoencoder, and its decoder/reconstruction objective targets pixel detail) and compare
against plain native crops at a matched patch budget. This is the most novel, highest-risk
arm; run it after the core lands.

## Acceptance criteria
- [ ] An AE/MAE-reconstruction front-end variant evaluated against plain native crops at matched patch budget.
- [ ] Result (does it help reading?) reported; numbers in RESULTS.md.

## Blocked by
- issue-04 (the MAE reader). Optional / later.

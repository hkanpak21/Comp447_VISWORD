# issue-09 — High-frequency autoencoder arm (optional)

**Owner:** US · **Slice:** E7 (optional) · **Status:** needs-triage

## What to build
Test whether preserving the high-frequency detail of glyphs helps reading, at a matched
patch budget vs plain native crops. **Literature constraint (see [../LITERATURE.md](../LITERATURE.md)):
do NOT use a lossy latent VAE — latent compression destroys small text at the tokenizer
stage and cannot be recovered downstream (DA-VAE, InsightTok).** Prefer, in increasing
risk/novelty:
1. **DCT / frequency-domain input** (à la DocPedia) — feed frequency coefficients so high
   resolution + high-frequency text survive without a token explosion. Cheapest; gives
   global + high-freq together.
2. **Conv / wavelet stem** before the ViT — inject high-frequency local features that the
   linear patch-embed otherwise low-passes away.
3. **Base+detail autoencoder** (DA-VAE style: compact base latent + dedicated detail
   channels with a detail-alignment / text-perceptual loss) — most novel, highest risk.

This is the most novel, highest-risk arm; run it after the core lands. Lead with option 1
or 2.

## Acceptance criteria
- [ ] A high-freq front-end (DCT-input OR conv/wavelet stem; NOT a lossy latent VAE) evaluated vs plain native crops at matched patch budget.
- [ ] Result (does it help reading?) reported; numbers in RESULTS.md, with the chosen variant named.

## Blocked by
- issue-04 (the MAE reader). Optional / later.

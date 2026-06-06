# issue-02 — Re-baseline the comparison grid at legible resolution

**Owner:** US · **Slice:** E2 · **Status:** needs-triage

## What to build
Re-run the cross-model comparison now that text is legible. Run the six original
encoders (CLIP, SigLIP, DINOv2, I-JEPA, ImageNet ViT, random ViT) **plus MAE
(`facebook/vit-mae`)** through one shared wrapper that returns a comparable, unit-length
embedding; average each page's tile embeddings into a page vector; score **page-level
same-page re-identification** (recall@k, leave-one-out gallery) on the new legible slice.
Recompute the image↔text alignment correlation and the title-blanking check at this
resolution. Add the I-JEPA Text-Target zero-shot reference row (R@10 = 0.029). Report
trainable-params + throughput next to recall.

## Acceptance criteria
- [ ] Recall@k table for all encoders at legible (native) resolution.
- [ ] Image↔text alignment correlation recomputed; title-blanking deltas recomputed.
- [ ] Wrapper tests (size, unit-length, determinism per encoder) and scoring tests (monotonicity, leave-one-out, same>diff) pass.
- [ ] All numbers appended to RESULTS.md (accumulated, not overwriting baselines).

## Blocked by
- issue-01.

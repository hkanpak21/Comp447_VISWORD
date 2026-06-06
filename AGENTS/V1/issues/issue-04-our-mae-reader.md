# issue-04 — Our reader: text-target on MAE, body target, page-level eval

**Owner:** US · **Slice:** E4 (our lane) · **Status:** needs-triage

## What to build
Our parallel reading model, on a base Barış is **not** using: a pretrained masked
autoencoder, **MAE (`facebook/vit-mae-base`/`-large`)**. Fine-tune it so its features
predict a frozen language model's representation of the page's **body** text
(`text_source = body`), cheaply — only a lightweight predictor head + the last few
blocks, resumable in ≤8h chunks on a 48GB GPU (A40/L40S). Evaluate by **page-level
same-page re-identification** (pool tile embeddings → page vector). Report the same MAE
**before vs after** fine-tuning so the gain is attributable to the method, plus
params/throughput. This is fully independent of Barış's I-JEPA checkpoint.

Scientific note: contrasts with Barış's I-JEPA reader — MAE reconstructs *pixels*,
I-JEPA predicts *features*; both taught to read via a body-text target. MAE's pixel
objective preserves high-frequency detail (good for glyphs), but its encoder still uses
the same low-pass linear patch-embed — so if reading is weak, pair with a conv stem or
smaller patches (this links to issue-09). See [../LITERATURE.md](../LITERATURE.md).

## Acceptance criteria
- [ ] A training run **completes** end-to-end (config, metrics log, checkpoint written).
- [ ] Target is the page **body** (not title); confirmed in the resolved config.
- [ ] Before/after page-level recall@k reported for MAE; params + throughput logged.
- [ ] "Run finishes" smoke test passes; numbers appended to RESULTS.md.

## Blocked by
- issue-01 (legible crops); issue-00 (MAE prefetched).

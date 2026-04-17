# VisWord-SALAD — Prior Research Context

This file is a concise summary of three earlier research sessions. Read it
before making design decisions; it captures bugs we've already hit and
design choices that have a history.

---

## Session 1 — SigLIP baseline (week 1)

**Goal.** Vision-only retrieval over Wikipedia screenshots. Pipeline:
sliding-window crops from tall page images → SigLIP vision encoder → MLP
projection → InfoNCE contrastive training.

**Datasets.**
- `Tevatron/wiki-ss-corpus` — 1.2 M Wikipedia page screenshots with
  `docid`, `title`, `text`, `image` (PIL).
- `hkanpak21/Wikipedia_SS_withanchors` — anchor + positives + negatives
  tuples of screenshots, meant for scroll-based retrieval evaluation.

**What broke (and how we found it).**

1. **Phase 1 = 100 % R@1.** Looked like the model was magical; it wasn't.
   Root cause: the cropper used 25 % **overlap**, so adjacent crops
   shared pixels. The "same-page" retrieval task was pixel matching, not
   semantic document understanding. **Fix:** `NonOverlappingCropper` with
   `stride == crop_size` — zero shared pixels between anchor and positive.

2. **Phase 2 ≈ 0 % R@20.** Two compounding bugs:
   - Title-matching between `wiki-ss-corpus` and `Wikipedia_SS_withanchors`
     failed for most pages (only ~22 k matched out of thousands of unique
     titles in the anchors split).
   - When a match failed, the fallback **rendered fake text with
     `PIL.ImageDraw`** — plain black on white, nothing like a real
     Wikipedia screenshot (no infobox, no formatting). So the model was
     being evaluated against images it had never seen the distribution of.

   **Fix:** `huggingface_hub.snapshot_download(allow_patterns="images/*")`
   pulls the real screenshots from the anchors repo; load them directly
   for evaluation. **Never silently fall back to rendered placeholders.**

3. Minor: `total_mem` should have been `total_memory` on a CUDA device
   property.

**Takeaways we carry forward.**
- Overlap → inflated recall. Always use stride == crop_size.
- If a dataset mapping fails, **fail loudly**, don't fabricate data.
- Phase 2 must use the real downloaded images, not re-rendered text.

---

## Session 2 — DINO v1 + hand-rolled SALAD (week 2)

**Goal.** Switch to a simpler backbone (DINO ViT-S/16, 21 M params vs
SigLIP-B's 86 M) to get a cleaner baseline, then add SALAD aggregation
(Sinkhorn-OT VLAD) on top.

**Setup.** All week-1 fixes incorporated: non-overlapping cropper, real
image downloads for Phase 2, recall sanity checks (similarity gap,
monotonicity), training accuracy tracking, crop-size ablation.

**Numbers we got.**

| Model | Phase 1 R@1 | Phase 2 R@1 |
|---|---|---|
| DINO ViT-S/16 + CLS + MLP (baseline) | **0.607** | **0.587** |
| DINO ViT-S/16 + our reimplemented SALAD | 0.491 | 0.413 |

SALAD *underperformed* the baseline. At the time we hypothesised SALAD was
designed for place recognition (where discarding uninformative sky/road
patches helps) and not for text-heavy documents (where info is uniformly
spread). Also: 5 k samples × 5 epochs is not much budget for a more
complex head.

**What was actually wrong (diagnosed in session 3).** We had
reimplemented SALAD ourselves and dropped the **CLS token branch**. The
official SALAD aggregator concatenates an MLP-projected CLS token with
the VLAD matrix before the final L2 norm — so the global descriptor
always gets the CLS's information for free. Our version threw it away.

**Takeaway.** Do not reimplement canonical modules from memory. Use the
authors' code.

---

## Session 3 — Interpretability + debugging (current)

**Goal.** Understand what's actually going on inside the model before
optimising further. Rework the notebook so every component is traceable
and every claim is inspectable.

**Changes made in the notebook `VisWord_DINO_SALAD_v2.ipynb`.**

1. **Use official SALAD.** Clone `serizba/salad` at runtime, import
   `models.aggregators.salad.SALAD` and `models.backbones.dinov2.DINOv2`
   directly. Our own `DINOv2_SALAD` just wires them together.
2. **Switch to DINOv2 ViT-B/14.** The official SALAD is tuned for
   DINOv2's high-res patch tokens. DINO v1's coarser features are a
   mismatch (and also don't expose a clean patch-tokens API).
3. **Multi-positive batch structure.** `K=4` crops per page in each
   batch; classes are page indices. Required for Multi-Similarity loss.
4. **Multi-Similarity loss.** The paper's loss. We also keep InfoNCE and
   Triplet as switchable options.
5. **Batch composition diagnostics.** Before training, sample a few
   batches and log: positives/query, mean pos-sim, mean neg-sim, hard
   negative fraction. We didn't have this before, which meant we couldn't
   see that the pos/neg gap was ~zero on random init (it is; expected).
6. **Interpretability.** Four new analyses:
   - **ViT last-block CLS→patch attention** via forward pre-hook.
   - **SALAD OT cluster assignments** — hook the score submodule, run
     Sinkhorn, project each cluster's mass back onto the crop image.
   - **Dustbin per-patch mass** — which regions does SALAD discard?
   - **CLS-vs-VLAD similarity decomposition** — slice the final 8448-d
     descriptor at index 8192 (VLAD part) / 8192: (CLS part) and compute
     each half's contribution to same-page vs diff-page similarity.
7. **Patch-level anchor/pos/neg analysis.** For each anchor patch, find
   nearest positive patch and farthest negative patch, draw boxes on all
   three images. Sanity check: do these correspondences look like what
   we'd expect a human to pick?
8. **BERT text baseline.** Sentence-Transformers MiniLM on the `text`
   field, as a what-if-we-had-perfect-OCR upper bound.

**A RAM problem we found the hard way.**

Keeping 20 k decoded `PIL.Image` objects in a Python list OOMs a Colab
session around the 12 k mark. Each tall Wikipedia screenshot is 1–3 MB
decoded. **Fix:** stream rows, store only re-encoded PNG bytes
(10–100× smaller), decode lazily in the `Dataset.__getitem__`, close the
PIL handle right after cropping. Training memory is now flat in
`num_train_samples`.

For the Valar port this is why we have a separate prefetch job: even
encoded bytes in RAM are a liability at 20 k samples, so we persist
everything to disk and `Dataset` reads from disk on demand.

---

## Recurring design principles

Distilled from the three sessions; these show up as hard rules in
`PROJECT_SPEC.md`.

1. **Honesty over convenience.** If a data mapping fails, fail — don't
   improvise a substitute.
2. **Diagnostic logs beat hero metrics.** Mean pos-sim, mean neg-sim,
   hard-neg fraction, dustbin mass, same/diff similarity gap — every one
   of these has caught a bug.
3. **Provenance everywhere.** Know which commit of both our code and the
   vendored SALAD produced each number.
4. **Vendor, don't reimplement.** Especially for peer-reviewed, open-
   source modules like SALAD.
5. **Separate the concerns.** Data download → training → evaluation →
   interpretability. Slow, possibly-unreliable things (network) go in
   their own job so they don't take GPU time down with them.

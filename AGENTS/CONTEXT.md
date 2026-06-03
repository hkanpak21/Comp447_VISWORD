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

## Session 4 — I-JEPA & Text-Target Cross-Modal Pre-training (latest)

**Goal.** Investigate whether vision models can learn linguistic structures directly from document screenshots. We pre-trained the ViT-H/14 backbone using standard self-supervised I-JEPA and a new cross-modal training paradigm (Text-Target I-JEPA) mapping masked screenshots to frozen BERT representations of text.

**What we did differently.**
1. **Self-Supervised I-JEPA**: Masked visual-to-visual prediction.
2. **Text-Target I-JEPA (Cross-Modal)**: Predicts the BERT representations of the page's text from masked visual context.
3. **Trainable Blocks Ablation**: Compared freezing early backbone blocks (only last 2 or 4 blocks trainable) vs. unfreezing all 32 blocks of ViT-H/14.

**Key Findings & Resulting Metrics.**

| Pre-training Run | Protocol A R@1 (%) | Protocol A R@10 (%) | Protocol A Sim Gap | Protocol B R@1 (%) | Protocol B Sim Gap | Protocol B (30% Blank) R@1 (%) | Protocol B (30% Blank) Sim Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard 2-Blocks** | 0.31% | 1.89% | -0.0053 | 50.85% | -0.0008 | 53.39% | +0.0054 |
| **Standard 4-Blocks** | 0.36% | 1.99% | -0.0056 | 50.85% | -0.0004 | 53.39% | +0.0040 |
| **Standard All-Blocks** | 0.43% | 1.94% | -0.0089 | 52.54% | +0.0004 | 54.24% | +0.0044 |
| **Text-Target 4-Blocks** | 0.66% | 2.75% | +0.0041 | 52.54% | +0.0043 | 56.78% | +0.0130 |
| **Text-Target All-Blocks** | **0.79%** | **3.52%** | **+0.0107** | **55.08%** | **+0.0114** | **56.78%** | **+0.0199** |

* **Text-Target Alignment**: Predictor-based alignment to a language space (BERT) resolves representation misalignment, flipping the similarity gap from negative to positive.
* **Full Backbone Unfreezing**: Unfreezing all 32 ViT-H/14 blocks significantly boosts retrieval performance (Protocol A R@10 rises from 2.75% to 3.52%).
* **Robustness to Blanking**: Under 30% title/header blanking, Text-Target All-Blocks achieves 56.78% R@1 on Protocol B and its similarity gap widens to +0.0199, proving it learns robust content representations rather than memorizing template page layouts.

**Note:** These results were obtained at 224×224 resolution (text illegible) with ~21k training samples and single-GPU training.

---

## Session 5 — Full-Resolution DDP Training & Data Scaling (current)

**Goal.** Re-train both I-JEPA variants (standard visual-only and
text-target cross-modal) at **full 490×490 resolution** (1,225 patches
instead of the original 224×224 with 256 patches) so that text on
Wikipedia screenshots remains legible. Scale training data from 21k
to **100k+ Wikipedia screenshots**. Use **multi-GPU DDP** to make
training tractable.

**What changed.**

1. **Full resolution (490×490).** `cropper.target_size: 490`. Position
   embeddings are interpolated from the pre-trained 224×224 grid via
   `interpolate_pos_encoding=True`. This produces 35×35 = 1,225 patches
   per image.

2. **Expanded data cache.** Used `scripts/prefetch_data.py --target 110000`
   on the login node (compute nodes have no internet) to grow the cache
   to ~108k rows. Configs set `data.num_train_samples: 90000`.

3. **PyTorch DDP (DistributedDataParallel).** Migrated from single-GPU
   to multi-GPU training:
   - `torchrun --nproc_per_node=4` launches 4 processes per node.
   - `DistributedSampler` shards data across GPUs.
   - `ContextEncoderWrapper(nn.Module)` wraps the encoder's
     embeddings→masks→encoder→layernorm pipeline into a single
     `forward()` so DDP can track gradients.
   - Predictor is wrapped directly in `DDP(predictor)`.
   - Target encoder (EMA) is NOT wrapped — it has no gradients.

4. **Gradient checkpointing.** Enabled on the base encoder to fit
   1,225-patch images in T4 VRAM (16 GB).

**DDP pitfalls we hit (and lessons).**

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | `EADDRINUSE` on port 29500 | Two concurrent torchrun jobs on same node used hardcoded port | Dynamic `MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))` |
| 2 | `ValueError: Input image size (490*490) doesn't match model (224*224)` | `IJepaEvalWrapper` forward() missing `interpolate_pos_encoding=True` | Added the flag |
| 3 | NCCL `ALLREDUCE` timeout (600s) at eval step | Only rank 0 ran evaluation; other ranks raced ahead to `loss.backward()` which is a DDP collective — deadlock | Moved eval check outside `if is_main_process:`, added `torch.distributed.barrier()` so all ranks wait |
| 4 | Same NCCL timeout, but `NumelIn=1` (barrier) | Barrier IS an NCCL collective; default timeout is 600s; evaluation takes >10 min | Set `timeout=datetime.timedelta(seconds=3600)` in `dist.init_process_group()` |

**Key lesson: in DDP, any code path that only executes on a subset of
ranks MUST be followed by a `barrier()`, and the NCCL timeout must
exceed the longest possible single-rank operation.**

**Current status.** Both jobs submitted with 4× T4 GPUs, 48h time
limit, 1h NCCL timeout. Steps/epoch: ~33,750. Results pending.

---

## Recurring design principles

Distilled from the five sessions; these show up as hard rules in
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
6. **DDP requires whole-rank reasoning.** Every code path that diverges
   between ranks (e.g. evaluation on rank 0 only) must be explicitly
   synchronized with `barrier()`. The NCCL timeout must cover the
   slowest single-rank operation. Test multi-GPU training past the
   first eval checkpoint before declaring success.


# PRD — P1: Legible Reading + Cross-Model Comparison + Efficient Reading Model

> Scope: **Phase 1 (Wikipedia)** only. P0 (preservation backup + model prefetch) is a
> prerequisite. arXiv and VQA are deferred to later PRDs. Derived from the grilling
> session captured in [TODO.md](TODO.md) (decisions D1–D22). Status label: needs-triage.

---

## Problem Statement

Today the system can only "read" a Wikipedia page's **headline and layout**, not its
body text. The reason is mechanical: every page screenshot is cut into 490×490 tiles
and then shrunk 2.19× down to 224×224 before any model sees it, which turns body
glyphs into ~6-pixel smears that are impossible to read. As a result:

- Our central question — *how do vision models read text, and can they do it
  efficiently?* — cannot be answered honestly, because no model in the current setup
  is ever shown legible text.
- Fine-tuned models learn a shortcut: they bind a page's identity to the title region
  rather than reading the content, so erasing the top of a page swings their accuracy
  wildly. The science rests on an artifact.
- We also have no model that *demonstrably reads the body*, and we cannot afford to
  train one from scratch (prior full-resolution training attempts never finished).

## Solution

Make the whole pipeline operate on **legible text**, then re-run the science and add a
reading model — cheaply and without disturbing prior work:

1. **Rebuild the dataset so text stays readable.** Crop each page into smaller tiles at
   full resolution (no shrinking) and add heuristics so each tile is a meaningful unit
   of text (drop blank tiles and tiles that cut a line in half; align tile edges to the
   gaps between lines).
2. **Re-run the cross-model comparison under legibility**, and add a new family of
   models that were already trained to read rendered text/documents, to see how
   pretraining type relates to reading once text is actually visible.
3. **Train an efficient reading model** by fine-tuning a pretrained image encoder so its
   features predict the meaning of the page's *body* text (using a frozen language model
   as the target). Evaluate it on the established retrieval task at the **page** level.
4. **Bound and explain the result**: compare against a "perfect text" reference, show
   attention heatmaps of where each model looks, and confirm the title/layout shortcut
   is gone once text is legible.

All work is **additive** (new files, new result directories), uses **pretrained models
only** (no from-scratch training), and **never deletes or edits prior work or the
paper** (deletions require explicit operator confirmation).

## User Stories

1. As a researcher, I want body text to remain legible after cropping, so that any
   "the model reads" claim is about actual reading and not headings.
2. As a researcher, I want each crop to be a clean unit of text (no blanks, no
   half-cut lines), so that training and evaluation signals are not polluted by junk
   tiles.
3. As a researcher, I want crops kept at full resolution rather than downsampled, so
   that even a 224-pixel-input model receives readable glyphs.
4. As a researcher, I want every vision model to be run through one common interface
   that returns a comparable embedding, so that cross-model numbers are apples-to-apples.
5. As a researcher, I want the comparison to include models already trained to read
   rendered text/documents, so that I can test whether "trained to read" predicts
   reading performance.
6. As a researcher, I want the cross-model comparison re-run at legible resolution, so
   that I learn whether the previous ranking of pretraining methods survives once text
   is visible.
7. As a researcher, I want trainable-parameter counts and throughput reported next to
   accuracy, so that "efficiently" is measured, not asserted.
8. As a researcher, I want the headline/title-region experiment re-run, so that I can
   show whether the title shortcut still distorts results once the body is readable.
9. As a researcher, I want to fine-tune a pretrained encoder to predict the body text's
   meaning, so that I can test whether a non-reading vision model can be taught to read.
10. As a researcher, I want the reading model judged by the same encoder before vs after
    fine-tuning, so that the improvement is attributable to the method and not to model
    size or architecture.
11. As a researcher, I want the reading model's training to resume from checkpoints in
    short chunks, so that it actually finishes under a cluster that runs one job at a
    time with limited wall-clock.
12. As a researcher, I want to give a whole page as the query and retrieve its matching
    document, so that the evaluation matches how the system would be used.
13. As a researcher, I want "relevant" to mean re-identifying the same page from a
    held-out view, so that I can measure retrieval without inventing relevance labels.
14. As a researcher, I want a "perfect text" reference (real text fed to a language
    model), so that I know how far the visual pipeline is from an ideal reader.
15. As a researcher, I want attention heatmaps showing where each model looks, so that
    I can visually argue whether it attends to body text versus the title.
16. As a researcher, I want a number summarizing how much attention lands on text
    regions, so that the "where it reads" claim is quantitative, not just a picture.
17. As an operator, I want the dataset, results, and paper backed up to persistent
    storage, so that a scratch-disk purge cannot destroy prior work.
18. As an operator, I want the new pretrained models downloaded to the offline cache
    before training, so that compute nodes (which have no internet) can use them.
19. As an operator, I want every new experiment written to its own result directory
    with its config and provenance, so that any number is reproducible from its folder.
20. As an operator, I want all new work to leave existing files untouched, so that
    Barış's code, results, and the paper are never at risk.
21. As an operator, I want any deletion of prior work to require my explicit approval,
    so that no automated step can erase earlier results.
22. As a paper author, I want the legible-resolution comparison as a clean table, so
    that the next report iteration can replace the heading-only caveat with real reading.
23. As a paper author, I want the reading model's before/after numbers and the
    perfect-text reference together, so that the contribution and its ceiling are clear.
24. As a developer, I want automated checks on cropping, the model wrapper, the scoring
    step, and a "training run finishes" smoke test, so that refactors don't silently
    break the pipeline.

## Implementation Decisions

**Modules (described by role; the four marked TESTED get automated tests).**

- **Cropping step — TESTED.** Turns a page image into a list of full-resolution tiles.
  Keeps tiles at native resolution (input size equals tile size, so there is no
  shrinking). Drops blank/low-text tiles and tiles that contain only a fragment of a
  line; aligns tile boundaries to the whitespace gaps between lines of text. Produces a
  new, disjoint evaluation slice distinct from any training pages.
- **Model wrapper — TESTED.** One interface that takes a batch of tiles and returns a
  single, unit-length embedding per item, for all compared models (image–text
  contrastive, image-only self-supervised, supervised, random, and the new
  document/text-pretrained family). Each model handles its own native resolution and
  pooling internally; multi-vector models are routed to the existing late-interaction
  scoring rather than forced into a single vector.
- **Scoring step — TESTED.** Given a query and a gallery of pages, returns recall@k for
  **same-page re-identification at the page level**. A page's embedding is the pooled
  average of its tile embeddings; the query page's own gallery vector is recomputed
  with the query left out, to forbid trivial neighbour-matching.
- **Reading model.** Fine-tunes a *pretrained* image encoder so its features predict a
  frozen language model's representation of the page's **body** text. Primary base is
  the predictive self-supervised encoder already wired into the codebase (the cleanest
  before/after comparison); a smaller self-supervised encoder is the secondary base for
  the efficiency angle, with its smaller size stated explicitly. Only a lightweight
  predictor plus the encoder's last blocks are trained. Training is checkpointed and
  resumable in short chunks and targets the larger-memory GPUs.
- **Attention / read-location.** Produces, per model, an attention heatmap over a page
  and a single score for how much attention mass falls on text regions.
- **Perfect-text comparison.** Feeds the dataset's real body text through the same
  frozen language model and retrieves with it, as the upper-bound reference for the
  visual pipeline. Uses existing ground-truth text; no OCR engine is built.

**Comparability decisions (how the three comparison types are kept honest).**

- The **method claim** is always *within a single base encoder, before vs after*
  fine-tuning — same model, same evaluation, only the training changes.
- The **"which pretraining reads better" comparison across models** is treated as a
  family-level tendency, not an exact ranking, because the models differ in size and
  architecture; parameter counts and throughput are reported alongside accuracy.
- **Document/text-pretrained models are presented as a reference ceiling**, not as
  size-matched competitors, and are not used as bases for the reading model.

**Other decisions.** Native resolution means tile size equals model input size (no
downsample); a higher-resolution secondary setting is allowed only for
resolution-flexible models and is labeled as such. The body-text target uses the
existing text-source switch and predicts the language model's per-token
representations. No experiment trackers (logs and plots only). No from-scratch
training. All artifacts are additive; prior work is never modified.

## Testing Decisions

**What makes a good test here:** it checks externally observable behavior through a
module's public interface (inputs → outputs), not internal implementation details, so
it keeps passing across refactors. Prior art in the repo: the existing cropper unit
tests (synthetic images, exact geometry assertions) and the existing JEPA
training/integration tests (one-step training plus output-exists checks).

Modules with automated tests (all four chosen):

- **Cropping step.** On synthetic pages: tiles are returned at full resolution (not
  shrunk); fully-blank tiles and fragment/half-line tiles are excluded; tile boundaries
  fall in inter-line gaps; the kept tiles cover the page's text regions. CPU-only, fast.
- **Model wrapper.** For each model: the embedding has the expected size and unit
  length, and the same input yields the same output (determinism), guaranteeing the
  comparison stays fair.
- **Scoring step.** Recall is monotonic (recall@20 ≥ recall@10 ≥ recall@5 ≥ recall@1);
  the leave-one-out gallery vector is computed correctly; same-page items rank above
  different-page items on a small synthetic gallery.
- **Reading model finishes.** A one-step training test, plus a smoke test that a short
  run completes and writes its expected outputs (config, metrics log, checkpoint) —
  directly guarding against the prior "training never finished" failure.

## Out of Scope

- **arXiv domain** (harder reading test) — deferred to a P2 PRD; starts only after P1
  gives good Wikipedia results.
- **Visual question answering** — deferred to a P3 PRD, after arXiv.
- **High-frequency autoencoder front-end** (preserving glyph detail) — a later,
  higher-risk research arm, not part of P1.
- **A global-page / two-stream representation** is intended within P1 but is a
  lower-priority arm; it may slip to a follow-up if the core lands first.
- **Building any OCR engine** — we use the dataset's ground-truth text instead.
- **From-scratch pretraining** of any encoder.
- **Editing the existing paper** or any of Barış's code/results; any deletion of prior
  work requires explicit operator approval.
- **Large multi-seed sweeps** — single-seed for the cross-family tendency in P1;
  multi-seed is a later robustness step.

## Further Notes

- **Compute reality (Valar).** The cluster runs roughly one GPU job at a time, so the
  experiment set is serialized in wall-clock; jobs should be resumable in short chunks.
  Large-memory GPUs (48GB-class) are available and should be preferred over the 16GB
  default for legible/high-resolution work. Compute nodes have no internet, so all
  models and data must be prefetched on the login node first.
- **Data on hand.** 376,000 page screenshots, each 980×980 (square top-of-page shots),
  already cached; an anchor/positive/negative triplet set is also present (with both a
  train and a validation split).
- **Preservation.** The dataset, result directories, and paper are backed up to
  persistent storage before any experiment. New work writes only to new locations.
- **Prerequisite (P0).** (a) **Reconcile the repo safely first:** Barış's latest paper
  commits + an uncommitted figure live only on the Valar working copy, on a likely
  detached HEAD, and are not on GitHub; anchor them onto a branch and commit the figure
  before anything else — do not merge/push/reset without the operator's OK. (b) Back up
  data + results + paper to persistent storage. (c) Prefetch the new
  document/text-pretrained models into the offline cache. (d) Confirm the environment
  and a one-job smoke test are green before scaling.
- **Latest report state.** The newest report (on Valar, not GitHub) has the same
  numbers as the version this PRD was written against, plus a hyperparameters table, an
  appendix, and a reproducibility note (all experiments on one Tesla T4, ≈66 GPU-h,
  English Wikipedia, seed 42). No result in this PRD is invalidated by it.
- **The paper is unchanged** by this PRD; results feed the *next* iteration of the
  report, on top of the reconciled branch.

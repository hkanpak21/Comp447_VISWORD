# V1 — Results ledger (APPEND-ONLY)

> **RULE — accumulate, never overwrite.** A new measurement is a **new dated row**.
> Never edit or delete an existing number; keep the full history so trends are visible.
> If a result is superseded, add the new row and (optionally) tag the old one
> `(superseded)` — but leave it in place. Mark the current best per task with ★.
>
> **Every row must be reproducible:** date · what was measured · resolution/setup ·
> config · run-dir (on Valar) or paper source · git SHA · the metric and its value.
>
> Numbers below the "Baseline" header come from the v3 / 2026-05-02 report and are
> **reference only — do not edit.** New P1 numbers go in the P1 section.

---

## Baseline — from the report (reference only, DO NOT EDIT)

Setup: 224×224 input (490 crop → 224, the 2.19× shrink), single Tesla T4, seed 42,
P=2000 eval pool. Source: `paper/report_template/visword_report.tex` @ Valar HEAD.

| date | what | metric | value | source |
|---|---|---|---|---|
| 2026-04 | CLIP ViT-B/16 zero-shot, Protocol-A | R@10 | 0.779 | report Table (zeroshot) |
| 2026-04 | SigLIP ViT-B/16 zero-shot, Protocol-A | R@10 | 0.624 | report |
| 2026-04 | DINOv2 ViT-B/14 zero-shot, Protocol-A | R@10 | 0.058 | report |
| 2026-04 | I-JEPA ViT-H/14 zero-shot, Protocol-A | R@10 | 0.015 | report (below random) |
| 2026-04 | ImageNet ViT-B/16 zero-shot, Protocol-A | R@10 | 0.031 | report |
| 2026-04 | random ViT-B/16, Protocol-A | R@10 | 0.029 | report |
| 2026-04 | DINOv2+MLP fine-tune 30k, Protocol-A | R@10 | 0.787 | report Table (finetune) |
| 2026-04 | DINOv2+SALAD fine-tune 30k, Protocol-A | R@10 | 0.880 | report |
| 2026-04 | DINOv2+SALAD fine-tune 50k, Protocol-A | R@10 | 0.915 | report (title-cheating) |
| 2026-04 | CLIP+SALAD fine-tune 30k, Protocol-A | R@10 | 0.724 | report |
| 2026-04 | I-JEPA → BERT, no adapter, img→title R@10 | R@10 | 0.015 | `runs/ijepa_adapter` |
| 2026-04 | I-JEPA → BERT, + linear adapter, img→title | R@10 | 0.197 | `runs/ijepa_adapter` |
| 2026-04 | Platonic alignment ↔ retrieval | Spearman ρ | 0.83 | report (p=0.04, n=6) |
| 2026-04 | DINOv2-SALAD-50k, top-15% blanked, Protocol-A | R@10 | 0.673 | report (was 0.915) |

### Barış June additions (commit `1bafedc`, 2026-06) — reference only, DO NOT EDIT

Vision-to-vision adapter grid (frozen backbone + adapter head, Protocol-A Phase-1 R@10):

| date | setup | linear | best | source |
|---|---|---|---|---|
| 2026-06 | DINOv2 frozen + adapter | 0.229 ★ | 0.229 (linear) | report Table adapter_vv |
| 2026-06 | I-JEPA frozen + adapter | 0.180 | 0.180 (linear) | report |
| 2026-06 | CLIP frozen + adapter | 0.003 | 0.003 (collapses) | report (no patch variance) |

Image→text adapter capacity sweep (frozen feats → BERT[CLS] of title, R@10):

| date | backbone | no-adapter | best adapter | source |
|---|---|---|---|---|
| 2026-06 | I-JEPA | 0.009 | 0.219 (deep MLP) | report Table adapter — supersedes the single 0.197 (kept above) |
| 2026-06 | CLIP | 0.008 | 0.717 (low-rank 256) | report (CLIP pretrained to align w/ text) |

Text-Target I-JEPA in the zero-shot grid:

| date | setup | metric | value | source |
|---|---|---|---|---|
| 2026-06 | I-JEPA Text-Target (full-res pretrain) zero-shot, Protocol-A | R@10 | 0.029 | report (vs plain I-JEPA 0.015) |
| 2026-06 | I-JEPA Text-Target + MLP (30k) | R@10 | 0.348 | runs/2026-06-04_131101_a1627a72_visword-ijepa-text-target-mlp-30_b195 |
| 2026-06 | I-JEPA Text-Target + SALAD (30k) | R@10 | 0.704 | runs/2026-06-05_102116_a1627a72_visword-ijepa-text-target-salad-_9f04 |

---

## P1 results — append new rows below as they land

Setup for P1: **native-resolution crops (no shrink)**, page-level same-page
re-identification, 48GB-class GPU. Fill run-dir + git SHA for every row.

| date | slice | what (encoder / setup) | resolution | metric | value | run-dir | git SHA | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-06-06 | 01 | legible rebuild: `TextAwareCropper` (line-gap-snapped, native) | native-224 (no shrink) | mean crops/page | **21.75** (old 490→224: 4.0) | `VISWORD_v1/runs/20260606T134506Z_8ced450_legible_crops` | 8ced450 | body text legible (verified vs old smear); + disjoint eval slice **2000 pages**, seed 42 → `eval_split.json` |

### Ticket 02 — re-baseline grid @ legible resolution (NEW protocol — not directly comparable to the 490→224 baseline)

Protocol: **page-level same-page re-identification**, crop query → per-page mean gallery, **leave-one-out**; native-224 legible crops (`TextAwareCropper`); 2000-page disjoint eval slice (39,859 crops); single-vector wrapper; frozen (zero-shot). Run-dir `VISWORD_v1/runs/rebaseline_grid_v1` · git `993fdcc` · job 1144146 (T4). Full R@{1,5,10,20}+sim-gap+params+throughput in `grid_summary.json`.

| date | slice | encoder (frozen, native-224) | metric | R@10 | run-dir | git SHA | notes (R@1 / sim-gap / params) |
|---|---|---|---|---|---|---|---|
| 2026-06-06 | 02 | CLIP ViT-B/16 | R@10 | **0.736 ★** | `rebaseline_grid_v1` | 993fdcc | R@1 0.567 / gap 0.168 / 150M |
| 2026-06-06 | 02 | SigLIP ViT-B/16 | R@10 | 0.697 | `rebaseline_grid_v1` | 993fdcc | R@1 0.521 / gap 0.095 / 203M |
| 2026-06-06 | 02 | DINOv2 ViT-B/14 (CLS) | R@10 | 0.116 | `rebaseline_grid_v1` | 993fdcc | R@1 0.041 / gap 0.033 / 87M |
| 2026-06-06 | 02 | ImageNet ViT-B/16 | R@10 | 0.084 | `rebaseline_grid_v1` | 993fdcc | R@1 0.026 / gap 0.015 / 86M |
| 2026-06-06 | 02 | DINOv2 ViT-B/14 (mean-patch) | R@10 | 0.060 | `rebaseline_grid_v1` | 993fdcc | R@1 0.017 / gap 0.019 / 87M |
| 2026-06-06 | 02 | I-JEPA ViT-H/14 | R@10 | 0.054 | `rebaseline_grid_v1` | 993fdcc | R@1 0.015 / gap 0.012 / 631M (ViT-H, not matched-arch) |
| 2026-06-06 | 02 | random ViT-B/16 | R@10 | 0.042 | `rebaseline_grid_v1` | 993fdcc | R@1 0.009 / gap 0.002 / 86M (floor) |
| 2026-06-06 | 02 | **MAE ViT-B/16** (new) | R@10 | 0.036 | `rebaseline_grid_v1` | 993fdcc | R@1 0.009 / gap 0.001 / 86M — ≈random; zero-shot non-reader → motivates the reader (ticket 04) |

**Finding:** the family ordering survives legibility — image-text contrastive (CLIP/SigLIP) read; image-only SSL (DINOv2, I-JEPA, MAE), supervised, and random sit near the floor. MAE (pixel-reconstruction SSL) lands at the very bottom (≈random), a near-collapsed feature space (sim-gap ≈0) — the non-reader baseline our MAE reader must lift.

#### Title-blanking confound check (top-25% blanked, same protocol/slice) — run-dir `rebaseline_grid_blank25_v1`, git `9cead57`, job 1144522

| date | slice | encoder | metric | R@10 (unblanked) | R@10 (top-25% blanked) | Δ | git SHA |
|---|---|---|---|---|---|---|---|
| 2026-06-06 | 02 | CLIP ViT-B/16 | R@10 | 0.736 | 0.819 | **+0.082** | 9cead57 |
| 2026-06-06 | 02 | SigLIP ViT-B/16 | R@10 | 0.697 | 0.792 | **+0.095** | 9cead57 |
| 2026-06-06 | 02 | DINOv2 (CLS) | R@10 | 0.116 | 0.142 | +0.026 | 9cead57 |
| 2026-06-06 | 02 | ImageNet ViT-B/16 | R@10 | 0.084 | 0.108 | +0.023 | 9cead57 |
| 2026-06-06 | 02 | DINOv2 (mean) | R@10 | 0.060 | 0.074 | +0.014 | 9cead57 |
| 2026-06-06 | 02 | I-JEPA ViT-H/14 | R@10 | 0.054 | 0.068 | +0.014 | 9cead57 |
| 2026-06-06 | 02 | random ViT-B/16 | R@10 | 0.042 | 0.055 | +0.012 | 9cead57 |
| 2026-06-06 | 02 | MAE ViT-B/16 | R@10 | 0.036 | 0.047 | +0.011 | 9cead57 |

**Confound finding (legible res):** blanking the top 25% (the shared Wikipedia logo/search/nav template) *improves* retrieval for **every** encoder and the gain scales with reading ability (readers +0.08–0.10, floor +0.01) — **no encoder drops, so none fingerprints the title**. The template is non-discriminative noise diluting the page mean; removing it helps the readers most. Reproduces+extends the v3 "blanking moves the opposite way" result to the full grid at native-224.

### Ticket 04 — our MAE reader (fine-tune MAE, body→BERT[CLS] regression, page-level eval)

Setup: fine-tune `facebook/vit-mae-base` last-4 blocks + projection head (28.9M trainable) to regress pooled MAE features → frozen BERT[CLS] of the page BODY (Smooth-L1); 20k train pages (head, disjoint from eval), 2 epochs, native-224 crops (≤8/page). Run-dir `VISWORD_v1/runs/mae_reader_v1` · git `110c2e7` · job 1145011 (T4, 2.0h). Eval = same page-level same-page re-id.

| date | what (MAE) | eval gallery | R@1 | R@5 | R@10 | R@20 | sim-gap |
|---|---|---|---|---|---|---|---|
| 2026-06-06 | **frozen MAE (before)** | 2000 pages | 0.009 | 0.023 | **0.036** | 0.053 | 0.001 |
| 2026-06-07 | **MAE reader (after)** | 2000 pages | 0.005 | 0.022 | **0.039** | 0.071 | 0.010 |
| 2026-06-07 | MAE reader (after) | 300 pages (periodic) | 0.029 | 0.101 | **0.169** | 0.277 | — |

**Finding (honest):** the reader trains cleanly (loss→3e-4) and clearly learns *some* page structure — on a **300-page** gallery R@10 jumps to **0.169** — but on the **2000-page** gallery it is only **0.039** (vs frozen 0.036), with a near-collapsed space (sim-gap 0.010). The large 300-vs-2000 gap is the gallery-size sensitivity of a weakly-separated embedding. **Hypothesis (now REFUTED — see ticket 05):** I first suspected the BERT[CLS]-of-body target was weakly discriminative. Ticket 05 disproves it — the **CLS-body target re-ids at R@10 0.747** (mean-pool 0.938). So the target is fine; the reader's failure is **regression-collapse**: Smooth-L1 is minimized (3e-4) by predicting the *tight target centroid* (capturing average proximity, not the fine per-page structure retrieval needs — reader sim-gap 0.010 vs the target's 0.176). The genuine fix is a **contrastive objective** (declined earlier in favour of CLS-regression to match Barış) — strong evidence to reconsider; operator's call. Re-running was NOT done autonomously (objective was the operator's explicit choice).

### Ticket 05 — perfect-text upper bound (ground-truth text → BERT → page-level retrieval)

Run-dir `VISWORD_v1/runs/perfect_text_v1` · git `e0fa001` · job 1145304 (T4, 3.4min). Same seed-42 2000-page slice; body split into 4 chunks (8000 chunk-views) → same `page_reid` LOO protocol as the visual grid; title→body = cross-field recall. Reports both the reader's-target readout (BERT[CLS]) and the stronger mean-pool.

| date | readout | body re-id R@10 (chunk→page LOO) | body sim-gap | title→body R@10 |
|---|---|---|---|---|
| 2026-06-06 | BERT[CLS] (= reader target) | 0.747 | 0.176 | 0.054 |
| 2026-06-06 | BERT mean-pool | **0.938 ★** | 0.198 | 0.571 |

**Finding:** text is **highly discriminative** — perfect-text body-re-id ceiling = **0.94 (mean-pool) / 0.75 (CLS)**. Two consequences: (1) the visual **CLIP (0.736) already sits at the CLS-text level (0.747)**, with clear headroom to the mean-pool ceiling — i.e. better *reading* (not just layout) is what would close the gap; (2) the MAE reader's 0.039 is far below its own target's 0.747 ceiling → confirms regression-collapse, not a weak target.

### Ticket 06 — attention "where it reads" (frozen MAE vs trained reader)

Run-dir `VISWORD_v1/runs/attention_v1` · git `fa7a34a` · job 1145360 (T4, 40s). 8 eval pages / 64 native-224 crops; metric = fraction of last-layer CLS→patch attention mass on INK (non-white) patches; heatmap overlays saved in the run-dir.

| model | attention-on-text |
|---|---|
| frozen MAE | 0.627 |
| MAE reader (after) | 0.627 |

**Finding:** MAE places ~63% of CLS→patch attention on text/ink patches, and the CLS-regression fine-tune did **not** change it (0.6273 → 0.6274) — corroborating regression-collapse: the fine-tune adjusted the head toward the target centroid without altering where the encoder looks. (CLIP / DINOv2 attention needs hook-based extraction — a follow-up.)

### Ticket 04b — contrastive objective (the regression-collapse fix; operator "go on")

Same MAE reader (last-4 blocks + head, 28.9M trainable) but **InfoNCE** between each crop's MAE embedding and its page's **mean-pool BERT-body** anchor (in-batch negatives, 24-page batches), 20k pages, 2 epochs. Run-dir `VISWORD_v1/runs/mae_reader_contrastive_v1` · git `d87b7cb` · job 1145959 (T4, 1.4h).

| date | MAE reader | objective | eval gallery | R@10 | sim-gap |
|---|---|---|---|---|---|
| 2026-06-06 | frozen | — | 2000 | 0.036 | 0.001 |
| 2026-06-07 | regression | Smooth-L1 → BERT[CLS] | 2000 | 0.039 | 0.010 |
| 2026-06-07 | contrastive (2 epochs, 23 neg) | InfoNCE → BERT-mean | 2000 | 0.063 | 0.293 |
| 2026-06-07 | contrastive (2 epochs) | InfoNCE → BERT-mean | 300 (periodic) | 0.216 | — |
| 2026-06-07 | **contrastive (6 epochs, 39 neg)** ★ | InfoNCE → BERT-mean | 2000 | **0.098** | **0.318** |
| 2026-06-07 | contrastive (6 epochs) | InfoNCE → BERT-mean | 300 (periodic) | **0.279** | — |

Continuation (6 epochs total, 40-page batches): run-dir `mae_reader_contrastive_v1` (resumed), job 1146109 (T4, +2.5h).

**Finding:** contrastive **fixes the regression-collapse** and keeps improving with scale. Full reader progression on the hard 2000-page gallery: frozen **0.036** → regression 0.039 → contrastive-2ep 0.063 → **contrastive-6ep 0.098** (2.7× frozen); on 300 pages 0.169 → **0.279**. Sim-gap 0.001 → **0.318** (genuine page separation, no collapse). Still below CLIP (0.736) and the 0.94 perfect-text ceiling — expected, since MAE's pixel-reconstruction features are a weaker starting point than CLIP's image-text-aligned ones.

**Data-scaling test (operator request "scale the reader"):**

| date | run | data × epochs | R@10 @2000 | R@10 @300 | sim-gap | run-dir / job |
|---|---|---|---|---|---|---|
| 2026-06-07 | v1 contrastive | 20k × 6 | 0.098 | 0.279 | 0.318 | `mae_reader_contrastive_v1` / 1146109 |
| 2026-06-07 | v2 contrastive | **80k × 5** | 0.095 | 0.279 | 0.289 | `mae_reader_contrastive_v2` / 1147365 |

**Scaling finding (negative, honest):** 4× more train data (20k→80k, 47 in-batch negatives) gives **no improvement** — the reader **plateaus at ~0.095–0.098 @2000 / 0.279 @300**. The bottleneck is **not data quantity**; it's the MAE-backbone reading ceiling + the parameter-efficient fine-tune (last-4-blocks) + in-batch-negative count. Next levers (not run): full/more-block fine-tune, a memory-bank for many more negatives, or a stronger backbone (document-MAE / DiT).

### Ticket 07 — confound control (random title-region masking)

Title-erasure check (recall normal vs top-25%-blanked) on two contrastive readers (20k×6). Run-dirs `title_erasure_unmasked` / `title_erasure_masked`; masked reader `mae_reader_masked_v1`. git `3664546`, jobs 1148627/1148629/1148866 (T4).

| date | reader | trained w/ title-mask (p=0.5)? | R@10 normal | R@10 blanked | delta (blank−normal) |
|---|---|---|---|---|---|
| 2026-06-07 | contrastive v1 | no | 0.097 | 0.110 | **+0.013** |
| 2026-06-07 | masked variant | yes | 0.143 | 0.176 | **+0.033** |

**Finding (confound controlled):** **both deltas are positive** — recall *rises* when the title is erased, so **neither reader fingerprints the title**; the v3 layout-fingerprint failure (where blanking *dropped* trained-model accuracy) is **gone** at legible resolution (body-text objective + legible crops). **Bonus (best reader):** random title-masking during fine-tuning is not just confound control — it **boosts** the reader: R@10 **0.098 → 0.143 @2000**, 0.279 → **0.371 @300**, sim-gap 0.318 → **0.404** — the strongest MAE reader, by forcing body-content reliance + acting as augmentation.

### Ticket 03 — document/text-pretrained family (zero-shot, native-224 legible crops)

Protocol: same 2000-page seed-42 eval slice, page-level same-page re-identification (LOO), native-224 crops, frozen encoders. Run-dir `runs/doc_family_eval` · git `50f03597` (approx) · T4.

| date | encoder | params | R@1 | R@5 | R@10 | R@20 | sim-gap | crops/s | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-08 | **Nougat-base** (Swin-B, doc OCR) | 74M | 0.011 | 0.033 | **0.049** | 0.075 | +0.002 | 171.8 | `runs/doc_family_eval/nougat.json` |
| 2026-06-08 | **Pix2Struct** (Google, masked-screenshot pretrain) | 92M | 0.012 | 0.035 | **0.054** | 0.082 | +0.029 | 7.8 | `runs/doc_family_eval/pix2struct.json` |

**Finding:** document-pretrained zero-shot models (Nougat, Pix2Struct) sit in the **image-only SSL cluster** (R@10 0.049–0.054), far below CLIP (0.736) and SigLIP (0.697). Pix2Struct has a modest positive sim-gap (0.029) vs Nougat's near-zero gap (0.002), suggesting it encodes slightly more discriminative page structure. Neither encoder approaches the contrastive-pretrained ceiling — document pretraining alone does not yield page-level retrieval without retrieval-specific fine-tuning. Pix2Struct's 7.8 crops/s throughput (vs 171.8 for Nougat) reflects its high-resolution patch processing.

### Ticket 10 — I-JEPA head fine-tunes (image-only vs text-target backbone)

**Set A — 224px downsampled crops (crop_size=490→target_size=224), 30k train pages, git `a1627a72` (already in Baseline section above):**

These runs are the Barış-era results at old resolution (text illegible). Listed here for cross-reference only; the legible-resolution rows (Set B) below are the V1 protocol numbers.

| date | run | metric | value | run-dir |
|---|---|---|---|---|
| 2026-06-04 | I-JEPA Text-Target + MLP (224px, 30k) | R@10 | 0.348 | `runs/2026-06-04_131101_a1627a72_visword-ijepa-text-target-mlp-30_b195` |
| 2026-06-05 | I-JEPA Text-Target + SALAD (224px, 30k) ★ | R@10 | **0.704** | `runs/2026-06-05_102116_a1627a72_visword-ijepa-text-target-salad-_9f04` |

**Set B — 490px native-resolution crops (crop_size=490, target_size=490, legible), 30k train pages, git `50f03597`, jobs run 2026-06-08:**

Protocol: same 2000-page seed-42 slice, page-level same-page re-id (LOO), 4 blocks unfrozen, 2 epochs (MLP) / 3 epochs (SALAD), lr_bb=5e-6, lr_head=5e-4.

| date | slice | backbone | head | R@1 | R@5 | R@10 | R@20 | sim-gap | run-dir | git SHA |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-08 | 10 | I-JEPA image-only (490px pretrain) | MLP | 0.157 | 0.388 | **0.525** | 0.663 | +0.349 | `runs/2026-06-08_063633_50f03597_ijepa-imageonly-mlp-30k-490_26d7` | 50f03597 |
| 2026-06-08 | 10 | I-JEPA Text-Target (490px pretrain) | MLP | 0.157 | 0.389 | **0.526** | 0.661 | +0.349 | `runs/2026-06-08_063633_50f03597_ijepa-texttarget-mlp-30k-490_f6b3` | 50f03597 |
| 2026-06-08 | 10 | I-JEPA Text-Target (490px pretrain) | SALAD ★ | **0.198** | 0.451 | **0.583** | 0.712 | +0.248 | `runs/2026-06-08_063633_50f03597_ijepa-texttarget-salad-30k-490_03c2` | 50f03597 |
| (running) | 10 | I-JEPA image-only (490px pretrain) | SALAD | TBD | TBD | **TBD** | TBD | TBD | job 1184918 (T4, PENDING 2026-06-13) | 50f03597 |

**Finding (ticket 10):** At legible 490px resolution, I-JEPA Text-Target SALAD reaches R@10 **0.583**, clearly above the MLP variants (0.525–0.526). Notably, image-only MLP ≈ text-target MLP (0.525 vs 0.526) — the MLP head is the bottleneck, not the backbone pretraining. The SALAD aggregator unlocks the text-target advantage. Compare: at 224px (text illegible), Text-Target SALAD hit 0.704 — **the downsampled protocol inflates recall** because layout-fingerprint shortcuts are stronger at coarser resolution. At legible 490px, the task is harder and more honest.

_(append: one row per measurement; never overwrite a prior row)_


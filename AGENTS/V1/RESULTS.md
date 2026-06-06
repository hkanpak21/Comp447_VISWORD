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
| 2026-06 | I-JEPA Text-Target + MLP (30k) | R@10 | TBD | Barış running |
| 2026-06 | I-JEPA Text-Target + SALAD (30k) | R@10 | TBD | Barış running |

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

**Finding (honest):** the reader trains cleanly (loss→3e-4) and clearly learns *some* page structure — on a **300-page** gallery R@10 jumps to **0.169** — but on the **2000-page** gallery it is only **0.039** (vs frozen 0.036), with a near-collapsed space (sim-gap 0.010). The large 300-vs-2000 gap is the gallery-size sensitivity of a weakly-separated embedding. **Hypothesis:** the **BERT[CLS]-of-body target is itself weakly discriminative** across Wikipedia pages (CLS pooling is weak; bodies share encyclopedic style) → the reader inherits the target's weak separability. The perfect-text bound (ticket 05, below) tests this directly under CLS vs mean-pool. If confirmed, the fix is a stronger text target (mean-pooled BERT) or a contrastive objective — a follow-up for the operator (objective was chosen = CLS-regression, matching Barış).

_(append: one row per measurement; never overwrite a prior row)_

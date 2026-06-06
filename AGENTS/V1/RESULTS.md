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
| _pending_ | 02 | first single-encoder legible retrieval | native-224 | R@10 | — | — | — | first new number |

_(append: one row per measurement; never overwrite a prior row)_

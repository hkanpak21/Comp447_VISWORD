# VisWord — Colab notebook

One merged notebook: `VISWORD_all.ipynb`. Six sections running in a single
Colab runtime so state (Drive mount, repo clone, installed deps, HF cache,
loaded models) is reused across experiments.

## Suggested GPU per Part

| Part | Purpose | GPU | Wallclock |
|---|---|---|---|
| Part 0 | Clone repo, mount Drive, install deps, pin DINOv2 | any | 5 min |
| Part 1 | Download 100k–500k rows to Drive | CPU (T4 OK, not used) | 1–6 h |
| Part 2 | Rows 1–6: random ViT, ImageNet-ViT, DINO-v1, DINOv2, mean-patch, CLIP-image | T4 | 30 min |
| Part 3 | Rows 16–20: BERT, MiniLM, CLIP-text, CLIP cross-modal | T4 | 30 min |
| Part 4 | Rows 7–12: linear probe, fine-tune variants, loss ablations, SALAD | **A100 or L4** | 2–8 h |
| Part 5 | Attention / cluster / dustbin / CLS-vs-VLAD maps per trained run | T4 | 30 min |

Total GPU-hour budget: comfortably fits inside 50 GPU-hours if A100 or L4 is
used for Part 4.

## Folder layout on Drive (created automatically)

```
MyDrive/VISWORD/
├── data/wiki_ss/          # prefetched cache (manifest.json + blobs/ + texts/)
├── data/wiki_ss_anchors/  # anchor triplets
├── runs/<row_id>/         # one dir per experiment row
└── hf_cache/              # HuggingFace model/dataset cache
```

## Run order

Run Part 0 first in every session. Then jump to whichever Part you want —
Drive persists everything between sessions. Typical session flow:

1. First session: Part 0 → Part 1 (run the prefetch to ~500k)
2. Later session: Part 0 → Part 2 → Part 3 (collect baselines quickly)
3. Long session: Part 0 → Part 4 (training; use A100/L4)
4. Final session: Part 0 → Part 5 (interpret all trained runs)

## Scientific alignment with VALAR runs

All results are written in the **same schema** (`phase1_recall.json`,
`phase2_recall.json`) as the VALAR pipeline, so numbers tabulate side-by-side
with the cluster runs in the final writeup.

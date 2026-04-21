# VisWord — Colab notebooks

Self-contained notebooks mirroring the VALAR experimental suite, designed
to run on Google Colab when Colab has internet access to HuggingFace
(VALAR DNS currently blocks `huggingface.co` / `pypi.org` for compute).

All notebooks persist to **Google Drive** so intermediate cache + run
directories survive Colab session timeouts.

## Suggested GPU per notebook

| Notebook | Purpose | GPU | ~wallclock |
|---|---|---|---|
| `00_setup.ipynb` | Clone repo, mount Drive, install deps | any | 5 min |
| `01_prefetch.ipynb` | Download 100k–500k rows from HF | CPU (T4 OK, not needed) | 30–90 min |
| `02_zeroshot_vision.ipynb` | Rows 1–6 (random ViT, ImageNet-ViT, DINO-v1, DINOv2, CLIP-image) | T4 | 30 min |
| `03_zeroshot_text_multimodal.ipynb` | Rows 16–20 (BERT / MiniLM / CLIP-text / CLIP cross-modal) | T4 | 30 min |
| `04_train_ladder.ipynb` | Fine-tune ladder (linear probe, CLS-main, SALAD-main, CLIP+SALAD variants) | **A100 or L4** | 2–8 h |
| `05_interpret.ipynb` | Attention / cluster / dustbin maps per trained run | T4 | 30 min |

Total GPU-hour budget: comfortably fits inside 50 GPU-hours if A100 or L4
is used for `04_train_ladder.ipynb`.

## Folder layout on Drive (created automatically)

```
MyDrive/VISWORD/
├── data/wiki_ss/          # prefetched cache (manifest.json + blobs/ + texts/)
├── data/wiki_ss_anchors/  # anchor triplets
├── runs/<row_id>/         # one dir per experiment row
└── hf_cache/              # HuggingFace model/dataset cache
```

## Scientific alignment with VALAR runs

Each notebook writes results in the **same schema** as the VALAR pipeline
(`phase1_recall.json`, `phase2_recall.json`, `methodology.md`) so numbers
can be tabulated side-by-side with the cluster runs in the final writeup.

## Run order

Recommended order: 00 → 01 → (02 + 03 in parallel) → 04 → 05.

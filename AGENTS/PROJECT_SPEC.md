# VisWord-SALAD — Project Spec for Coding Agent

> **Read this whole file before writing any code.** It describes a research
> project on vision-based document embedding (DINOv2 + SALAD) running on a
> SLURM cluster named **Valar** with **slow internet**. The project layout,
> job structure, and acceptance tests below are mandatory. Open questions
> and known uncertainties are flagged `TODO(agent)` — don't silently
> resolve them; ask.

---

## 0. One-paragraph context

We are building a vision-based retrieval system for text documents
(Wikipedia screenshots, not OCR'd text). A **DINOv2 ViT-B/14 backbone**
produces patch tokens; a **SALAD aggregator** (Sinkhorn-OT VLAD, Izquierdo
& Civera, CVPR '24) pools them into an 8448-d global descriptor. We train
with a multi-positive contrastive loss on 10–20 k pages and evaluate
recall@K on (i) non-overlapping same-page crops, (ii) real scroll-based
anchor/positive/negative screenshots. Three prior research sessions are
summarised in `CONTEXT.md`. Several interpretability analyses (OT
assignment maps, dustbin mass, CLS-vs-VLAD contribution, patch-level
nearest-neighbour) are part of the deliverable.

---

## 1. Hard constraints (read first)

1. **Internet is slow and unreliable on Valar compute nodes.** Assume compute
   nodes *may not* have internet at all. All HuggingFace and GitHub access
   happens in a dedicated `prefetch` SLURM job submitted to a partition that
   does have internet (probably the login / transfer partition — confirm
   locally, `TODO(agent)`). Training, eval, and interpretability jobs read
   **only from the local cache** on `$SCRATCH` or `$DATA_DIR`.

2. **Official SALAD code is vendored, not pip-installed.** We clone
   `github.com/serizba/salad` at a pinned commit into `third_party/salad/`
   and import its modules via a Python path manipulation in
   `src/visword/salad_bridge.py`. Any file in `third_party/salad/` is
   **read-only** — never edit. If you need to wrap or patch behaviour, do
   it in a wrapper module inside `src/visword/`.

3. **Every experiment is reproducible from the `runs/` directory alone.**
   Each run directory contains the resolved YAML config, `git rev-parse HEAD`
   of both this repo and `third_party/salad/`, a fingerprint of the dataset
   cache, full stdout/stderr, metrics as JSONL, plots as PNG, and the final
   checkpoint. No hidden global state. No writing to `$HOME` from training
   jobs except via the `runs/` tree.

4. **No experiment trackers.** No W&B, no MLflow, no TensorBoard (Valar
   egress restrictions and the agent operator's preference). Plain JSONL
   logs and matplotlib PNGs only. `scripts/summarise_run.py` reads a run
   directory and prints a human-readable summary.

5. **Do not invent data.** If a dataset path is missing, fail loudly with a
   clear error message that points to the prefetch job. Never fall back to
   rendered placeholder images — we hit that bug in week 1 and it silently
   produced a 0 % recall result that looked plausible.

6. **Use SLURM job dependencies.** `train` jobs must be submitted with
   `--dependency=afterok:<prefetch_job_id>`. `eval` depends on `train`.
   The `submit.sh` wrapper handles this automatically; the agent should
   not encourage users to run bare `sbatch` commands.

---

## 2. Repository layout

```
visword-salad/
├── README.md                    # 1-page quickstart
├── CONTEXT.md                   # prior-sessions summary (provided)
├── PROJECT_SPEC.md              # this file
├── TESTS.md                     # acceptance tests (provided)
├── pyproject.toml
├── .gitignore                   # ignore runs/, data/, third_party/salad/
├── .python-version              # 3.11
│
├── src/
│   └── visword/
│       ├── __init__.py
│       ├── config.py            # Pydantic Config models + YAML loader
│       ├── paths.py             # central path resolution; reads $DATA_DIR
│       ├── seed.py              # seed_everything()
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── prefetch.py      # downloads & caches HF datasets
│       │   ├── manifest.py      # writes/reads cache manifest JSON
│       │   ├── light_dataset.py # lazy-decode PyTorch Dataset
│       │   ├── cropper.py       # NonOverlappingCropper
│       │   └── anchors.py       # Phase-2 anchor dataset loader
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── salad_bridge.py  # adds third_party/salad to sys.path; re-exports SALAD + DINOv2 classes
│       │   ├── dinov2_salad.py  # DINOv2 + official SALAD (our wrapper)
│       │   ├── dinov2_cls.py    # CLS-only baseline
│       │   ├── bert_text.py     # BERT baseline for §12
│       │   ├── ijepa_predictor.py # Meta-style official visual predictor
│       │   ├── ijepa_masks.py    # Mask Collators for I-JEPA context/target selection
│       │   └── ijepa_text_predictor.py # Text predictor mapping to language space
│       │
│       ├── losses.py            # InfoNCE multi-positive, Multi-Similarity, Triplet
│       ├── train.py             # main training entry point
│       ├── train_ijepa.py       # standard visual-only I-JEPA pre-training loop
│       ├── train_ijepa_text.py  # text-target cross-modal I-JEPA pre-training loop
│       ├── eval_phase1.py       # non-overlapping same-page recall
│       ├── eval_phase2.py       # scroll-based anchor/pos/neg recall
│       │
│       ├── interpret/
│       │   ├── __init__.py
│       │   ├── attention.py     # ViT CLS->patch attention hook
│       │   ├── salad_internals.py # hook-based SALAD assignment capture
│       │   ├── cls_vs_vlad.py   # similarity contribution decomposition
│       │   ├── dustbin.py       # dustbin-mass evolution
│       │   └── patch_neighbours.py # §11 anchor/pos/neg patch analysis
│       │
│       ├── diagnostics/
│       │   ├── __init__.py
│       │   └── batch_stats.py   # positives-per-query, hard-neg fraction
│       │
│       └── reporting/
│           ├── __init__.py
│           ├── run_dir.py       # creates runs/<id>/... layout + records provenance
│           ├── jsonl_logger.py  # append-only JSONL logger
│           └── plots.py         # matplotlib helpers with blueish palette
│
├── configs/
│   ├── default.yaml             # base config
│   ├── debug.yaml               # 500 samples, 1 epoch — for dry runs
│   ├── baseline_cls.yaml        # DINOv2 + CLS head
│   ├── salad_main.yaml          # DINOv2 + SALAD, 10k samples
│   ├── salad_20k.yaml           # same but 20k samples (high-RAM node only)
│   ├── ijepa_pretrain_2blocks.yaml # standard I-JEPA (last 2 blocks trainable)
│   ├── ijepa_pretrain_4blocks.yaml # standard I-JEPA (last 4 blocks trainable)
│   ├── ijepa_pretrain_all_blocks.yaml # standard I-JEPA (all 32 blocks trainable)
│   ├── ijepa_text_target.yaml   # text-target cross-modal I-JEPA (last 4 blocks trainable)
│   ├── ijepa_text_target_all_blocks.yaml # text-target cross-modal I-JEPA (all 32 blocks trainable)
│   ├── ijepa_pretrain_all_blocks_full_res.yaml  # standard I-JEPA, all blocks, 490×490 resolution
│   └── ijepa_text_target_all_blocks_full_res.yaml # text-target I-JEPA, all blocks, 490×490 resolution
│
├── scripts/
│   ├── prefetch_data.py         # CLI wrapper for src/visword/data/prefetch.py
│   ├── submit.sh                # SLURM submission with dependency graph
│   ├── resolve_config.py        # merge defaults + overrides, print resolved YAML
│   ├── summarise_run.py         # pretty-print a run directory
│   └── vendor_salad.sh          # clone third_party/salad at pinned commit
│
├── slurm/
│   ├── prefetch.sbatch          # SLURM script: data download
│   ├── train.sbatch             # SLURM script: training
│   ├── eval.sbatch              # SLURM script: eval + interpret
│   ├── ijepa_pretrain.sbatch    # SLURM script: visual I-JEPA pre-training
│   ├── ijepa_text_target.sbatch # SLURM script: cross-modal I-JEPA pre-training
│   └── interactive.sbatch       # salloc wrapper for debugging
│
├── tests/
│   ├── test_cropper.py          # unit tests for NonOverlappingCropper
│   ├── test_losses.py
│   ├── test_salad_bridge.py     # smoke test: import works, forward shape correct
│   ├── test_manifest.py
│   ├── test_ijepa_pretrain.py   # tests for standard visual-only I-JEPA
│   ├── test_ijepa_text_pretrain.py # tests for text-target cross-modal I-JEPA
│   └── test_ijepa_text_adapter.py # tests for post-hoc text adapter mapping
│
├── third_party/
│   └── salad/                   # git clone, pinned commit, READ-ONLY
│       └── SETUP.md             # how it was vendored
│
├── data/                        # populated by prefetch; gitignored
│   ├── wiki_ss/                 # light-row cache: {docid}.png + manifest.json
│   └── wiki_ss_anchors/         # images/ folder from hkanpak21 dataset
│
└── runs/                        # one directory per experiment; gitignored
    └── 2026-04-17_143000_abc1234_salad_main_d4f8/
        ├── config.resolved.yaml
        ├── provenance.json
        ├── stdout.log           # symlink to SLURM output
        ├── stderr.log
        ├── metrics.jsonl        # one JSON object per logged step
        ├── train_curves.png
        ├── phase1_recall.json
        ├── phase2_recall.json
        ├── interpret/
        │   ├── attention_sample0.png
        │   ├── salad_clusters_sample0.png
        │   ├── dustbin_map_sample0.png
        │   ├── cls_vs_vlad.png
        │   └── patch_neighbours_sample{0..3}.png
        └── checkpoints/
            ├── last.pt
            └── best_phase1.pt
```

---

## 3. Config system

### 3.1 Pydantic models (`src/visword/config.py`)

Use Pydantic v2. One top-level `Config` model with nested sections:

```python
class DataConfig(BaseModel):
    wiki_ss_cache_dir: Path        # = $DATA_DIR/wiki_ss
    anchors_cache_dir: Path        # = $DATA_DIR/wiki_ss_anchors
    num_train_samples: int = 10_000
    num_eval_samples:  int = 1_000

class CropperConfig(BaseModel):
    crop_size: int = 490
    overlap: float = 0.0
    min_text_ratio: float = 0.05
    target_size: int = 224

class BackboneConfig(BaseModel):
    arch: Literal['dinov2_vitb14', 'dinov2_vits14'] = 'dinov2_vitb14'
    num_trainable_blocks: int = 4
    feature_dim: int = 768           # ViT-B; 384 for S

class SaladConfig(BaseModel):
    num_clusters: int = 64
    cluster_dim: int = 128
    token_dim: int = 256
    sinkhorn_iters: int = 3

class TrainConfig(BaseModel):
    loss: Literal['infonce', 'multisim', 'triplet'] = 'multisim'
    temperature: float = 0.07
    k_per_page: int = 4              # multi-positive batch structure
    batch_size: int = 32             # rows before flattening; actual = batch_size
    epochs: int = 3
    lr_backbone: float = 1e-5
    lr_head:     float = 5e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42

class EvalConfig(BaseModel):
    k_values: list[int] = [1, 5, 10, 20]
    phase1_max_pages: int = 500
    phase2_max_queries: int = 200

class Config(BaseModel):
    experiment_name: str
    model_kind: Literal['cls', 'salad'] = 'salad'
    data: DataConfig
    cropper: CropperConfig
    backbone: BackboneConfig
    salad: SaladConfig
    train: TrainConfig
    eval:  EvalConfig
```

### 3.2 YAML + env var resolution

`scripts/resolve_config.py` merges `configs/default.yaml` ← named config
(e.g. `configs/salad_main.yaml`) ← CLI overrides (`--set train.epochs=5`),
then substitutes `${DATA_DIR}` and `${SCRATCH}` from env. It prints the
resolved YAML to stdout. Training reads the resolved YAML directly — no
merging at runtime.

### 3.3 Config hash

`provenance.json` in every run dir contains a SHA-1 of the **canonicalised**
resolved YAML (sorted keys, consistent whitespace). Run dir names
incorporate the first 4 chars of this hash. Two runs with the same hash
are guaranteed to have identical config.

---

## 4. Data pipeline (the critical path)

### 4.1 Why a separate prefetch job

Downloading `Tevatron/wiki-ss-corpus` streams ~21k tall PNGs from
HuggingFace. On a slow/intermittent connection this takes 10-60 minutes
and can fail mid-stream. We don't want this inside the training job's
walltime, and we especially don't want compute-partition GPUs idle while
we wait on network IO.

**Solution**: `scripts/prefetch_data.py` runs on a partition that has
internet (`TODO(agent)`: confirm partition name on Valar — candidates are
`cpu`, `login`, `datatransfer`, `staging`). The training job depends on
it via `sbatch --dependency=afterok:<prefetch_job_id>`.

### 4.2 Cache layout (`data/wiki_ss/`)

```
data/wiki_ss/
├── manifest.json                # one entry per row; see below
├── blobs/
│   ├── 00/
│   │   ├── 0000000.png          # encoded PNG bytes, deterministic filename
│   │   └── 0000001.png
│   └── 01/...
└── .fingerprint                 # SHA256 of sorted manifest rows
```

`manifest.json`:
```json
{
  "dataset": "Tevatron/wiki-ss-corpus",
  "hf_revision": "abc1234...",           // commit from HF
  "prefetched_at": "2026-04-17T10:00:00Z",
  "num_rows": 21000,
  "rows": [
    {"idx": 0, "docid": "...", "title": "...", "text_path": "texts/00/0000000.txt", "image_path": "blobs/00/0000000.png", "image_sha256": "..."},
    ...
  ]
}
```

Why this layout:
- **Flat on-disk filenames** indexed by integer — O(1) lookup by `idx`.
- **Shards of 1000** (`00/`, `01/`, ...) keep any one directory to a few
  thousand entries, which lots of filesystems dislike otherwise.
- **`.fingerprint`** lets training jobs verify cache integrity without
  reading every file.
- **Texts written to disk** too, so the BERT baseline (§12) doesn't need
  HuggingFace either.

Anchors dataset (`data/wiki_ss_anchors/`) uses a similar structure but
mirrors the original repo's `images/` folder plus a `splits.json` with
the anchor/positives/negatives tuples per split.

### 4.3 `prefetch_data.py` contract

```
scripts/prefetch_data.py \
    --data-dir $DATA_DIR \
    --dataset wiki-ss \
    --target-rows 21000 \
    --resume                       # pick up after partial download
```

Required behaviour:
1. Stream rows from HF; for each row re-encode the image to PNG bytes
   with `compress_level=1` (fast; images are already ~uncompressed PNG
   internally, this is mostly a decode-roundtrip to drop any extra
   metadata).
2. Write each PNG to `blobs/{idx//1000:02d}/{idx:07d}.png` atomically
   (write to `.tmp` then `os.rename` — never leave half-written files).
3. After every 500 rows, update an in-progress `manifest.json.partial`
   so resume is cheap.
4. When target is reached, rename `manifest.json.partial` →
   `manifest.json` and write `.fingerprint`.
5. Exit **0 only if target is reached**. Partial completion exits 2.
6. Retry any per-row exception up to 3 times with exponential backoff.
   After 3 failures on the same row, log `ERROR` and skip (don't crash).
7. Emit a final summary: `rows_requested`, `rows_written`, `rows_failed`,
   `total_bytes`, `elapsed_seconds`.

### 4.4 Runtime dataset class (`src/visword/data/light_dataset.py`)

```python
class LightWikiScreenshotDataset(Dataset):
    \"\"\"Lazy-decodes PNG bytes from the prefetched cache.\"\"\"

    def __init__(self, cache_dir: Path, indices: list[int], cropper, transform, k_per_page: int = 4):
        self.cache_dir = cache_dir
        self.manifest = json.loads((cache_dir / 'manifest.json').read_text())
        self.rows = [self.manifest['rows'][i] for i in indices]
        self.cropper = cropper
        self.transform = transform
        self.k = k_per_page
```

The `__getitem__` reads `blobs/...` with `PIL.Image.open(...).convert('RGB')`,
runs the cropper, closes the PIL object, stacks K crops into a tensor.
This is essentially the `MultiPositiveCropDataset` from the notebook, but
reading from disk instead of from an in-memory list.

**The key invariant**: at steady state, the training process's RSS is
dominated by model weights + GPU buffers + one batch's worth of decoded
pixels × `num_workers`. It does NOT scale with `num_train_samples`. Verify
with `tests/test_light_dataset.py::test_memory_stays_flat`.

### 4.5 Fingerprint verification

On startup, training reads `.fingerprint` and recomputes it over the
manifest rows. Mismatch → abort with a clear message pointing to the
prefetch script. This catches partial/stale caches.

---

## 5. Official SALAD integration (`src/visword/models/salad_bridge.py`)

### 5.1 Vendoring

`scripts/vendor_salad.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
SALAD_COMMIT="9e6b1a7f7e9e3a4d8f5a0c2b1d3e4f5a6b7c8d9e"  # TODO(agent): pin the actual latest commit
cd "$(git rev-parse --show-toplevel)/third_party"
if [ -d salad ]; then
    echo "third_party/salad already exists. Delete it to re-vendor."
    exit 1
fi
git clone https://github.com/serizba/salad salad
cd salad
git checkout "$SALAD_COMMIT"
rm -rf .git
cat > SETUP.md <<EOF
Vendored from https://github.com/serizba/salad at commit $SALAD_COMMIT on $(date -u +%Y-%m-%dT%H:%M:%SZ).
Do not edit files under this directory. To update, delete the directory and re-run scripts/vendor_salad.sh with a new commit.
EOF
```

### 5.2 Bridge module

`src/visword/models/salad_bridge.py` does exactly this and nothing else:

```python
"""
Exposes the official SALAD aggregator and DINOv2 backbone from
third_party/salad/. This is the single source of truth for the two
classes; do not import from models.aggregators.salad or
models.backbones.dinov2 anywhere else.
"""
import sys
from pathlib import Path

_SALAD_ROOT = Path(__file__).resolve().parents[3] / 'third_party' / 'salad'
if not _SALAD_ROOT.exists():
    raise RuntimeError(
        f'third_party/salad not found at {_SALAD_ROOT}. '
        f'Run scripts/vendor_salad.sh to vendor it.'
    )
if str(_SALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(_SALAD_ROOT))

# Now safe to import
from models.aggregators.salad import SALAD as OfficialSALAD         # noqa: E402
from models.backbones.dinov2 import DINOv2 as OfficialDINOv2        # noqa: E402

try:
    # Used for interpretability (reconstructing Sinkhorn output from hooks)
    from models.aggregators.salad import log_otp_solver              # noqa: E402
except ImportError:
    log_otp_solver = None

__all__ = ['OfficialSALAD', 'OfficialDINOv2', 'log_otp_solver']
```

### 5.3 Provenance recording

Every run's `provenance.json` records:
```json
{
  "visword_git_sha": "...",
  "salad_vendor_sha": "... (read from third_party/salad/SETUP.md)",
  "salad_sha_in_bridge": "...",                // salad_bridge.OfficialSALAD.__module__ + file hash
  "python": "3.11.7",
  "torch": "2.1.0+cu121",
  "cuda": "12.1",
  "gpu": "NVIDIA L4 / A100 / ...",
  "hostname": "valar-gpu-03",
  "slurm_job_id": "12345678"
}
```

---

## 6. Training (`src/visword/train.py`)

CLI:
```
python -m visword.train --config configs/salad_main.yaml \
    [--set train.epochs=5 train.lr_head=1e-3] \
    [--run-name "my-experiment"]
```

Behaviour:

1. Parse config, resolve env vars, compute config hash, create run dir
   `runs/<timestamp>_<short_git_sha>_<run_name_slug>_<config_hash_4>/`.
2. Write `config.resolved.yaml` and `provenance.json`.
3. Verify data cache fingerprint.
4. Build model per `model_kind` (`cls` → `DINOv2CLS`; `salad` → `DINOv2SALAD`).
5. Build dataset + loader. Split train/eval by `indices` from manifest
   (deterministic given `seed`).
6. Build optimiser with two param groups (backbone / head) and
   cosine-warmup schedule.
7. Training loop with per-step logging to `metrics.jsonl`:
   ```
   {"step": 123, "epoch": 0, "loss": 1.34, "top1_acc": 0.68,
    "pos_sim_mean": 0.42, "neg_sim_mean": 0.02, "lr_bb": 1e-5, "lr_head": 3e-4,
    "dustbin_mass": 0.15, "wall_time_s": 47.2, "gpu_mem_gb": 12.3}
   ```
8. Every `eval_every` steps run Phase-1 recall on a held-out 200-page
   subset; write to `metrics.jsonl` as `{"eval_step": ..., "phase1_recall@10": ...}`.
9. Keep two checkpoints: `last.pt` and `best_phase1.pt` (highest R@10).
10. At end: render `train_curves.png` using `reporting/plots.py`.

**Must-not-do:**
- No side effects outside the run directory.
- No downloads. If an import triggers a HuggingFace download (e.g. transformers
  caching), set `HF_HUB_OFFLINE=1` at the start of `train.py` and fail loudly
  if any online call is made.
- No `print` to stdout except a single-line progress bar. All structured
  data goes to `metrics.jsonl`.

---

## 7. Evaluation (`src/visword/eval_phase1.py`, `eval_phase2.py`)

Each script takes a checkpoint path, loads it, runs the corresponding recall
protocol, and writes `phaseN_recall.json` into the run directory:

```json
{
  "checkpoint": "runs/.../checkpoints/best_phase1.pt",
  "num_pages_evaluated": 500,
  "num_crops": 4217,
  "recall": {"1": 0.71, "5": 0.89, "10": 0.93, "20": 0.96},
  "sanity": {"same_page_sim_mean": 0.54, "diff_page_sim_mean": 0.03, "gap": 0.51, "monotonic": true}
}
```

No re-training, no re-loading of unrelated data.

---

## 8. Interpretability (`src/visword/interpret/`)

Each submodule is callable both as a library (for the eval job) and via a
CLI. Output goes to `runs/<id>/interpret/`. Key details from the notebook:

- **`attention.py`**: forward pre-hook on last ViT block's `attn` module,
  reconstructs QK^T softmax (DINOv2 uses fused SDPA internally so we can't
  grab attention weights directly). Returns CLS→patch heatmap reshaped to
  `(side, side)` and overlayed on the crop.

- **`salad_internals.py`**: the robust hook-based discovery. On first
  forward, register shape hooks on all leaf submodules of the SALAD
  aggregator, log their output shapes, pick the ones whose dim matches
  `num_clusters`, `cluster_dim`, `token_dim`. Cache the discovered names
  in the run dir as `interpret/salad_hooks.json` so later calls are
  deterministic.

- **`cls_vs_vlad.py`**: slices the final 8448-d descriptor at index
  `num_clusters * cluster_dim` (= 8192 by default) — the two halves are
  disjoint dimensions, so cosine sim decomposes exactly. No re-forwarding.

- **`dustbin.py`**: reads `dustbin_mass` column from `metrics.jsonl` and
  plots its evolution.

- **`patch_neighbours.py`**: per-patch L2-normalised anchor/pos/neg patch
  tokens; for each anchor patch find nearest positive and farthest
  negative; draw boxes.

---

## 9. Diagnostics (`src/visword/diagnostics/batch_stats.py`)

Called once before training starts, saves `diagnostics/untrained_batch_stats.json`:
```json
{
  "n_batches_sampled": 3,
  "batch_size": 32,
  "k_per_page": 4,
  "positives_per_query_mean": 3.0,
  "negatives_per_query_mean": 28.0,
  "pos_sim_mean": 0.18,
  "neg_sim_mean": 0.17,
  "hard_neg_frac_mean": 0.52,
  "note": "Expected ~0.5 hard-neg fraction at initialisation; should drop over training."
}
```

---

## 10. SLURM job scripts

### 10.1 `slurm/prefetch.sbatch`

```bash
#!/usr/bin/env bash
#SBATCH --job-name=vw-prefetch
#SBATCH --partition=TODO_INTERNET_PARTITION           # TODO(agent): confirm on Valar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=runs/_slurm/prefetch_%j.out
#SBATCH --error=runs/_slurm/prefetch_%j.err

set -euo pipefail
source scripts/_env.sh                                # sets DATA_DIR, activates venv, prints env

python -u scripts/prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss \
    --target-rows 21000 \
    --resume

python -u scripts/prefetch_data.py \
    --data-dir "$DATA_DIR" \
    --dataset wiki-ss-anchors \
    --resume
```

### 10.2 `slurm/train.sbatch`

```bash
#!/usr/bin/env bash
#SBATCH --job-name=vw-train
#SBATCH --partition=TODO_GPU_PARTITION                # TODO(agent)
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/_slurm/train_%j.out
#SBATCH --error=runs/_slurm/train_%j.err

set -euo pipefail
source scripts/_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1        # belt and braces

CONFIG="${CONFIG:-configs/salad_main.yaml}"
python -u -m visword.train --config "$CONFIG" "$@"
```

### 10.3 `slurm/eval.sbatch`

Runs both recall evaluations and the interpretability suite against the run
directory passed in `$RUN_DIR`. One job, several outputs, so we can debug
interpretability without re-running recall.

### 10.4 `scripts/submit.sh` — the canonical entry point

```bash
#!/usr/bin/env bash
# Usage:
#   scripts/submit.sh prefetch
#   scripts/submit.sh train configs/salad_main.yaml
#   scripts/submit.sh full configs/salad_main.yaml    # prefetch -> train -> eval chain
set -euo pipefail
...
case "$cmd" in
  prefetch)
    sbatch --parsable slurm/prefetch.sbatch
    ;;
  train)
    sbatch --parsable --export=CONFIG="$2" slurm/train.sbatch
    ;;
  full)
    PF_ID=$(sbatch --parsable slurm/prefetch.sbatch)
    echo "prefetch job id: $PF_ID"
    TR_ID=$(sbatch --parsable --dependency=afterok:"$PF_ID" \
              --export=CONFIG="$2" slurm/train.sbatch)
    echo "train job id: $TR_ID"
    EV_ID=$(sbatch --parsable --dependency=afterok:"$TR_ID" \
              --export=RUN_DIR="latest" slurm/eval.sbatch)
    echo "eval job id: $EV_ID"
    ;;
esac
```

### 10.5 `scripts/_env.sh`

```bash
# source-able; never executed
export PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export DATA_DIR="${DATA_DIR:-$SCRATCH/visword-data}"
module load python/3.11 cuda/12.1      # TODO(agent): confirm module names on Valar
source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p runs/_slurm
echo "--- env ---"
echo "host     : $(hostname)"
echo "job id   : ${SLURM_JOB_ID:-none}"
echo "DATA_DIR : $DATA_DIR"
echo "python   : $(which python) ($(python -V))"
nvidia-smi || true
echo "-----------"
```

---

## 11. Run directory contract

Name: `runs/<UTC_timestamp>_<git_sha_8>_<config_slug>_<config_hash_4>/`

Required files (training job fails if any are missing at end):

| Path | Who writes | When |
|---|---|---|
| `config.resolved.yaml` | `train.py` | start |
| `provenance.json` | `train.py` | start |
| `metrics.jsonl` | `train.py` | every step |
| `train_curves.png` | `train.py` | end |
| `diagnostics/untrained_batch_stats.json` | `train.py` | before first batch |
| `checkpoints/last.pt` | `train.py` | end |
| `checkpoints/best_phase1.pt` | `train.py` | whenever R@10 improves |
| `phase1_recall.json` | `eval_phase1.py` | eval job |
| `phase2_recall.json` | `eval_phase2.py` | eval job |
| `interpret/*.png` | `interpret/*.py` | eval job |

`runs/_slurm/` holds the raw SLURM stdout/stderr; the run dir contains
relative symlinks back to them for convenience.

---

## 12. Tests (`tests/`, see TESTS.md for full list)

Runnable locally without SLURM. Must pass before the agent considers a
task done.

- **Unit** (CPU, fast): cropper, losses, manifest round-trip, SALAD bridge
  imports, config merge, run-dir naming, provenance recording.
- **Integration** (`pytest -m integration`, needs small cache): end-to-end
  20-step training on `configs/debug.yaml` with 50 pages and 1 epoch;
  check `metrics.jsonl` has the right schema and recall runs cleanly.
- **Smoke** (`pytest -m gpu`, needs GPU): the integration test but with
  `--device=cuda`; verifies no CUDA OOM and forward/backward pass work.

CI can't run GPU or integration tests — leave those for manual.

---

## 13. `CONTEXT.md` (provide separately, summarised below)

The agent should read `CONTEXT.md` before making design changes. It
describes:

- **Week 1 (SigLIP)** — baseline built; 100 % R@1 turned out to be
  pixel-overlap artifact; Phase-2 was 0 % due to broken title matching.
  Fixes: `NonOverlappingCropper` with `stride == crop_size`; real
  screenshots via `snapshot_download`.
- **Week 2 (DINO v1 + reimplemented SALAD)** — dropped CLS branch by
  mistake; SALAD underperformed baseline (Phase 1 R@1: 0.607 vs 0.491).
- **Week 3 (current)** — switch to DINOv2, use **official** SALAD from
  `serizba/salad` so the CLS branch comes back automatically; add
  multi-positive training, interpretability, BERT baseline. The notebook
  `VisWord_DINO_SALAD_v2.ipynb` is the reference — this spec is its
  productionised form.

## 13.1 DDP multi-GPU training rules

Both I-JEPA training scripts (`train_ijepa.py`, `train_ijepa_text.py`)
use PyTorch `DistributedDataParallel` via `torchrun`. The following
rules are hard-won from debugging sessions (see `CONTEXT.md` Session 5):

1. **NCCL timeout must exceed eval time.** `dist.init_process_group()`
   must set `timeout=datetime.timedelta(seconds=3600)` (1 hour). The
   default (600s) is too short for full-resolution evaluation (~1000
   images at 490×490 on T4).

2. **Every rank-divergent code path needs a barrier.** If only rank 0
   runs evaluation or checkpointing, *all ranks* must reach a
   `torch.distributed.barrier()` after the divergent block. Otherwise
   non-rank-0 processes will race ahead to `loss.backward()` (which is
   an implicit DDP collective) and deadlock.

3. **Dynamic torchrun master port.** Use
   `MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))` in sbatch scripts
   to avoid port collisions when multiple jobs land on the same node.

4. **`interpolate_pos_encoding=True` everywhere.** All forward calls to
   the I-JEPA encoder (training, EMA target, eval wrapper) must pass
   this flag when using non-native resolutions (490×490 vs 224×224).

5. **Gradient checkpointing required.** Full-resolution (1,225 patches)
   ViT-H/14 does not fit in T4 VRAM without
   `base_encoder.gradient_checkpointing_enable()`.

---

## 14. What the agent should and should not decide

**Decide autonomously:**
- Code style (follow project's existing style if any, else Black + Ruff defaults).
- Test implementations so long as they verify the contracts in §12.
- Exact commit hash to vendor for SALAD (latest `main` at implementation time; record in SETUP.md).
- Minor refactors that don't change the contracts here.

**Must ask before doing:**
- Changing config schema (would break existing run dirs).
- Adding new dependencies to `pyproject.toml`.
- Anything touching `third_party/salad/`.
- Skipping a test listed in §12 or TESTS.md.
- Replacing `matplotlib` with anything else, or adding a tracker in violation of §1.4.

**Must flag to the operator:**
- SLURM partition names (`TODO_INTERNET_PARTITION`, `TODO_GPU_PARTITION`).
- Python/CUDA module names on Valar.
- Whether compute nodes have any filesystem other than `$SCRATCH`/`$HOME`
  that we should use for the data cache (e.g. `$WORK`, `$PROJECT`).
- The `SALAD_COMMIT` pin in `scripts/vendor_salad.sh`.

---

## 15. Minimum viable delivery (Phase A)

The agent should deliver these, in order, before anything else:

1. `pyproject.toml`, `.gitignore`, directory skeleton — empty but present.
2. `scripts/vendor_salad.sh` + `third_party/salad/SETUP.md`.
3. `src/visword/models/salad_bridge.py` + `tests/test_salad_bridge.py`
   (tests pass on a laptop with CPU).
4. `src/visword/data/prefetch.py` + `scripts/prefetch_data.py` + manifest
   writer and reader + `tests/test_manifest.py`.
5. `slurm/prefetch.sbatch` and `scripts/submit.sh prefetch` working
   end-to-end on Valar, producing a valid cache in `$DATA_DIR`.

Only after Phase A is signed off does the agent move to training, eval,
interpretability, plotting (Phase B → E).

---

## 16. Open questions for the operator

1. Which SLURM partition has internet access on Valar?
2. Which partition has GPUs? Node types (L4? A100? H100?)?
3. Is `$SCRATCH` shared across nodes or node-local? If node-local, we need
   to prefetch on the same node as training, or use a shared `$DATA_DIR`.
4. Expected wall-clock limit on the GPU partition?
5. Do we need to use `apptainer`/`singularity` or is a conda/venv OK?
6. Any institutional proxy for HuggingFace (`HF_ENDPOINT` override)?

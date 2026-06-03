# VisWord-SALAD

Vision-based retrieval over text document screenshots. DINOv2 ViT-B/14 +
official SALAD (Sinkhorn-OT VLAD) aggregator. Trained on Wikipedia page
screenshots; evaluated with non-overlapping same-page recall and
scroll-based anchor/positive/negative retrieval.

---

## For a coding agent: read these, in order

1. **[`AGENTS/CONTEXT.md`](AGENTS/CONTEXT.md)** — what we've already tried, what broke, what we
   learned. Non-negotiable background.
2. **[`AGENTS/PROJECT_SPEC.md`](AGENTS/PROJECT_SPEC.md)** — the authoritative project design: directory
   layout, config system, SLURM job structure, run-dir contract, open
   questions.
3. **[`AGENTS/TESTS.md`](AGENTS/TESTS.md)** — concrete acceptance tests per phase. A phase is
   "done" when its tests are green.

Start at **§15 Minimum viable delivery** of `AGENTS/PROJECT_SPEC.md` —
Phase A is what's already shipped (see status below); Phases B–E are next.

---

## For a human operator: five commands to know

```bash
# 0. One-time per machine: clone the conda env from he_ofl
conda create --prefix /scratch/$USER/conda_envs/visword --clone he_ofl -y
# (then restore torch + install deps — see Phase A notes below; documented in
#  AGENTS/CONTEXT.md and ~/.claude/projects/.../memory/valar.md)

# 1. One-time: vendor the official SALAD repo (needs internet)
scripts/vendor_salad.sh

# 2. One-time per cluster: populate the data cache (SLURM, ~1h)
scripts/submit.sh prefetch                   # default 21000 rows
scripts/submit.sh prefetch 50                # smoke test

# 3. Train a model (depends on prefetch completion) — Phase B
scripts/submit.sh train configs/salad_main.yaml

# 4. Chain prefetch -> train -> eval in one go — Phase B+D
scripts/submit.sh full configs/salad_main.yaml

# 5. Look at a completed run — Phase D
scripts/summarise_run.py runs/2026-04-17_143000_abc1234_salad_main_d4f8/
```

---

## Critical constraints

- **Valar compute nodes have slow internet (≤2 MB/s).** All HuggingFace
  downloads happen in the `prefetch` SLURM job (`t4_ai`, no GPU). Training
  jobs run with `HF_HUB_OFFLINE=1` and read from the local cache only.
- **Official SALAD is vendored, not pip-installed.** See
  `scripts/vendor_salad.sh` and `third_party/salad/SETUP.md`.
- **Conda env on `/scratch`, cloned from `he_ofl`.** Always activate by
  full path: `conda activate /scratch/$USER/conda_envs/visword`. After any
  `pip install`, re-verify `torch.__version__ == '2.3.0+cu121'` —
  `conda create --clone` can swap torch out from under you (see the
  ABI-trap recipe in `AGENTS/CONTEXT.md` / valar.md).
- **No experiment trackers.** JSONL logs and PNG plots only.
- **One run directory per experiment** — every artifact traceable from
  its directory alone.

---

## Layout at a glance

```
src/visword/        # the Python package
  config.py, paths.py, seed.py
  data/             # cropper.py, manifest.py, prefetch.py
  models/           # salad_bridge.py
  interpret/, diagnostics/, reporting/   # Phase B-E placeholders
configs/            # default.yaml, debug.yaml
scripts/            # vendor_salad.sh, prefetch_data.py, resolve_config.py,
                    # _env.sh, submit.sh, summarise_run.py
slurm/              # prefetch.sbatch (live), train.sbatch / eval.sbatch (stubs)
third_party/salad/  # vendored SALAD repo (read-only)
tests/              # pytest — A1-A4 currently green
data/               # prefetched cache (gitignored)
runs/               # experiment outputs (gitignored)
AGENTS/             # CONTEXT.md, PROJECT_SPEC.md, TESTS.md (project docs for agents)
```

See `AGENTS/PROJECT_SPEC.md §2` for the full intended tree (Phases B–E
add `train.py`, `eval_phase{1,2}.py`, `interpret/*.py`, etc.).

---

## Current status — Pre-training & Evaluation complete

The main pipeline (Phases A–E) has been fully implemented, and we have extended the scope to cover visual-only and text-target cross-modal I-JEPA pre-training across various parameter-efficiency scales:

| Step / Phase | Deliverable / Component | Status |
|---|---|---|
| Phase A | Data pipeline, skeleton, SALAD bridge, NonOverlappingCropper, and tests | ✓ |
| Phase B | Model wrappers, batch statistics, InfoNCE / Multi-Similarity losses, and tests | ✓ |
| Phase C | End-to-end training loops (`train.py`), resolved configurations, and integration tests | ✓ |
| Phase D | Evaluation scripts (`eval_phase1.py` and `eval_phase2.py`) with metrics output | ✓ |
| Phase E | Interpretability suite (attention maps, cluster visualisations, CLS-vs-VLAD decomposition) | ✓ |
| Phase F | I-JEPA visual-only & text-target cross-modal pre-training loops, config sweeps (2-blocks, 4-blocks, all-blocks), and tests | ✓ |

**Key Pre-training Runs Completed:**
*   `ijepa-pretrain-2blocks` / `ijepa-pretrain-4blocks` / `ijepa-pretrain-all-blocks`
*   `ijepa-text-target-4blocks` / `ijepa-text-target-all-blocks`

**Local Pytest Status:** All unit and integration tests (including standard I-JEPA, text-target I-JEPA, and text adapters) pass successfully on CPU and GPU devices.

### Spec amendments adopted (operator-confirmed)

1. `.python-version`: `3.11` → `3.9`. `pyproject.toml requires-python: >=3.9`. Rationale: clone `he_ofl` instead of building Python 3.11 + torch from scratch over slow Valar internet.
2. `scripts/_env.sh`: drops `module load python/3.11 cuda/12.1` (spec §10.5 was `TODO(agent)`); adds `module load git/2.9.5` because git is missing from the default PATH on Valar.
3. `slurm/prefetch.sbatch` partition resolved from `TODO_INTERNET_PARTITION` to `t4_ai` without `--gres=gpu` (no GPU consumed).

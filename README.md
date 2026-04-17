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

## Current status — Phase A complete

Per `AGENTS/PROJECT_SPEC.md §15` "Minimum viable delivery":

| Step | Deliverable | Status |
|---|---|---|
| 0 | Conda env at `/scratch/hkanpak21/conda_envs/visword` (clone of `he_ofl`, torch+torchvision restored after the documented ABI trap; +pydantic, pyyaml, huggingface_hub, datasets, pillow, matplotlib, pytest under `--constraint` torch pin) | ✓ |
| 1 | Repo skeleton: `pyproject.toml`, `.gitignore`, `.python-version=3.9`, full directory tree | ✓ |
| 2 | `scripts/vendor_salad.sh` ran → `third_party/salad/` @ `6aede13a` (serizba/salad HEAD on 2026-04-17) | ✓ |
| 3 | `src/visword/models/salad_bridge.py` + `tests/test_salad_bridge.py` | A1 ✓ (3/3) |
| 4 | `src/visword/data/manifest.py` + `tests/test_manifest.py` | A2 ✓ (3/3) |
| 5 | `src/visword/data/cropper.py` + `tests/test_cropper.py` | A3 ✓ (3/3) |
| 6 | `src/visword/{config,paths,seed}.py` + `scripts/resolve_config.py` + `configs/{default,debug}.yaml` + `tests/test_config.py` | A4 ✓ (2/2) |
| 7 | `src/visword/data/prefetch.py` + `scripts/prefetch_data.py`. 5-row local sanity prefetch ran cleanly (PNG valid, fingerprint verified) | ✓ |
| 8 | `slurm/{prefetch,train,eval}.sbatch` + `scripts/{_env.sh,submit.sh,summarise_run.py}`. SLURM smoke job submitted: `scripts/submit.sh prefetch 50` | A5 in flight |

**Local pytest:** `pytest -q` → **11 passed in ~8 s** (CPU only; SALAD bridge forward shape verified on a 768×16×16 dummy tensor).

### Spec amendments adopted (operator-confirmed)

1. `.python-version`: `3.11` → `3.9`. `pyproject.toml requires-python: >=3.9`. Rationale: clone `he_ofl` instead of building Python 3.11 + torch from scratch over slow Valar internet.
2. `scripts/_env.sh`: drops `module load python/3.11 cuda/12.1` (spec §10.5 was `TODO(agent)`); adds `module load git/2.9.5` because git is missing from the default PATH on Valar.
3. `slurm/prefetch.sbatch` partition resolved from `TODO_INTERNET_PARTITION` to `t4_ai` without `--gres=gpu` (no GPU consumed).

# CLAUDE.md — start here (zero-context agent guide)

VISWORD is a research project on **vision-based document retrieval**: can a vision
model retrieve the right Wikipedia page from a screenshot, and *how* does it read the
text in that image? Course project (Koç Univ, COMP547), authors Barış Cem Bakay &
Halil İbrahim Kanpak. Paper title: *"Can Vision Models Read?"*

**Current direction (refined June 2026):** the old setup only let models read
*headlines* (crops were downsampled 2.19× → body text became illegible). The active
goal is to make models perceive **all** the text and answer: *how do vision models
read text, and can they read it efficiently?* See the alignment doc and PRD below.

---

## ⚠ Hard rules — non-negotiable

1. **Never delete or overwrite prior work.** The paper (`paper/`), Barış's code,
   configs, SLURM scripts, and `runs/` results stay intact. Work is **additive** (new
   files, new run dirs). **Any deletion/force-overwrite/reset of prior work needs the
   operator's explicit OK first — ask, then wait.**
2. **No from-scratch training.** Fine-tune *pretrained* encoders only; prefer models
   already trained to read text/documents.
3. **Compute nodes have no internet.** All HuggingFace/GitHub downloads happen on the
   **login node** first; training/eval run with `HF_HUB_OFFLINE=1` against the cache.
4. **No experiment trackers.** JSONL logs + matplotlib PNGs only. One run directory per
   experiment, fully reproducible from its own folder.
5. **Accumulate result numbers, never overwrite them.** Every new measurement is a new
   dated, reproducible row in the results ledger ([`AGENTS/V1/RESULTS.md`](AGENTS/V1/RESULTS.md));
   prior numbers are never edited or deleted, only superseded-and-kept. Same for paper
   tables — add, don't silently replace.

---

## Read these, in order

1. [`AGENTS/CONTEXT.md`](AGENTS/CONTEXT.md) — prior sessions: what was tried, what broke, lessons.
2. [`AGENTS/PROJECT_SPEC.md`](AGENTS/PROJECT_SPEC.md) — authoritative design: layout, config system, SLURM, run-dir contract.
3. [`AGENTS/TESTS.md`](AGENTS/TESTS.md) — acceptance tests per phase.
4. [`AGENTS/TODO.md`](AGENTS/TODO.md) — **the alignment doc**: current direction, every decision (D1–D22), the verified Valar infra/results, and the phased plan. Read this to understand *where we are and why*.
5. [`AGENTS/V1/PRD1.md`](AGENTS/V1/PRD1.md) — the spec for the current work (Phase 1: legible reading + cross-model comparison + an efficient reading model), with its task tickets under [`AGENTS/V1/issues/`](AGENTS/V1/issues/).

---

## PRD / issues flow (extendable convention)

Each unit of work lives in a **versioned folder** under `AGENTS/`:

```
AGENTS/
  V1/                 ← Phase 1 (Wikipedia core)
    PRD1.md           ← the spec (problem, solution, user stories, decisions)
    issues/           ← small, independently-doable task tickets
      issue-01-*.md
      issue-02-*.md
  V2/  (future)       ← Phase 2 (arXiv) — same shape
  V3/  (future)       ← Phase 3 (VQA)
```

To start new work: create `AGENTS/V{n}/PRD{n}.md`, then break it into
`AGENTS/V{n}/issues/issue-*.md`. Each issue is a thin end-to-end slice (data → model →
eval → a number/figure you can verify), with acceptance criteria and "blocked by".

---

## Branches

- `master` = Barış's branch (the I-JEPA / text-target code). **Build on top of it.**
- `main` = the v3 paper state (2 commits behind master).
- The paper on both is identical; the new full-resolution JEPA results are not yet
  written up.

---

## Infra quickfacts (Valar HPC — verified)

- SSH: `ssh valar` (login.valar.ku.edu.tr, user `hkanpak21`). Repo at
  `/scratch/hkanpak21/VISWORD`; conda env `/scratch/hkanpak21/conda_envs/visword`.
- Data: **376k page screenshots, each 980×980**, cached in `data/wiki_ss` (125G); an
  anchor/positive/negative triplet set in `data/wiki_ss_anchors` (train + val splits).
- GPUs (partition `ai`): T4 16GB, **A40 48GB**, V100 32GB, A6000; `avg` has L40S 48GB;
  `kutem_gpu` has A100. Prefer 48GB-class for legible/high-res work.
- **QOS ≈ 1 concurrent GPU job** → jobs run sequentially; design runs to be resumable
  in ≤8h chunks. Login-node `nohup` is disabled (use `setsid`).
- Persistence: `/scratch` can be purged; `/home` is persistent (1.7 PB free). Back up
  data/runs/paper to `/home`. HF model cache is in `/home/hkanpak21/.cache/huggingface`.

---

## How to run (human operator)

```bash
scripts/vendor_salad.sh                 # one-time: vendor the official SALAD repo
scripts/submit.sh prefetch              # cache data (login/internet partition)
scripts/submit.sh train configs/<x>.yaml
scripts/submit.sh full  configs/<x>.yaml  # prefetch -> train -> eval chain
scripts/summarise_run.py runs/<run-dir>/
pytest -q                               # CPU unit tests
```

Config system: `configs/default.yaml` ← named config ← `--set a.b=c` overrides,
resolved by `scripts/resolve_config.py`. The cropper's `crop_size` vs `target_size`
controls downsampling — keep them equal for legible (native-resolution) crops.

---

## What's next (current state)

Grilling is closed; the plan is in `AGENTS/TODO.md` §7 and `AGENTS/V1/`. Order:
**P0** (back up + prefetch new models) → **P1** (legible crops → re-baseline the
cross-model comparison → add document-pretrained models → train the reading model →
perfect-text reference + attention maps + confound control). arXiv (V2) and VQA (V3)
come after P1 shows good results.

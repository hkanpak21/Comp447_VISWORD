# issue-00 — Reconcile repo + back up + prefetch models

**Owner:** US · **Slice:** P0 · **Status:** needs-triage

## What to build
The safe prerequisite for all P1 work, in order, **non-destructively**:
1. **Anchor Barış's unpushed work.** His latest paper commits + an uncommitted figure
   live only on the Valar working copy on a likely-detached HEAD. Put them on a named
   branch and commit the figure so nothing can be lost. **Any git operation (branch,
   commit, push, merge) requires the operator's explicit OK first — never merge/push/
   reset autonomously.**
2. **Back up to persistent storage.** Additively copy `data/` (125G), `runs/` (20G),
   and `paper/` from `/scratch` to `/home` (1.7 PB free). Originals untouched.
3. **Prefetch new HF models** on the login node (compute nodes are offline) into the
   `/home` HF cache: MAE (`facebook/vit-mae-base`, `-large`), Pix2Struct, Donut,
   Nougat, ColPali/ColQwen2. (BERT already cached.)
4. **Verify** the `visword` conda env and a one-job smoke test run green.

## Acceptance criteria
- [ ] Barış's paper commits are on a named branch and the stray figure is committed (after operator OK); nothing deleted/overwritten.
- [ ] `/home` holds verified copies of `data/`, `runs/`, `paper/`; `/scratch` originals intact.
- [ ] All listed models load **offline** (`HF_HUB_OFFLINE=1`) on a compute node.
- [ ] `pytest -q` green; one tiny SLURM job completes.

## Blocked by
- None — this is first. (Git steps gated on operator confirmation.)

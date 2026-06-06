# VISWORD V1 — Orchestrator prompt

> Paste this whole file as the opening prompt of a fresh Claude Code session (or hand it
> to an orchestrating agent) on branch `dev/v1-legible-reading`. It is self-sufficient:
> it tells you what to read, the rules you may never break, how to drive the work with
> subagents, and how to keep the paper continuously in sync with reality.

---

## 0. Your mission

You are the **orchestrator** for VISWORD Phase 1 (V1). Drive the V1 backlog to
completion by deploying subagents, running experiments on the Valar cluster,
accumulating every result, and **continuously updating the paper** (`paper/report_template/visword_report.tex`)
— data, method, results, figures, abstract, conclusions — so the written paper always
reflects what has actually been produced. You coordinate; subagents do the narrow work.

The scientific goal: *how do vision models read text, and can they read it efficiently?*
V1 makes text legible, re-runs the cross-model comparison, and builds **our own reading
model on a pretrained MAE autoencoder** — a lane that does not overlap with Barış's
I-JEPA reader.

---

## 1. First actions (read before doing anything)

Read, in order, and hold them as ground truth:
1. [`/CLAUDE.md`](../../CLAUDE.md) — project, hard rules, how to run, Valar facts.
2. [`AGENTS/TODO.md`](../TODO.md) — the full alignment record and decisions **D1–D26**
   (esp. D11/D17 comparability, D13 page-level eval, D18/D25/D26 the Barış update +
   division of labor, D19 native-res, D21–D24 preservation + accumulate-numbers).
3. [`AGENTS/V1/README.md`](README.md) — the plain one-pager.
4. [`AGENTS/V1/PRD1.md`](PRD1.md) — the spec (problem, user stories, modules, tests).
5. [`AGENTS/V1/issues/README.md`](issues/README.md) + the 12 tickets — your backlog.
6. [`AGENTS/V1/RESULTS.md`](RESULTS.md) — the append-only results ledger.

Then verify the live state yourself (don't trust stale notes): `git status`, current
branch, `ssh valar` reachable, and re-read the relevant source files before editing them.

---

## 2. Non-negotiable rules (a subagent violating any of these is a failure)

1. **Preservation.** Never delete or overwrite prior work — Barış's code, configs,
   SLURM scripts, `runs/`, or the paper's existing content. **Work is additive.** Any
   deletion, force-overwrite, `git reset --hard`, force-push, or `rm -rf` of prior work
   **stops and asks the operator first.**
2. **Stay in our lane.** Tickets tagged `[BARIŞ]` (10, 11 — the **I-JEPA** reader) are
   his; do **not** execute them or touch his checkpoint (`/scratch/bbakay22/…`, which is
   permission-locked anyway). Monitor them only to integrate his numbers into the paper
   when they land.
3. **No from-scratch training.** Fine-tune pretrained encoders only.
4. **Compute nodes are offline.** Prefetch every model/dataset on the **login node**
   first; run training/eval with `HF_HUB_OFFLINE=1`.
5. **Accumulate numbers, never overwrite.** Every result is a new dated, reproducible
   row in [`RESULTS.md`](RESULTS.md); keep superseded rows. Same discipline in the paper:
   fill TBDs and add tables; never silently delete an existing finding.
6. **No experiment trackers.** JSONL logs + matplotlib PNGs; one run dir per experiment,
   reproducible from its own folder (config + provenance + metrics).
7. **Every claim is verified.** A ticket is done only when its acceptance criteria are
   met, its tests are green, and a second (verification) subagent has confirmed it.

---

## 3. Environment (Valar)

- `ssh valar` → repo `/scratch/hkanpak21/VISWORD`, conda env
  `/scratch/hkanpak21/conda_envs/visword`. Operator = `hkanpak21`.
- Data: 376k Wikipedia screenshots, each **980×980**, in `data/wiki_ss` (125G); anchors
  in `data/wiki_ss_anchors`. Persist backups to `/home` (1.7 PB free); `/scratch` can be purged.
- GPUs: prefer **A40 / L40S / A100 (48GB)** over T4 for legible/high-res work. QOS ≈ **1
  concurrent GPU job** → jobs run sequentially; make runs **resumable in ≤8h chunks**.
  Login-node `nohup` is disabled — use `setsid` for detached login-node work.
- Submit via `scripts/submit.sh`; summarise via `scripts/summarise_run.py`. Config:
  `configs/default.yaml` ← named config ← `--set a.b=c`. **Native-resolution crops mean
  `crop_size == target_size` (no downsample).**

---

## 4. Backlog and execution order (our lane)

Dependency DAG (from [issues/README.md](issues/README.md)):

```
00 reconcile+backup+prefetch
   └─ 01 legible cropping
        ├─ 02 re-baseline grid (+MAE) ── 03 doc family
        │                              ├─ 05 perfect-text bound
        │                              ├─ 06 attention
        │                              └─ 08 global token (optional)
        └─ 04 MAE reader (body target) ── 07 confound control
                                        └─ 09 high-freq AE arm (optional)
[BARIŞ] 10, 11 run in parallel — monitor, integrate, do not execute.
```

Do **00 first** (its git steps need operator OK). Then 01. Then fan out: 02→{03,05,06,08}
and 04→{07,09} can progress independently. Respect blocked-by. Optional tickets (08, 09)
last.

---

## 5. How to execute one ticket (the unit loop)

For each ready ticket:
1. **Scope** — re-read the ticket + the relevant source. Spawn an **Explore** subagent if
   you need to locate code/conventions across the repo; take its conclusion, not file dumps.
2. **Implement test-first** — for the four tested modules (cropping, model-wrapper,
   scoring, reader-run-finishes), write the test before the implementation; tests check
   external behavior through public interfaces, mirroring `tests/test_cropper.py` and
   `tests/test_ijepa_text_pretrain.py`. Vertical slices: one thin path end-to-end, not a
   whole layer at once.
3. **Run on Valar** — submit via `scripts/submit.sh`, on a 48GB GPU, resumable; poll the
   job; pull metrics. Never run heavy work on the login node.
4. **Verify** — spawn a separate **verification subagent** to check the acceptance
   criteria against the actual outputs (run dir, metrics, plots) and try to *refute* the
   result. Only accept if it survives.
5. **Record** — append the number(s) to [`RESULTS.md`](RESULTS.md) (dated, with run-dir +
   git SHA), then **update the paper** (§7).
6. **Commit** — one atomic commit per slice on `dev/v1-legible-reading`, message stating
   what shipped + the number. Push the branch.

---

## 6. Subagent playbook

Deploy subagents for parallelism and independent verification; you stay the integrator.
- **Explore subagent** — read-only fan-out search to locate code / naming / conventions.
- **Implementation subagent** — one per ticket (or per module within a ticket); give it
  the ticket text, the rules in §2, and the exact files it may touch. Worktree isolation
  if two implementation subagents would edit overlapping files in parallel.
- **Verification subagent** — adversarial; given a finished slice, tries to break it
  against the acceptance criteria. Majority-refute ⇒ reject and reopen.
- **Paper-writer subagent** — given a verified result + the numbers, edits the specific
  paper section/table (§7) additively.
- Run independent tickets' subagents concurrently; serialize anything that shares the
  single GPU slot (QOS=1). When several independent pieces are ready, dispatch them in
  one batch. (If the operator has opted into a Workflow/ultracode harness, structure the
  find→implement→verify→write loop as a pipeline; otherwise use the Agent/Task tool.)

---

## 7. The paper-update loop (do this after every verified result — the core requirement)

Keep `paper/report_template/visword_report.tex` continuously in sync with reality. After
each ticket lands, **directly update the paper** — additively, never deleting Barış's
existing content:

- **Data (§Data).** Document the legible rebuild: native-resolution crops (no 2.19×
  downsample), 980×980 source, text-aware cropping heuristics, the new disjoint eval
  slice, dataset sizes.
- **Method (§Method).** Add/extend: the cropping change; the shared model-wrapper +
  cross-model grid now including **MAE** and the **document-pretrained family**
  (Pix2Struct/Donut/Nougat/ColPali); **page-level same-page re-identification** as the
  retrieval protocol; **our MAE reader** (body-text target, parameter-efficient
  fine-tune) and how it contrasts with Barış's I-JEPA reader (pixel-reconstruction vs
  feature-prediction); the perfect-text upper bound; the high-freq-AE arm if run.
- **Results (§Results).** Fill in TBD rows; **add** new tables/rows for the
  legible-resolution grid, the document family, the MAE reader before/after, the
  perfect-text ceiling, the confound-control check. Numbers must match RESULTS.md.
- **Figures.** Add attention "where it reads" heatmaps; update the comparison
  scatter/tables. Put new figures in `paper/report_template/figures/`.
- **Abstract / Intro / Conclusion.** Update the headline claims to the legible-resolution
  story (reading is no longer heading-only) and the MAE-reader result, once they exist.
- **Build.** Run `paper/report_template/build.sh` and confirm the PDF compiles after each
  paper change; fix LaTeX errors before committing.
- **Discipline.** Do not overwrite Barış's results — accumulate. When a number is
  superseded, keep the prior one (table footnote or a kept row) per the accumulate rule.
  Keep the colour-coded encoder families and existing notation. Commit each paper update
  with a message naming the section and the result.

Treat the paper as a living artifact: at any point in V1 it should be buildable and
truthfully reflect every completed experiment.

---

## 8. When to STOP and ask the operator (do not proceed autonomously)

- Any **git history rewrite / push to `master` / merge / reset / force-push**, and the
  **D23 divergence reconciliation** (Barış's possibly-detached Valar paper commits).
- Anything that would **delete or overwrite** prior work, results, or paper content.
- Touching **Barış's lane** (I-JEPA reader / his checkpoint) or his scratch.
- Adding a **dependency** to `pyproject.toml`, changing the **config schema**, or editing
  `third_party/`.
- A result that **contradicts** a published paper claim (surface it, don't quietly bury it).
- Large/expensive compute beyond routine fine-tunes.

---

## 9. Coordination with Barış's lane

Tickets 10/11 (I-JEPA reader heads; optional body-target) are Barış's. He reads this repo.
Do not run them. When his numbers appear (RESULTS.md TBD rows, or paper TBDs), integrate
them into the comparison and the paper — clearly attributed — so the final story sets our
**MAE** reader beside his **I-JEPA** reader on the same legible-resolution, page-level task.

---

## 10. Definition of done (V1)

- Tickets 00–07 green (08, 09 optional); each verified; numbers in RESULTS.md with
  provenance.
- The comparison grid re-run at legible resolution, with MAE and the document family.
- Our MAE reader trained (body target), evaluated at page level, before/after reported,
  the title-confound shown controlled.
- The **paper builds** and its Data/Method/Results/Figures/Abstract/Conclusion reflect
  all of the above, additively, with Barış's content intact and his I-JEPA results
  integrated when available.
- All work committed on `dev/v1-legible-reading` and pushed; nothing destructive done
  without operator sign-off.

Begin by reading §1, then propose your first move (almost certainly ticket 00, pausing
for operator OK before any git operation).

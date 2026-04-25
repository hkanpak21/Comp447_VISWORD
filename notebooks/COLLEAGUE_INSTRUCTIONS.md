# VISWORD experiments — colleague onboarding

## What this project is doing (1 paragraph)

We compare image encoders (CLIP, DINOv2, SigLIP, I-JEPA, plain ViT) on
a Wikipedia-screenshot retrieval task to test the Platonic
Representation Hypothesis. The cluster is running the full grid
already; **we're using your compute to speed up the *head-ablation*
rows of the same grid.** These rows tell us whether the fancy SALAD
aggregator beats a plain MLP head — useful but non-headline: if your
runs fail, the cluster will produce the same results in ~16 hours.
**If they succeed, we save ~6 hours of cluster queue time and can
finish the paper sooner.** No pressure either way.

## Two paths — pick whichever fits

| Path | Who | Compute | Setup time |
|---|---|---|---:|
| **A. Colab** | Anyone with a Google account | Your Colab GPU credits (T4 free / A100 Pro) | ~5 min |
| **B. Cluster** | You if you have your own SLURM account on Valar | A second T4 (the per-user 1-T4 cap is per account, so your jobs run on a different GPU than mine) | ~10 min |

Path B is faster (a separate T4 not contending with my queue) but
needs cluster access. You confirmed read access to
`/scratch/hkanpak21/VISWORD` — if you also have SLURM submit
privileges on the `t4_ai` partition, **prefer Path B**.

## Your assignments (in priority order)

You can stop after any step. Each is independent.

| # | What to run | Why | Time on T4 | Time on A100 |
|---|---|---|---:|---:|
| 1 | `EXPERIMENT = 'finetune_dinov2_mlp'` | MLP-head ablation row for DINOv2 | ~3.5h | ~1.3h |
| 2 | `EXPERIMENT = 'finetune_clip_mlp'` | MLP-head ablation row for CLIP | ~3h | ~1.2h |
| 3 | `EXPERIMENT = 'zeroshot_grid'` | Cross-validates our zero-shot results + fills the SigLIP cell that disconnected on the first session | ~2.5h | ~1h |

**Recommended order**: do step 1 first by itself. If it succeeds (you
see `--- DONE ---` and JSON files in your Drive), open a *second*
Colab tab and start step 2 in parallel. Step 3 is bonus.

The most important thing: **if step 1 fails for any reason, stop and
ping back the error message** — we'd rather debug once than have you
burn three sessions hitting the same bug.

## Setup (one-time, ~5 minutes)

1. Open this URL in a fresh tab:
   <https://colab.research.google.com/github/hkanpak21/Comp447_VISWORD/blob/main/notebooks/visword_colab.ipynb>

2. **Enable GPU**: top menu → Runtime → Change runtime type → Hardware
   accelerator → **T4** (free) or **A100** (Pro). Save.

3. **Add Hugging Face token** (so the notebook can download the
   wiki-ss dataset):
   - Go to <https://huggingface.co/settings/tokens> and create a
     **read** token. Copy it.
   - In Colab, click the **🔑 key icon** in the left sidebar →
     "**Add new secret**" → name `HF_TOKEN`, paste the token, enable
     "**Notebook access**".

That's all the setup you need. The notebook handles dependencies, repo
clone, SALAD vendoring, etc.

## Running an assignment

1. Run cells **1 through 4** (GPU check, repo clone, dependencies, HF
   token + Drive mount). When Drive asks for permission, allow it.

2. In **cell 5** (the "Pick the experiment" cell), edit the line:
   ```python
   EXPERIMENT = 'zeroshot_grid'
   ```
   to your assignment, e.g.:
   ```python
   EXPERIMENT = 'finetune_dinov2_mlp'
   ```
   Run cell 5.

3. Run cell 6 (data download). Takes 30–45 minutes the first time on
   T4 (it's bandwidth-bound — there's nothing wrong if it sits at
   "downloading" for 30+ min). The download is cached in your runtime,
   so re-running this cell after a restart is fast as long as the
   runtime is the same.

4. Run cell 7 (the actual experiment). This is the long one
   — see the time estimates above. **Stay on the page or use Colab
   Pro** to avoid disconnects. The cell streams progress to the
   notebook output so you can see it's still alive.

5. When the cell prints `--- DONE in X.X min ---`, you're done.
   Results are in `MyDrive/visword_results/<EXPERIMENT>/` —
   verify by running cell 8.

## Returning results

After your run finishes:

1. In Google Drive, find the folder `MyDrive/visword_results/`
2. Right-click → **Share** → "Anyone with the link" → "Viewer"
3. Copy the link, send it to me.

That's all I need. I'll pull the JSONs and run them through the
results aggregator.

## If it fails

**Common failures and fixes**:

- **`fatal: ... third_party/salad not found`** — the training cell
  should auto-vendor it now. If you still see this, run a fresh
  cell with: `!cd /content/visword && bash scripts/vendor_salad.sh`
  then re-run the training cell.

- **`HF_TOKEN not set`** — go back to step 3 of Setup.

- **`No CUDA device`** or **GPU OOM** — Runtime → Disconnect and
  delete runtime → Change runtime type → make sure it's GPU → reconnect.

- **Disconnected mid-run** — the data download is cached so just
  re-run cells 6 and 7. The training itself starts over (we don't
  checkpoint on Colab), so favour A100 if available.

- **Anything weirder** — copy the FULL error message and send it.
  Don't try to fix it yourself unless you've seen it before; one
  failure pinged is much cheaper than three.

## What you should NOT touch

- Don't change `NUM_TRAIN_PAGES` or `NUM_EVAL_PAGES` in cell 5 — those
  must stay at 30000 / 2000 for the results to be comparable with the
  rest of the grid.
- Don't change the configs in `configs/grid_*_30k.yaml`. They're set up
  to match the cluster's training recipe exactly.

---

# Path B — running on the Valar cluster (recommended if you have SLURM access)

## One-time setup (~10 minutes)

1. SSH into the cluster as your own user:
   ```bash
   ssh <your-username>@<valar-host>
   ```

2. Make a working directory under your scratch space and clone the
   repo (or just copy the source from the shared cluster path —
   either works):
   ```bash
   mkdir -p /scratch/<your-username>/visword_runs
   cd /scratch/<your-username>
   git clone https://github.com/hkanpak21/Comp447_VISWORD.git visword
   cd visword
   bash scripts/vendor_salad.sh  # ~10 sec
   ```

3. Set up a Python env. Easiest: use the same Conda env I built:
   ```bash
   # Either use the pre-built env (read-only, just activate):
   source /scratch/hkanpak21/conda_envs/visword/bin/activate

   # OR build your own (~10 min):
   module load conda
   conda create -p /scratch/<your-username>/conda_envs/visword python=3.9
   conda activate /scratch/<your-username>/conda_envs/visword
   cd /scratch/<your-username>/visword
   pip install -e .
   pip install open_clip_torch timm sentence-transformers \
       imagehash sentencepiece protobuf
   ```

4. Point at the **already-downloaded data cache** in my scratch
   (read-only). No need to re-prefetch the 100GB+ wiki-ss cache:
   ```bash
   # The configs/grid_*_30k.yaml files reference data.wiki_ss_cache_dir;
   # we override it on the sbatch command line to point at my shared cache.
   ```

## What to run

I added `slurm/train_user.sbatch` and `slurm/eval_full_user.sbatch`
specifically for you — they accept all paths via env vars and don't
hardcode my own QoS / scratch path. Set the placeholders
`<USER>`, `<QOS>`, and `<ACCOUNT>` to your values.

**Assignment 1 — DINOv2 + MLP fine-tune (~4h walltime budget):**
```bash
cd /scratch/<USER>/visword
mkdir -p _slurm runs
sbatch \
  --partition=t4_ai \
  --time=07:00:00 \
  --gres=gpu:tesla_t4:1 \
  --mem=32G --cpus-per-task=4 \
  --qos=<QOS> --account=<ACCOUNT> \
  --output=/scratch/<USER>/visword/_slurm/train_%j.out \
  --error=/scratch/<USER>/visword/_slurm/train_%j.err \
  --export=ALL,CONFIG=configs/grid_dinov2_mlp_30k.yaml,REPO_ROOT=/scratch/<USER>/visword,DATA_CACHE=/scratch/hkanpak21/VISWORD/data/wiki_ss,RUNS_ROOT=/scratch/<USER>/visword/runs \
  slurm/train_user.sbatch
```
(If you don't know your QoS / account, omit those two `--qos`/
`--account` flags — SLURM will use partition defaults.)

Watch progress:
```bash
squeue -u <USER>
tail -f /scratch/<USER>/visword/_slurm/train_<JOB_ID>.out
```

**Assignment 2 — CLIP + MLP fine-tune (~3.5h):** same command, swap
`configs/grid_dinov2_mlp_30k.yaml` → `configs/grid_clip_mlp_30k.yaml`.

**Assignment 3 — eval the trained checkpoints:** after each train
job finishes, submit an eval that runs Protocol-A, title-blanked
Phase-2, and the legacy Phase-1/2:
```bash
RUN_DIR=$(ls -td /scratch/<USER>/visword/runs/*grid-dinov2-mlp* | head -1)
sbatch \
  --partition=t4_ai \
  --time=02:00:00 \
  --gres=gpu:tesla_t4:1 \
  --mem=24G --cpus-per-task=4 \
  --output=/scratch/<USER>/visword/_slurm/eval_%j.out \
  --error=/scratch/<USER>/visword/_slurm/eval_%j.err \
  --export=ALL,RUN_DIR=$RUN_DIR,REPO_ROOT=/scratch/<USER>/visword \
  slurm/eval_full_user.sbatch
```
This produces `phase1_recall.json`, `phase1_holdout.json`
(Protocol-A — the headline), `phase2_recall.json`, and
`phase2_titleblanked_15.json` in the run dir.

You can chain a train and its eval with `--dependency=afterany`:
```bash
TRAIN_JID=$(sbatch --parsable ... slurm/train_user.sbatch)
sbatch --dependency=afterany:$TRAIN_JID ... slurm/eval_full_user.sbatch
```

## Returning results

The simplest path: **just leave the run directories under
`/scratch/<your-username>/visword_runs/`**. I have read access (or you
can `chmod -R g+r` to make sure). Tell me the path of each
finished run dir.

If you'd prefer to send only the JSONs (no checkpoints), one-liner:
```bash
cd /scratch/<your-username>/visword_runs
tar czf /scratch/hkanpak21/VISWORD/share/colleague_results.tar.gz \
    */phase*.json */config.resolved.yaml
```

## If a SLURM job fails

- **`DependencyNeverSatisfied`**: a previous job in your chain failed.
  Check the train log: `cat _slurm/train_<JOB>.err`.
- **GPU OOM**: the SALAD-full descriptor is heavy. Try MLP first.
- **Walltime exceeded**: the 7-hour cap should be enough for 30k×3ep.
  If you hit it, the model checkpoint is in
  `<run-dir>/checkpoints/best_phase1.pt` — eval still works on it.
- **Anything else**: paste the SLURM `.err` file content and I'll
  triage.

---

## Reference

If you want to know what's in this project before running:

- README on GitHub: <https://github.com/hkanpak21/Comp447_VISWORD>
- The Colab notebook itself has inline markdown explaining each cell.
- Training spec, encoder grid, and Protocol-A definition are in
  `paper/report_template/visword_report.tex` (build the PDF via
  `paper/report_template/build.sh` for the readable version).

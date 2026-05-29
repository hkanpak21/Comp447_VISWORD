# VISWORD notebooks

## visword_colab.ipynb

A single Colab notebook that runs any one of five experiments per session.
Open in multiple Colab tabs, set `EXPERIMENT` differently in each, and
they run in parallel without colliding (results land in a unique
sub-folder under `MyDrive/visword_results/<EXPERIMENT>/`).

Available experiments:

| `EXPERIMENT` | What it runs | T4 wall-time | A100 wall-time |
|---|---|---:|---:|
| `zeroshot_grid` | 6 image + 4 text encoders → Protocol-A retrieval per image encoder + the full Platonic alignment grid | ~2.5h | ~1h |
| `finetune_dinov2_salad` | DINOv2 + SALAD on 30k pages × 3ep + Protocol-A + title-blanked Phase-2 | ~4h | ~1.5h |
| `finetune_dinov2_mlp` | DINOv2 + MLP (CLS) head | ~3.5h | ~1.3h |
| `finetune_clip_salad` | CLIP + SALAD with `lr_bb=1e-7` | ~3.5h | ~1.3h |
| `finetune_clip_mlp` | CLIP + MLP head | ~3h | ~1.2h |

## Before running on Colab — push the latest code to GitHub

The notebook clones the repo from GitHub. The GitHub remote
(`https://github.com/hkanpak21/Comp447_VISWORD`) is currently behind
the cluster's local `master` by **33 commits** including the new
`grid_*.yaml` configs, `eval_phase1_holdout.py`, `eval_phase2_titleblanked.py`,
`scripts/zeroshot_protocol_a.py`, and the new SigLIP / I-JEPA / plain-ViT
zero-shot encoder wrappers — all of which the notebook depends on.

### Option A — push from your laptop (preferred, ~5 min)

From a machine where you have GitHub credentials configured:

```bash
# 1. Pull the cluster's master into a local clone (uses your SSH key to the cluster)
git clone https://github.com/hkanpak21/Comp447_VISWORD.git
cd Comp447_VISWORD
git remote add cluster ssh://<your-username>@<cluster-host>/scratch/<your-username>/VISWORD
git fetch cluster master

# 2. Fast-forward main to cluster/master (or merge if you have local edits on main)
#    The histories diverged from `b454ffb init`; cluster/master is far ahead
#    on a different lineage, so this is a force replace, NOT a merge:
git checkout main
git reset --hard cluster/master
git push --force-with-lease origin main
```

After that, the Colab notebook clones the up-to-date repo and the
experiments work end-to-end.

### Option B — git bundle (if the cluster isn't reachable from your laptop)

I created `share/visword_local.bundle` (~530 KB) on the cluster that
contains all 33 commits ahead of GitHub. Download it via VS Code,
SCP, or whatever you use, then locally:

```bash
git clone https://github.com/hkanpak21/Comp447_VISWORD.git
cd Comp447_VISWORD
git fetch /path/to/visword_local.bundle 'refs/heads/master:cluster-master'
git checkout main
git reset --hard cluster-master
git push --force-with-lease origin main
```

### Option C — work around without pushing (Drive-bundle path)

Upload the cluster's `share/visword_local.bundle` to your Drive folder
`MyDrive/visword_bundles/`, then **uncomment the bundle-fetch cell** in
the notebook (cell 5b — see comment "BUNDLE FALLBACK"). The notebook
will clone GitHub's main, then patch in the cluster's commits via the
bundle. Works without ever pushing to GitHub.

## Once GitHub is synced — running on Colab

1. Open `notebooks/visword_colab.ipynb` in Colab (drop the file into
   <https://colab.research.google.com> or Colab → File → Open notebook → GitHub
   → select repo `hkanpak21/Comp447_VISWORD` → branch `main` → notebook
   `notebooks/visword_colab.ipynb`).
2. Runtime → Change runtime type → choose **T4** (free) or **A100** (Pro/Pro+).
3. Run cells 1–4 (setup).
4. In cell 5, set `EXPERIMENT`. Run.
5. To run multiple experiments **in parallel**: open the same notebook
   in 2–4 more Colab tabs (each gets its own GPU under your quota), set
   each tab to a different `EXPERIMENT`, run.
6. Each session writes to `MyDrive/visword_results/<EXPERIMENT>/`. Share
   that folder back via Drive's "Share with anyone" link, paste the URL
   in chat, and I'll pick up the JSON results and plug them into the
   paper tables.

## What's in the build-helper script

`build_colab_notebook.py` constructs the `.ipynb` JSON from Python (cells
defined as a clean Python list rather than hand-edited JSON). To
regenerate the notebook after editing the script:

```bash
python notebooks/build_colab_notebook.py
```

## Recommended split for the 72h demo

If you have 4 Colab Pro sessions available (one A100 each), the most
useful parallel split is:

| Tab | EXPERIMENT | Output |
|---|---|---|
| 1 | `zeroshot_grid` | the central retrieval table + Platonic heatmap |
| 2 | `finetune_clip_salad` | the highest-priority fine-tune row (CLIP is the H-Platonic anchor) |
| 3 | `finetune_dinov2_salad` | DINOv2-SALAD baseline at 30k for the grid |
| 4 | `finetune_dinov2_mlp` | DINOv2-MLP for the head-ablation slice |

That covers all of Track A + Track C in ~2h wall-clock. The cluster
queue (Track A + B + C) stays as a backup and produces the saturation
curve.

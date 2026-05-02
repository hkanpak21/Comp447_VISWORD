"""Side-by-side example of an original vs title-blanked Wikipedia
page screenshot. Single appendix figure for the title-region ablation."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from visword.data import manifest as M

ROOT = Path("/scratch/hkanpak21/VISWORD")
CACHE = ROOT / "data/wiki_ss"
OUT = ROOT / "paper/report_template/figures/blank_example.pdf"

man = M.read_manifest(CACHE)
# Pick a page with a clear title region. Use the seed-42 first eval page.
import numpy as np
rng = np.random.default_rng(42)
n_train = 32000
perm = rng.permutation(man["num_rows"])
eval_idx = perm[n_train: n_train + 5].tolist()

row = man["rows"][eval_idx[0]]
img = Image.open(CACHE / row["image_path"]).convert("RGB")

# Cap rendering size.
max_dim = 600
scale = min(1.0, max_dim / max(img.size))
img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)),
                 resample=Image.BILINEAR)

W, H = img.size
blank15 = img.copy()
ImageDraw.Draw(blank15).rectangle((0, 0, W, int(0.15 * H)),
                                  fill=(255, 255, 255))

fig, axes = plt.subplots(1, 2, figsize=(5.2, 4.8))
axes[0].imshow(img); axes[0].axis("off")
axes[0].set_title(f"original\n{row['title'][:50]}", fontsize=9)
axes[1].imshow(blank15); axes[1].axis("off")
axes[1].set_title("top-15\\% painted white\n(blank-15)", fontsize=9)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", dpi=130)
print(f"wrote {OUT}")

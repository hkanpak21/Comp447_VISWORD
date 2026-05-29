#!/usr/bin/env python3
"""Build the central paper figure: Platonic alignment vs Protocol-A R@10.

The Platonic Representation Hypothesis predicts that an image
encoder's alignment with text encoders should rank-correlate with its
downstream cross-modal retrieval performance. We test this on the
six ViT-B (and one ViT-H/14 for I-JEPA) encoders by plotting:

    x = max over text-encoders of mutual-kNN@10 (image_enc, text_enc)
    y = Protocol-A R@10 zero-shot (the same image encoder, P=2000)

A positive Spearman correlation supports H-Platonic.

Outputs:
    paper/report_template/figures/platonic_vs_retrieval.pdf
    paper/report_template/figures/platonic_vs_retrieval.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import os
import getpass
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

def get_path(rel_path: str) -> Path:
    local_path = ROOT / rel_path
    if local_path.exists():
        return local_path
    
    # Resolve via env variables
    shared_root = os.environ.get("SHARED_PROJECT_ROOT")
    if shared_root:
        shared_path = Path(shared_root) / rel_path
        if shared_path.exists():
            return shared_path
            
    # Fallback to current user's scratch space dynamically
    user = os.environ.get("USER") or getpass.getuser()
    user_path = Path(f"/scratch/{user}/VISWORD") / rel_path
    if user_path.exists():
        return user_path
        
    return local_path

PLAT = get_path("runs/platonic_alignment_2026-04-26_070508/report.json")
ZS_DIR = get_path("runs/_zeroshot")
OUT_DIR = ROOT / "paper/report_template/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Encoder display labels + JSON file basenames.
# Pastel colours coded by pretraining family (consistent with paper tables):
#   image-text contrastive  -> light blue
#   image-only SSL          -> light green
#   supervised classification-> light salmon
#   random init             -> light grey
encoders = {
    "clip_image":   {"label": "CLIP",          "color": "#7BAFD4", "marker": "o"},
    "siglip_image": {"label": "SigLIP",        "color": "#3F7AB1", "marker": "o"},
    "dinov2":       {"label": "DINOv2",        "color": "#9EC795", "marker": "s", "zs_alias": "dinov2_cls"},
    "ijepa":        {"label": "I-JEPA",        "color": "#5C9C52", "marker": "s"},
    "imagenet_vit": {"label": "ImageNet ViT",  "color": "#E8A29A", "marker": "^"},
    "plain_vit":    {"label": "Plain ViT (random)", "color": "#B0B0B0", "marker": "D"},
}
text_encoders = ["bert", "minilm", "clip_text", "siglip_text"]


def main() -> None:
    plat = json.loads(PLAT.read_text())["pairs"]

    xs, ys, labels, colors = [], [], [], []
    for ie, meta in encoders.items():
        # max alignment over the four text encoders
        align_max = 0.0
        align_argmax = ""
        for te in text_encoders:
            v = plat.get(f"{ie}__{te}") or plat.get(f"{te}__{ie}")
            if v is None:
                continue
            knn = v["mutual_knn_10"]
            if knn > align_max:
                align_max = knn
                align_argmax = te

        # Protocol-A R@10 from the zero-shot JSON
        zs_alias = meta.get("zs_alias", ie)
        zs_path = ZS_DIR / f"{zs_alias}_protocolA_n2000.json"
        if not zs_path.exists():
            print(f"WARN: missing {zs_path}, skipping {ie}")
            continue
        zs = json.loads(zs_path.read_text())
        r10 = zs["recall"]["10"]

        xs.append(align_max)
        ys.append(r10)
        labels.append(meta["label"])
        colors.append(meta["color"])
        print(f"  {meta['label']:20s}  align(vs {align_argmax:11s}) = {align_max:.3f}  R@10 = {r10:.3f}")

    # Spearman correlation
    rho, p = spearmanr(xs, ys)
    print(f"\nSpearman rho = {rho:.3f} (p = {p:.4f}) over n = {len(xs)} encoders")

    markers = [encoders[k]["marker"] for k in encoders if (ZS_DIR / f"{encoders[k].get('zs_alias', k)}_protocolA_n2000.json").exists()]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for x, y, c, m, l in zip(xs, ys, colors, markers, labels):
        ax.scatter(x, y, c=c, s=130, edgecolors="#333333", linewidths=0.6,
                   marker=m, zorder=3, label=l)

    # Inline labels with manual position tweaks for readability
    label_offsets = {
        "CLIP":             (0.006, 0.020, "left"),
        "SigLIP":           (0.006, 0.020, "left"),
        "DINOv2":           (0.0, 0.060, "center"),
        "ImageNet ViT":     (0.012, 0.005, "left"),
        "I-JEPA":           (0.0, -0.080, "center"),
        "Plain ViT (random)": (-0.002, 0.060, "right"),
    }
    for x, y, l in zip(xs, ys, labels):
        dx, dy, ha = label_offsets.get(l, (0.006, 0.018, "left"))
        ax.annotate(l, (x, y), xytext=(x + dx, y + dy),
                    fontsize=8.5, ha=ha)

    ax.set_xlabel("Max mutual-kNN@10 vs. a text encoder", fontsize=10)
    ax.set_ylabel("Protocol-A R@10 (zero-shot, P=2000)", fontsize=10)
    ax.set_title("Spearman $\\rho = %.2f$ ($p = %.2f$, $n = %d$)"
                 % (rho, p, len(xs)), fontsize=10)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_xlim(-0.012, max(xs) * 1.30)
    ax.set_ylim(-0.10, 1.05)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    pdf = OUT_DIR / "platonic_vs_retrieval.pdf"
    png = OUT_DIR / "platonic_vs_retrieval.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"\nwrote {pdf} and {png}")


if __name__ == "__main__":
    main()

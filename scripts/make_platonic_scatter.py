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
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path("/scratch/hkanpak21/VISWORD")
PLAT = ROOT / "runs/platonic_alignment_2026-04-26_070508/report.json"
ZS_DIR = ROOT / "runs/_zeroshot"
OUT_DIR = ROOT / "paper/report_template/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Encoder display labels + JSON file basenames.
encoders = {
    "clip_image":   {"label": "CLIP",          "color": "#1f77b4"},
    "siglip_image": {"label": "SigLIP",        "color": "#ff7f0e"},
    "dinov2":       {"label": "DINOv2",        "color": "#2ca02c", "zs_alias": "dinov2_cls"},
    "imagenet_vit": {"label": "ImageNet ViT",  "color": "#d62728"},
    "ijepa":        {"label": "I-JEPA",        "color": "#9467bd"},
    "plain_vit":    {"label": "Plain ViT (random)", "color": "#7f7f7f"},
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

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.scatter(xs, ys, c=colors, s=110, edgecolors="black", linewidths=0.7, zorder=3)
    for x, y, l in zip(xs, ys, labels):
        # tweak text positions to avoid overlaps
        dx, dy = 0.005, 0.012
        if l == "DINOv2":
            dx, dy = 0.005, -0.04
        elif l == "Plain ViT (random)":
            dx, dy = -0.005, -0.04
        elif l == "I-JEPA":
            dx, dy = 0.005, 0.025
        ax.annotate(l, (x, y), xytext=(x + dx, y + dy), fontsize=9)

    ax.set_xlabel("Max mutual-$k$NN@10 over text encoders\n(image enc.\\ vs.\\ \\{BERT, MiniLM, CLIP-text, SigLIP-text\\})")
    ax.set_ylabel("Protocol-A R@10 zero-shot ($P{=}2000$)")
    ax.set_title(f"Platonic alignment predicts retrieval ($\\rho = {rho:.2f}$)")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.005, max(xs) * 1.2)
    ax.set_ylim(-0.05, 1.0)

    fig.tight_layout()
    pdf = OUT_DIR / "platonic_vs_retrieval.pdf"
    png = OUT_DIR / "platonic_vs_retrieval.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"\nwrote {pdf} and {png}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate Track-A / Track-B / Track-C outputs into the demo deliverables.

Run this after some / all of the queued jobs finish. Idempotent: missing
inputs become "-" rows in the tables. Designed for the 72h demo plan.

Outputs (under runs/_demo/):
  retrieval_table.md        — Protocol-A R@1/5/10/20 across encoders × (zero-shot, MLP, SALAD)
  titleblanked_table.md     — original vs blanked Phase-2 R@1 per encoder
  saturation_curve.csv      — Protocol-A R@10 vs training step on DINOv2-SALAD-50k
                              (falls back to legacy Phase-1 R@10 if Protocol-A wasn't logged in-loop)
  platonic_grid.json        — full encoder × text grid alignment scores
  scatter_alignment_vs_retrieval.csv — Platonic alignment vs Protocol-A R@10 per encoder

Usage::

    PYTHONPATH=src python -m scripts.aggregate_demo_results
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/scratch/hkanpak21/VISWORD")
RUNS = ROOT / "runs"
DEMO = RUNS / "_demo"
ZS = RUNS / "_zeroshot"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _find_latest_run(pattern: str) -> Path | None:
    matches = sorted(RUNS.glob(pattern), key=lambda p: p.name, reverse=True)
    for p in matches:
        if p.is_dir() and (p / "config.resolved.yaml").exists():
            return p
    return None


def retrieval_table() -> str:
    """Build the headline retrieval table.

    Columns: encoder | zero-shot R@1/5/10/20 | MLP fine-tune R@10 | SALAD fine-tune R@10
    """
    encoders_zs = ["dinov2_cls", "clip_image", "siglip_image",
                   "imagenet_vit", "plain_vit", "ijepa"]

    # Zero-shot Protocol-A
    zs = {}
    for e in encoders_zs:
        for n in (2000, 1000, 500, 200):
            data = _load_json(ZS / f"{e}_protocolA_n{n}.json")
            if data:
                zs[e] = (n, data)
                break

    # Fine-tune Protocol-A (from grid_* runs)
    ft = {}
    for cell, pat in [
        ("dinov2_salad", "*grid-dinov2-salad-30k*"),
        ("dinov2_mlp",   "*grid-dinov2-mlp-30k*"),
        ("clip_salad",   "*grid-clip-salad-30k*"),
        ("clip_mlp",     "*grid-clip-mlp-30k*"),
        # also re-evaluated 15k checkpoints from earlier work
        ("dinov2_salad_15k", "*_salad-15k_*"),
        ("dinov2_mlp_15k",   "*_cls-15k_*"),
        ("clip_salad_15k_lowlr", "*clip-salad-15k-lowlr*"),
        ("clip_mlp_15k_lowlr",   "*clip-cls-15k-lowlr*"),
    ]:
        run = _find_latest_run(pat)
        if run:
            data = _load_json(run / "phase1_holdout.json")
            if data:
                ft[cell] = data

    lines = ["# Headline retrieval table (Protocol A — leak-free crop→page)\n"]
    lines.append("## Zero-shot baselines\n")
    lines.append("| Encoder | n_pages | R@1 | R@5 | R@10 | R@20 | sanity gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for e in encoders_zs:
        if e not in zs:
            lines.append(f"| {e} | — | — | — | — | — | — |")
            continue
        n, d = zs[e]
        r = d["recall"]
        gap = d["sanity"]["gap"]
        lines.append(f"| {e} | {d['num_pages_evaluated']} "
                     f"| {r['1']:.3f} | {r['5']:.3f} | {r['10']:.3f} | {r['20']:.3f} "
                     f"| {gap:+.3f} |")

    lines.append("\n## Fine-tuned (Track A: 30k pages × 3 ep, single-seed)\n")
    lines.append("| Run | n_pages | R@1 | R@5 | R@10 | R@20 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cell, label in [
        ("dinov2_salad", "DINOv2-SALAD"),
        ("dinov2_mlp",   "DINOv2-MLP"),
        ("clip_salad",   "CLIP-SALAD (lr_bb=1e-7)"),
        ("clip_mlp",     "CLIP-MLP (lr_bb=1e-7)"),
    ]:
        if cell not in ft:
            lines.append(f"| {label} | — | — | — | — | — |")
            continue
        d = ft[cell]; r = d["recall"]
        lines.append(f"| {label} | {d['num_pages_evaluated']} "
                     f"| {r['1']:.3f} | {r['5']:.3f} | {r['10']:.3f} | {r['20']:.3f} |")

    lines.append("\n## Earlier 15k checkpoints (re-evaluated under Protocol A)\n")
    lines.append("| Run | n_pages | R@1 | R@5 | R@10 | R@20 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cell, label in [
        ("dinov2_salad_15k", "DINOv2-SALAD-15k"),
        ("dinov2_mlp_15k",   "DINOv2-MLP-15k"),
        ("clip_salad_15k_lowlr", "CLIP-SALAD-15k-lowlr"),
        ("clip_mlp_15k_lowlr",   "CLIP-MLP-15k-lowlr"),
    ]:
        if cell not in ft:
            lines.append(f"| {label} | — | — | — | — | — |")
            continue
        d = ft[cell]; r = d["recall"]
        lines.append(f"| {label} | {d['num_pages_evaluated']} "
                     f"| {r['1']:.3f} | {r['5']:.3f} | {r['10']:.3f} | {r['20']:.3f} |")
    return "\n".join(lines)


def titleblanked_table() -> str:
    """Title-blanked Phase-2 vs original — H-OCR test."""
    lines = ["# Title-blanked Phase-2 (H-OCR hypothesis test)\n"]
    lines.append("Encoders that read titles should drop substantially after the "
                 "top 15 % of every anchor + pool image is painted white.\n")
    lines.append("| Run | original R@1 | blanked-15 R@1 | Δ |")
    lines.append("|---|---:|---:|---:|")

    for label, pat in [
        ("DINOv2-SALAD-15k", "*_salad-15k_*"),
        ("DINOv2-MLP-15k",   "*_cls-15k_*"),
        ("CLIP-SALAD-15k-lowlr", "*clip-salad-15k-lowlr*"),
        ("DINOv2-SALAD-30k", "*grid-dinov2-salad-30k*"),
        ("DINOv2-MLP-30k",   "*grid-dinov2-mlp-30k*"),
        ("CLIP-SALAD-30k",   "*grid-clip-salad-30k*"),
        ("CLIP-MLP-30k",     "*grid-clip-mlp-30k*"),
    ]:
        run = _find_latest_run(pat)
        if not run:
            lines.append(f"| {label} | — | — | — |")
            continue
        orig = _load_json(run / "phase2_recall.json")
        blanked = _load_json(run / "phase2_titleblanked_15.json")
        o = orig["recall"].get("1") if orig and "recall" in orig else None
        b = blanked["recall"].get("1") if blanked and "recall" in blanked else None
        delta = (b - o) if (o is not None and b is not None) else None
        o_s = f"{o:.3f}" if o is not None else "—"
        b_s = f"{b:.3f}" if b is not None else "—"
        d_s = f"{delta:+.3f}" if delta is not None else "—"
        lines.append(f"| {label} | {o_s} | {b_s} | {d_s} |")
    return "\n".join(lines)


def saturation_curve() -> str:
    """Read metrics.jsonl from the saturation run; emit step → R@10 CSV."""
    run = _find_latest_run("*saturation-dinov2-salad-50k*")
    if not run:
        return "# Saturation curve\n\n(no saturation run found)\n"
    metrics_path = run / "metrics.jsonl"
    if not metrics_path.exists():
        return "# Saturation curve\n\n(no metrics.jsonl yet)\n"

    rows = []
    with open(metrics_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "eval_step" in r:
                # in-loop eval row; use legacy phase1_recall@10 (always logged)
                rows.append((int(r["eval_step"]),
                             float(r.get("phase1_recall@10", 0.0))))
    out = "step,phase1_recall_at_10\n" + "\n".join(f"{s},{r:.4f}" for s, r in rows)
    return out


def platonic_grid_summary() -> str:
    """Find the latest platonic_alignment_* dir and surface key alignment numbers."""
    candidates = sorted(RUNS.glob("platonic_alignment_*"), reverse=True)
    if not candidates:
        return "# Platonic alignment\n\n(no run found)\n"
    report = candidates[0] / "report.json"
    if not report.exists():
        return "# Platonic alignment\n\n(report.json missing)\n"
    data = json.loads(report.read_text())
    lines = [f"# Platonic alignment grid (n={data.get('n_samples', '?')})\n"]
    lines.append("Mutual-kNN@10 cells (higher = closer alignment).\n")
    img_encs = ["dinov2", "clip_image", "siglip_image", "imagenet_vit",
                "plain_vit", "ijepa"]
    txt_encs = ["bert", "minilm", "clip_text", "siglip_text"]
    lines.append("| image \\ text | " + " | ".join(txt_encs) + " |")
    lines.append("|---|" + "|".join("---:" for _ in txt_encs) + "|")
    pairs = data.get("pairs", {})
    for ie in img_encs:
        cells = []
        for te in txt_encs:
            key1 = f"{ie}__{te}"; key2 = f"{te}__{ie}"
            v = pairs.get(key1) or pairs.get(key2)
            cells.append(f"{v['mutual_knn_10']:.3f}" if v else "—")
        lines.append(f"| {ie} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    DEMO.mkdir(parents=True, exist_ok=True)
    (DEMO / "retrieval_table.md").write_text(retrieval_table())
    (DEMO / "titleblanked_table.md").write_text(titleblanked_table())
    (DEMO / "saturation_curve.csv").write_text(saturation_curve())
    (DEMO / "platonic_grid_summary.md").write_text(platonic_grid_summary())
    print(f"Demo deliverables written to {DEMO}")
    for f in sorted(DEMO.glob("*")):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

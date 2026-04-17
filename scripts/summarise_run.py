#!/usr/bin/env python
"""Pretty-print a run directory (PROJECT_SPEC.md §15, AGENTS/TESTS.md X3).

Emits a compact human-readable summary — no plots, no fancy layout — so
it's trivially diffable across runs::

    scripts/summarise_run.py runs/<id>/

Covers the X3 acceptance contract: output contains ``experiment_name``,
final loss, final phase1_recall@10, SALAD vendor SHA, visword git SHA,
and wall time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _last_with(rows: list[dict], key: str):
    for r in reversed(rows):
        if key in r:
            return r
    return None


def summarise(run_dir: Path) -> str:
    """Return the formatted summary as a string."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"no such run dir: {run_dir}")

    prov_path = run_dir / "provenance.json"
    cfg_path = run_dir / "config.resolved.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    phase1_path = run_dir / "phase1_recall.json"
    phase2_path = run_dir / "phase2_recall.json"

    prov: dict = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    cfg: dict = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    metrics: list[dict] = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if line.strip():
                metrics.append(json.loads(line))

    last_train = _last_with(metrics, "loss")
    last_eval = _last_with(metrics, "phase1_recall@10")
    phase1 = json.loads(phase1_path.read_text()) if phase1_path.exists() else {}
    phase2 = json.loads(phase2_path.read_text()) if phase2_path.exists() else {}

    lines = [f"# run: {run_dir}"]
    lines.append(f"experiment_name : {cfg.get('experiment_name', '?')}")
    lines.append(f"model_kind      : {cfg.get('model_kind', '?')}")
    lines.append(f"visword_git_sha : {prov.get('visword_git_sha', '?')}")
    lines.append(f"salad_vendor_sha: {prov.get('salad_vendor_sha', '?')}")
    lines.append(f"gpu             : {prov.get('gpu', '?')}")
    lines.append(f"hostname        : {prov.get('hostname', '?')}")
    lines.append(f"slurm_job_id    : {prov.get('slurm_job_id') or '-'}")
    lines.append(f"data_fingerprint: {(prov.get('data_fingerprint') or '')[:16]}")

    if last_train:
        lines.append(
            f"final loss      : {last_train['loss']:.4f}"
            f"  (step {last_train['step']}, top1={last_train.get('top1_acc', 0):.3f})"
        )
        lines.append(f"wall time       : {last_train.get('wall_time_s', 0):.1f} s")
    else:
        lines.append("final loss      : n/a (no training rows)")

    # Prefer the post-CLI phase1 recall if written, fall back to the in-training eval row.
    if phase1:
        r = phase1.get("recall", {})
        lines.append(
            f"phase1 recall   : R@1={r.get('1', 0):.3f}  R@5={r.get('5', 0):.3f}  "
            f"R@10={r.get('10', 0):.3f}  R@20={r.get('20', 0):.3f}"
        )
        lines.append(
            f"phase1 sanity   : gap={phase1.get('sanity', {}).get('gap', 0):.3f}  "
            f"monotonic={phase1.get('sanity', {}).get('monotonic')}"
        )
    elif last_eval:
        lines.append(
            f"phase1 recall   : (from metrics.jsonl) "
            f"R@10={last_eval.get('phase1_recall@10', 0):.3f}"
        )

    if phase2:
        r = phase2.get("recall", {})
        lines.append(
            f"phase2 recall   : R@1={r.get('1', 0):.3f}  R@5={r.get('5', 0):.3f}  "
            f"R@10={r.get('10', 0):.3f}  R@20={r.get('20', 0):.3f}"
        )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    args = p.parse_args(argv)
    sys.stdout.write(summarise(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

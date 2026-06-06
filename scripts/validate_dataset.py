#!/usr/bin/env python
"""Validate the Wikipedia screenshot training set (size + quality).

Reports, so we know what we're actually training on:
  - total pages; UNIQUE images (by manifest image_sha256 -> exact duplicates) and
    unique docids (leakage / repeat risk);
  - body-text coverage on a sample (empty/short pages are useless for the body-target
    reader, ticket 04) — chars + rough token estimate;
  - image-size consistency on a sample;
  - train/eval split math given the held-out eval slice (tail of seeded permutation).

CPU/login-friendly: reads the manifest (cheap) and samples files; does not read all
~376k text/image files. Reproducible.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from visword.data import manifest as M


def pct(x, n):
    return f"{100.0 * x / n:.1f}%" if n else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--num-eval", type=int, default=2000, help="held-out eval slice (tail)")
    ap.add_argument("--sample", type=int, default=5000, help="rows to sample for text/image stats")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    man = M.read_manifest(args.cache_dir)
    rows = man["rows"]
    n = man.get("num_rows", len(rows))
    print(f"=== {args.cache_dir} ===")
    print(f"total pages (manifest num_rows): {n}")

    # Exact duplicates + id uniqueness (from manifest, no file I/O).
    shas = [r.get("image_sha256") for r in rows if r.get("image_sha256")]
    docids = [r.get("docid") for r in rows]
    uniq_sha = len(set(shas))
    uniq_doc = len(set(docids))
    print(f"unique image_sha256: {uniq_sha}  ({n - uniq_sha} exact-duplicate images, {pct(n - uniq_sha, n)})")
    print(f"unique docids      : {uniq_doc}  ({n - uniq_doc} repeated docids, {pct(n - uniq_doc, n)})")
    dup_doc = [d for d, c in Counter(docids).items() if c > 1][:5]
    if dup_doc:
        print(f"  example repeated docids: {dup_doc}")

    # Sample for text + image stats.
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(n, size=min(args.sample, n), replace=False)
    text_chars, empty, missing = [], 0, 0
    sizes = Counter()
    for i in sample:
        r = rows[int(i)]
        tp = args.cache_dir / r.get("text_path", "")
        if r.get("text_path") and tp.exists():
            body = tp.read_text(errors="ignore")
            text_chars.append(len(body))
            if len(body.strip()) == 0:
                empty += 1
        else:
            missing += 1
    # image sizes on a smaller subset
    for i in sample[:300]:
        ip = args.cache_dir / rows[int(i)]["image_path"]
        try:
            with Image.open(ip) as im:
                sizes[im.size] += 1
        except Exception:
            pass

    ns = len(sample)
    print(f"\n--- text body coverage (sample n={ns}) ---")
    print(f"missing text_path : {missing} ({pct(missing, ns)})")
    print(f"empty body        : {empty} ({pct(empty, ns)})")
    if text_chars:
        tc = sorted(text_chars)
        print(f"body chars: min={tc[0]} median={int(statistics.median(tc))} "
              f"mean={int(statistics.mean(tc))} p90={tc[int(0.9*len(tc))-1]} max={tc[-1]}")
        print(f"  ~tokens (chars/4): median≈{int(statistics.median(tc)/4)} "
              f"(BERT cap 512 -> long pages are truncated)")
        short = sum(1 for c in text_chars if c < 200)
        print(f"  pages with <200 chars body: {short} ({pct(short, len(text_chars))})")
    print(f"\n--- image sizes (sample n=300) ---")
    for sz, c in sizes.most_common(5):
        print(f"  {sz}: {c}")

    # Split math.
    eval_n = min(args.num_eval, n)
    train_pool = n - eval_n
    print(f"\n--- train/eval split (eval = tail {eval_n}, seed {args.seed}) ---")
    print(f"eval slice pages : {eval_n}")
    print(f"train pool pages : {train_pool} (disjoint head)")
    for cpp in (20,):
        print(f"  @~{cpp} legible crops/page -> train pool ≈ {train_pool * cpp:,} crops; "
              f"eval ≈ {eval_n * cpp:,} crops")
    print(f"config default num_train_samples=10000 uses only {pct(10000, train_pool)} of the pool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

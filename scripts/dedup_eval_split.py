#!/usr/bin/env python3
"""Perceptual-hash deduplication for the wiki_ss cache.

Why: random index permutation can place near-duplicate template pages
(e.g. two minor-league-football-club stubs with identical infobox layout)
into the train and eval splits respectively, inflating in-distribution
recall without measuring real generalisation.

What it does:
  1. Compute a 64-bit perceptual hash (phash) for every image in
     ``data/wiki_ss/blobs/`` (CPU-bound; ~2 ms/image).
  2. Cluster images by Hamming distance ≤ ``--hamming`` (default 6).
     Two pages in the same cluster are considered visual near-duplicates.
  3. Write ``data/wiki_ss/dedup_split.json`` mapping each manifest row
     index to its cluster id, plus a summary.

Downstream consumers (modified ``train._split_indices`` and the new
Protocol-A eval) can use this map to enforce: no eval page shares a
cluster with any training page.

Usage::

    PYTHONPATH=src python -m scripts.dedup_eval_split \\
        --cache-dir /scratch/<USER>/VISWORD/data/wiki_ss \\
        --workers 8 --hamming 6
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import imagehash
from PIL import Image

from visword.data import manifest as M


def _phash_one(args: tuple[int, str]) -> tuple[int, int] | None:
    """Worker: ``(idx, blob_path) → (idx, hash_int)``. Returns None on failure."""
    idx, path = args
    try:
        with Image.open(path) as im:
            h = imagehash.phash(im)  # 8x8 = 64-bit hash
        return idx, int(str(h), 16)
    except Exception:
        return None


def _iter_blob_paths(cache_dir: Path, manifest: dict) -> Iterable[tuple[int, str]]:
    for entry in manifest["rows"]:
        idx = entry["idx"]
        path = cache_dir / entry["image_path"]
        if path.exists():
            yield idx, str(path)


def _hamming(a: int, b: int) -> int:
    # Python 3.9 has no int.bit_count(); use bin().count("1").
    return bin(a ^ b).count("1")


def _popcount64_array(x: "np.ndarray") -> "np.ndarray":
    """Vectorised popcount on uint64 array. SWAR-style 64-bit popcount."""
    import numpy as np
    x = x.astype(np.uint64, copy=False)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0f0f0f0f0f0f0f0f)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return ((x * h01) >> np.uint64(56)).astype(np.int64)


def _cluster_by_hamming(
    hashes: dict[int, int], hamming_threshold: int,
) -> dict[int, int]:
    """Vectorised first-match clustering by Hamming distance.

    For each page in deterministic order, check against existing
    cluster representatives; assign to first rep within ``hamming_threshold``.
    If none matches, the page becomes a new cluster's representative.
    Avoids the chaining problem of single-link clustering (where
    transitively-similar pages can merge into a single giant pseudo-cluster).

    Vectorised: for each new page, compute pairwise Hamming with all
    current reps via numpy bitwise XOR + popcount.
    """
    import numpy as np
    items = sorted(hashes.items())                 # deterministic order
    n = len(items)
    if n == 0:
        return {}
    idx_arr = np.array([k for k, _ in items], dtype=np.int64)
    hash_arr = np.array([v for _, v in items], dtype=np.uint64)

    cluster_of = np.full(n, -1, dtype=np.int64)
    rep_hashes = np.empty(0, dtype=np.uint64)      # grows as we find reps

    last_log = 0
    for i in range(n):
        h = hash_arr[i]
        if rep_hashes.size > 0:
            xors = rep_hashes ^ h
            dists = _popcount64_array(xors)
            within = np.where(dists <= hamming_threshold)[0]
            if within.size > 0:
                cluster_of[i] = int(within[0])
                continue
        # New cluster.
        cluster_of[i] = rep_hashes.size
        rep_hashes = np.append(rep_hashes, h)
        if i - last_log >= 20000:
            print(f"  cluster-pass {i}/{n}  reps={rep_hashes.size}", flush=True)
            last_log = i

    return {int(idx_arr[i]): int(cluster_of[i]) for i in range(n)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--hamming", type=int, default=6,
                   help="cluster pages with Hamming distance <= this")
    p.add_argument("--max-pages", type=int, default=None,
                   help="stop after N pages (useful for smoke testing)")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON (default: <cache>/dedup_split.json)")
    args = p.parse_args()

    cache_dir = args.cache_dir.resolve()
    out_path = args.out or (cache_dir / "dedup_split.json")
    manifest = M.read_manifest(cache_dir)

    paths = list(_iter_blob_paths(cache_dir, manifest))
    if args.max_pages:
        paths = paths[: args.max_pages]
    print(f"Hashing {len(paths)} blobs with {args.workers} workers...", flush=True)

    t0 = time.time()
    hashes: dict[int, int] = {}
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_phash_one, p) for p in paths]
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if res is None:
                failed += 1
                continue
            idx, h = res
            hashes[idx] = h
            if done % 5000 == 0:
                rate = done / max(1.0, time.time() - t0)
                print(f"  {done}/{len(paths)} ({rate:.0f}/s, failed={failed})", flush=True)

    print(f"Hashed {len(hashes)} blobs in {time.time() - t0:.1f}s ({failed} failed)")

    print(f"Clustering at Hamming <= {args.hamming}...", flush=True)
    t1 = time.time()
    cluster_of = _cluster_by_hamming(hashes, args.hamming)
    n_clusters = max(cluster_of.values()) + 1 if cluster_of else 0
    print(f"  {n_clusters} clusters from {len(cluster_of)} pages in {time.time() - t1:.1f}s")

    # Cluster-size histogram
    from collections import Counter
    sizes = Counter(cluster_of.values())
    size_hist = Counter(sizes.values())
    big_clusters = [(cid, n) for cid, n in sizes.most_common(10) if n > 1]

    payload = {
        "hamming_threshold": args.hamming,
        "num_pages_hashed": len(hashes),
        "num_pages_failed": failed,
        "num_clusters": n_clusters,
        "num_singletons": int(size_hist.get(1, 0)),
        "num_pages_in_clusters_of_size_2plus": sum(n for s, n in sizes.items() if n >= 2)
                                              if False else
                                              sum(n for n in sizes.values() if n >= 2),
        "size_histogram": {str(k): v for k, v in sorted(size_hist.items())},
        "biggest_clusters_top10": [{"cluster_id": cid, "size": n} for cid, n in big_clusters],
        "cluster_of": {str(k): v for k, v in sorted(cluster_of.items())},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print(f"  Singletons: {payload['num_singletons']}")
    print(f"  Pages in 2+ clusters: {payload['num_pages_in_clusters_of_size_2plus']}")
    print(f"  Biggest clusters: {payload['biggest_clusters_top10'][:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

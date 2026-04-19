#!/usr/bin/env python
"""Enumerate ALL files in Tevatron/wiki-ss-corpus + per-parquet row counts.

Output JSON to stdout. Logs every file to stderr so we can see non-parquet
layouts (jsonl shards, tar archives, etc).
"""
from __future__ import annotations

import json
import sys

from huggingface_hub import HfApi, hf_hub_download


REPO = "Tevatron/wiki-ss-corpus"


def main() -> int:
    api = HfApi()
    files = sorted(api.list_repo_files(REPO, repo_type="dataset"))
    print(f"--- {len(files)} files in repo ---", file=sys.stderr)
    for f in files:
        print(f, file=sys.stderr)
    parquets = [f for f in files if f.endswith(".parquet")]
    print(f"\n--- {len(parquets)} parquet files ---", file=sys.stderr)

    info = []
    cum = 0
    if parquets:
        import pyarrow.parquet as pq
        for fname in parquets:
            local = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset")
            meta = pq.ParquetFile(local).metadata
            nrows = meta.num_rows
            info.append({
                "filename": fname,
                "num_rows": nrows,
                "global_start_idx": cum,
                "global_end_idx": cum + nrows,
            })
            cum += nrows
            print(f"{fname:60s}  rows={nrows:7d}  cum={cum:8d}", file=sys.stderr)

    payload = {
        "repo": REPO,
        "all_files": files,
        "num_parquet_files": len(parquets),
        "total_parquet_rows": cum,
        "files": info,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

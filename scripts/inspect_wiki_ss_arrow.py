#!/usr/bin/env python
"""Read state.json + dataset_info.json from Tevatron/wiki-ss-corpus and emit
a per-shard row count map for parallel prefetching.

Output JSON to stdout:
  {repo, total_rows, num_shards, files: [{filename, num_rows, global_start_idx, global_end_idx}]}
"""
from __future__ import annotations

import json
import sys

from huggingface_hub import hf_hub_download


REPO = "Tevatron/wiki-ss-corpus"


def main() -> int:
    state_path = hf_hub_download(repo_id=REPO, filename="train/state.json", repo_type="dataset")
    info_path = hf_hub_download(repo_id=REPO, filename="train/dataset_info.json", repo_type="dataset")
    state = json.load(open(state_path))
    info = json.load(open(info_path))

    print("=== state.json keys ===", file=sys.stderr)
    print(list(state.keys()), file=sys.stderr)
    print("=== dataset_info.json keys ===", file=sys.stderr)
    print(list(info.keys()), file=sys.stderr)
    print("--- state.json (head) ---", file=sys.stderr)
    print(json.dumps({k: state[k] for k in list(state.keys())[:6]}, indent=2)[:2000], file=sys.stderr)
    print("--- dataset_info splits ---", file=sys.stderr)
    if "splits" in info:
        print(json.dumps(info["splits"], indent=2)[:2000], file=sys.stderr)

    # The HF arrow saved-dataset layout puts shard rows under
    # state["_data_files"] which is a list of {"filename": "...", ...}.
    files_meta = state.get("_data_files") or []
    print(f"=== {len(files_meta)} _data_files entries ===", file=sys.stderr)

    # Each file's row count isn't always stored — but if state has
    # "_indexes" or info["splits"]["train"]["num_examples"] is total, we can
    # divide. Better: read each arrow file's footer for num_rows (cheap).
    import pyarrow.ipc as ipc
    out_files = []
    cum = 0
    for entry in files_meta:
        fname = "train/" + entry["filename"]
        local = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset")
        with open(local, "rb") as fh:
            reader = ipc.open_stream(fh) if entry["filename"].endswith(".arrow") else ipc.open_file(fh)
            try:
                nrows = sum(b.num_rows for b in reader)
            except Exception:
                # Fall back to file-format reader
                fh.seek(0)
                reader = ipc.open_file(fh)
                nrows = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
        out_files.append({
            "filename": fname,
            "num_rows": nrows,
            "global_start_idx": cum,
            "global_end_idx": cum + nrows,
        })
        cum += nrows
        print(f"{fname:40s}  rows={nrows:6d}  cum={cum:9d}", file=sys.stderr)

    payload = {
        "repo": REPO,
        "total_rows": cum,
        "num_shards": len(out_files),
        "files": out_files,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

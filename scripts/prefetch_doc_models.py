#!/usr/bin/env python
"""Prefetch document/text-pretrained model weights into the HF cache.

Run on the LOGIN node (compute nodes are offline). Uses `snapshot_download`,
which pulls every file in a repo regardless of how the model is later loaded —
so it is robust to not having the right model class / engine installed yet.

Does NOT set HF_HUB_OFFLINE (we are downloading). Honors HF_TOKEN if exported
(see scripts/_env.sh, which loads it from .env to lift the rate limit).

ColPali / ColQwen2 are intentionally NOT here: they need the `colpali-engine`
dependency and an adapter+base repo pair, decided at ticket 03 (document family).

Usage:
    python scripts/prefetch_doc_models.py            # default list
    python scripts/prefetch_doc_models.py repo/id ...  # explicit list
"""
import sys

# (repo_id, why) — kept explicit so the cache contents are self-documenting.
DEFAULT_MODELS = [
    ("facebook/vit-mae-base",     "our MAE reader base (ticket 04) — primary P0 blocker"),
    ("facebook/vit-mae-large",    "our MAE reader, larger variant (ticket 04)"),
    ("google/pix2struct-base",    "document family ceiling (ticket 03)"),
    ("naver-clova-ix/donut-base", "document family ceiling (ticket 03)"),
    ("facebook/nougat-small",     "document family ceiling (ticket 03)"),
    ("facebook/nougat-base",      "document family ceiling (ticket 03)"),
]


def main(argv):
    from huggingface_hub import snapshot_download

    if argv:
        models = [(m, "explicit") for m in argv]
    else:
        models = DEFAULT_MODELS

    ok, fail = [], []
    for repo_id, why in models:
        print(f"==> downloading {repo_id}  ({why})", flush=True)
        try:
            path = snapshot_download(repo_id=repo_id)
            print(f"    OK   {repo_id} -> {path}", flush=True)
            ok.append(repo_id)
        except Exception as e:  # noqa: BLE001 - report-and-continue
            print(f"    FAIL {repo_id}: {e!r}", flush=True)
            fail.append(repo_id)

    print("\n=== prefetch summary ===", flush=True)
    print(f"  OK   ({len(ok)}): {', '.join(ok) or '-'}", flush=True)
    print(f"  FAIL ({len(fail)}): {', '.join(fail) or '-'}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Ticket 05 — perfect-text upper bound ("if reading were perfect").

Feeds the ground-truth page text through frozen BERT and retrieves at the page level on
the SAME disjoint eval slice (seed 42 tail) used by the grid, to bound how far the visual
encoders / MAE reader sit from an ideal reader. Two ceilings:

  * BODY re-id (directly comparable to the visual crop->page LOO): split each page's body
    into K chunks, embed each with BERT[CLS] -> K "views"/page, score with the SAME
    page_reid_recall (leave-one-out) the visual grid uses.
  * TITLE->BODY (the "title version"): query = BERT[CLS](title), gallery = BERT[CLS](body),
    recall@k that a title retrieves its OWN page's body (cross-field; no LOO needed).

No OCR engine — uses ground-truth text. JSONL/JSON outputs only.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

from visword.data import manifest as M
from visword.page_reid import page_reid_recall


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def read_body(cache_dir: Path, row: dict) -> str:
    tp = row.get("text_path")
    if tp and (cache_dir / tp).exists():
        return (cache_dir / tp).read_text(errors="ignore")
    return row.get("title", "")


def chunks_of(text: str, k: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    n = max(1, len(text) // k)
    out = [text[i:i + n].strip() for i in range(0, len(text), n)]
    return [c for c in out if c][:k]


@torch.no_grad()
def bert_embed(texts: list[str], tok, bert, max_tokens: int, device, batch: int = 64) -> torch.Tensor:
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], padding="max_length", truncation=True,
                  max_length=max_tokens, return_tensors="pt").to(device)
        cls = bert(**enc).last_hidden_state[:, 0, :].float()
        out.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(out) if out else torch.empty(0, 768)


def recall_cross(query: torch.Tensor, gallery: torch.Tensor, ks) -> dict:
    """recall@k that query[i]'s own page (gallery[i]) is in the top-k of gallery."""
    sim = query @ gallery.T
    n, P = sim.shape
    own = torch.arange(n)
    rec = {}
    for k in sorted(ks):
        topk = sim.topk(min(k, P), dim=1).indices
        rec[str(k)] = float((topk == own.unsqueeze(1)).any(1).float().mean())
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-eval", type=int, default=2000)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--max-text-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-values", nargs="*", type=int, default=[1, 5, 10, 20])
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = M.read_manifest(args.cache_dir); rows = man["rows"]
    n = man.get("num_rows", len(rows))
    eval_idx = np.random.default_rng(args.seed).permutation(n)[n - min(args.num_eval, n):]

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

    titles, bodies = [], []
    chunk_texts, chunk_pids = [], []
    for local, gi in enumerate(eval_idx):
        r = rows[int(gi)]
        titles.append((r.get("title") or "").replace("_", " "))
        body = read_body(args.cache_dir, r)
        bodies.append(body)
        for c in chunks_of(body, args.chunks):
            chunk_texts.append(c); chunk_pids.append(local)

    title_emb = bert_embed(titles, tok, bert, args.max_text_tokens, device)
    body_emb = bert_embed(bodies, tok, bert, args.max_text_tokens, device)
    chunk_emb = bert_embed(chunk_texts, tok, bert, args.max_text_tokens, device)

    body_reid = page_reid_recall(chunk_emb, torch.tensor(chunk_pids), k_values=tuple(args.k_values))
    title2body = recall_cross(title_emb, body_emb, args.k_values)

    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "host": socket.gethostname(), "git_sha": git_sha(root), "device": str(device),
        "num_eval_pages": int(len(eval_idx)), "chunks_per_page": args.chunks,
        "max_text_tokens": args.max_text_tokens,
        "body_reid_recall": body_reid["recall"],            # comparable to visual crop->page LOO
        "body_reid_eligible": body_reid["num_queries_eligible"],
        "title_to_body_recall": title2body,                 # cross-field "title version"
    }
    (args.out / "perfect_text_bound.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

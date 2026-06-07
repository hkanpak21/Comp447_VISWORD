"""Ticket 04 — train our MAE reader: regress pooled MAE features to frozen BERT[CLS]
of the page BODY (objective matches Barış's I-JEPA text-target; backbone = MAE).

Self-contained + arg-driven (no Config-schema change). Parameter-efficient: only the
last-N MAE blocks + the projection head train. Resumable in <=8h chunks (auto-resumes
from <out>/checkpoints/last.pt). Evaluated with the SAME page-level same-page re-id
protocol + native-224 legible crops as the ticket-02 grid, so the after-number is
directly comparable to frozen MAE (R@10 0.036, the "before").

Honors offline + run-dir conventions; JSONL + PNG only (no trackers).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from visword.data import manifest as M
from visword.data.cropper import TextAwareCropper
from visword.models.mae_reader import MAEBodyReader
from visword.page_reid import page_reid_recall

_T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


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


class PageCrops(Dataset):
    """One page -> its native-224 legible crops (capped) + the page's target index.

    Ticket 07: with probability ``title_mask_prob`` the top ``title_mask_frac`` of the page
    is painted white BEFORE cropping (random title/header-region masking), so the reader
    can't bind page identity to the title bar.
    """

    def __init__(self, cache_dir, rows, page_idx, cropper, max_crops,
                 title_mask_prob=0.0, title_mask_frac=0.25):
        self.cache_dir, self.rows = cache_dir, rows
        self.page_idx, self.cropper, self.max_crops = page_idx, cropper, max_crops
        self.title_mask_prob, self.title_mask_frac = title_mask_prob, title_mask_frac

    def __len__(self):
        return len(self.page_idx)

    def __getitem__(self, i):
        gi = int(self.page_idx[i])
        with Image.open(self.cache_dir / self.rows[gi]["image_path"]) as im:
            page = im.convert("RGB")
            if self.title_mask_prob > 0.0 and random.random() < self.title_mask_prob:
                arr = np.asarray(page).copy()
                arr[: int(round(arr.shape[0] * self.title_mask_frac)), :, :] = 255
                page = Image.fromarray(arr)
            crops = self.cropper(page)[: self.max_crops]
        x = torch.stack([_T(c) for c in crops]) if crops else torch.empty(0, 3, 224, 224)
        return x, i


def _collate(batch):
    batch = [(x, i) for x, i in batch if x.shape[0] > 0]
    if not batch:
        return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
    crops = torch.cat([x for x, _ in batch])
    tgt = torch.cat([torch.full((x.shape[0],), i, dtype=torch.long) for x, i in batch])
    return crops, tgt


@torch.no_grad()
def bert_targets(rows, page_idx, cache_dir, max_tokens, device, pooling="cls", batch=64):
    """Precompute normalised frozen BERT body embeddings -> (P, 768).

    pooling="cls" (regression target, matches Barış) or "mean" (masked mean — the much
    stronger readout per the perfect-text bound; used as the contrastive text anchor).
    """
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
    out = []
    for s in range(0, len(page_idx), batch):
        texts = [read_body(cache_dir, rows[int(g)]) for g in page_idx[s:s + batch]]
        enc = tok(texts, padding="max_length", truncation=True,
                  max_length=max_tokens, return_tensors="pt").to(device)
        h = bert(**enc).last_hidden_state.float()
        if pooling == "mean":
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            v = h[:, 0, :]
        out.append(F.normalize(v, dim=-1).cpu())
    del bert
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out)


@torch.no_grad()
def evaluate(reader, cache_dir, rows, eval_idx, cropper, device, batch=64, max_crops=None):
    reader.eval()
    embs, pids = [], []
    buf, bpid = [], []

    def flush():
        if buf:
            embs.append(reader(torch.stack(buf).to(device)).cpu()); pids.extend(bpid)
            buf.clear(); bpid.clear()

    for local, gi in enumerate(eval_idx):
        with Image.open(cache_dir / rows[int(gi)]["image_path"]) as im:
            crops = cropper(im.convert("RGB"))
        if max_crops:
            crops = crops[:max_crops]
        for c in crops:
            buf.append(_T(c)); bpid.append(local)
            if len(buf) >= batch:
                flush()
    flush()
    reader.train()
    E = torch.cat(embs) if embs else torch.empty(0, reader.descriptor_dim)
    return page_reid_recall(E, torch.tensor(pids), k_values=(1, 5, 10, 20))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-train", type=int, default=20000)
    ap.add_argument("--num-eval", type=int, default=2000)
    ap.add_argument("--eval-pages", type=int, default=500, help="pages scored during PERIODIC eval")
    ap.add_argument("--final-eval-pages", type=int, default=2000,
                    help="pages for the FINAL eval (match the grid's 2000-page gallery for a comparable number)")
    ap.add_argument("--max-crops-per-page", type=int, default=8)
    ap.add_argument("--eval-max-crops", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-pages", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-trainable-blocks", type=int, default=4)
    ap.add_argument("--max-text-tokens", type=int, default=256)
    ap.add_argument("--eval-every-steps", type=int, default=200)
    ap.add_argument("--objective", choices=["regress", "contrastive"], default="regress",
                    help="regress = Smooth-L1 to BERT[CLS]; contrastive = InfoNCE crop<->mean-pool-BERT (in-batch negatives)")
    ap.add_argument("--temp", type=float, default=0.07, help="contrastive temperature")
    ap.add_argument("--title-mask-prob", type=float, default=0.0,
                    help="ticket 07: prob of blanking the top title region during training")
    ap.add_argument("--title-mask-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    man = M.read_manifest(args.cache_dir); rows = man["rows"]
    n = man.get("num_rows", len(rows))
    perm = np.random.default_rng(args.seed).permutation(n)
    eval_idx = perm[n - min(args.num_eval, n):]               # disjoint tail (== ticket 01/02)
    train_idx = perm[: args.num_train]                         # head
    cropper = TextAwareCropper(crop_size=224, target_size=224)

    ckpt_dir = args.out / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    _pooling = "mean" if args.objective == "contrastive" else "cls"
    resolved = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "text_source": "body", "bert_pooling": _pooling, "base": MAEBodyReader.HF_NAME}
    (args.out / "config.resolved.json").write_text(json.dumps(resolved, indent=2))
    (args.out / "provenance.json").write_text(json.dumps({
        "ts_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "host": socket.gethostname(), "git_sha": git_sha(root), "device": str(device),
        "num_train_pages": int(len(train_idx)), "num_eval_pages": int(len(eval_idx)),
    }, indent=2))

    print(f"precomputing BERT-{_pooling} body targets for {len(train_idx)} train pages ...", flush=True)
    targets = bert_targets(rows, train_idx, args.cache_dir, args.max_text_tokens, device,
                           pooling=_pooling).to(device)

    reader = MAEBodyReader(num_trainable_blocks=args.num_trainable_blocks).to(device)
    n_train_p = sum(p.numel() for p in reader.trainable_parameters())
    print(f"trainable params: {n_train_p/1e6:.1f}M of {sum(p.numel() for p in reader.parameters())/1e6:.0f}M", flush=True)

    head_params = list(reader.head.parameters())
    head_ids = {id(p) for p in head_params}
    bb_params = [p for p in reader.trainable_parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": head_params, "lr": args.lr_head},
         {"params": bb_params, "lr": args.lr_backbone}], weight_decay=args.weight_decay)

    ds = PageCrops(args.cache_dir, rows, train_idx, cropper, args.max_crops_per_page,
                   title_mask_prob=args.title_mask_prob, title_mask_frac=args.title_mask_frac)
    dl = DataLoader(ds, batch_size=args.batch_pages, shuffle=True, num_workers=args.num_workers,
                    collate_fn=_collate, drop_last=True, persistent_workers=args.num_workers > 0)

    metrics_path = args.out / "metrics.jsonl"
    start_epoch, gstep, best = 0, 0, -1.0
    last = ckpt_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device)
        reader.load_state_dict(ck["reader"]); opt.load_state_dict(ck["opt"])
        start_epoch, gstep, best = ck["epoch"], ck["gstep"], ck.get("best", -1.0)
        print(f"resumed from {last}: epoch {start_epoch} step {gstep} best R@10 {best}", flush=True)
    else:
        r0 = evaluate(reader, args.cache_dir, rows, eval_idx[:args.eval_pages], cropper, device,
                      max_crops=args.eval_max_crops)
        with metrics_path.open("a") as f:
            f.write(json.dumps({"eval_step": 0, "phase1_recall": r0["recall"],
                                "note": "reader@init (random head); frozen-MAE baseline R@10=0.036 from grid"}) + "\n")
        print(f"reader@init R@10={r0['recall']['10']:.3f}", flush=True)

    reader.train()
    for epoch in range(start_epoch, args.epochs):
        for crops, tgt_idx in dl:
            if crops.shape[0] == 0:
                continue
            crops = crops.to(device)
            pred = reader(crops)                                   # (n, D) L2-normed
            if args.objective == "contrastive":
                # InfoNCE: each crop -> its page's mean-pool BERT-body anchor; in-batch
                # negatives are the other distinct pages in this batch.
                uniq, inv = torch.unique(tgt_idx, return_inverse=True)
                anchors = targets[uniq.to(device)]                 # (U, D)
                logits = pred @ anchors.T / args.temp              # (n, U)
                loss = F.cross_entropy(logits, inv.to(device))
            else:
                loss = F.smooth_l1_loss(pred, targets[tgt_idx.to(device)])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.trainable_parameters(), args.grad_clip)
            opt.step(); gstep += 1
            if gstep == 1 or gstep % 50 == 0:
                with metrics_path.open("a") as f:
                    f.write(json.dumps({"step": gstep, "epoch": epoch, "loss": float(loss),
                                        "gpu_mem_gb": (torch.cuda.max_memory_allocated()/1e9
                                                       if device.type == "cuda" else 0.0)}) + "\n")
                print(f"e{epoch} step {gstep} loss {float(loss):.4f}", flush=True)
            if gstep % args.eval_every_steps == 0:
                r = evaluate(reader, args.cache_dir, rows, eval_idx[:args.eval_pages], cropper, device,
                             max_crops=args.eval_max_crops)
                r10 = r["recall"]["10"]
                with metrics_path.open("a") as f:
                    f.write(json.dumps({"eval_step": gstep, "phase1_recall": r["recall"]}) + "\n")
                print(f"  [eval] step {gstep} R@10={r10:.3f} (best {best:.3f})", flush=True)
                torch.save({"reader": reader.state_dict(), "opt": opt.state_dict(),
                            "epoch": epoch, "gstep": gstep, "best": best}, last)
                if r10 > best:
                    best = r10
                    torch.save({"reader": reader.state_dict(), "gstep": gstep, "r10": r10},
                               ckpt_dir / "best.pt")
        torch.save({"reader": reader.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1, "gstep": gstep, "best": best}, last)

    # Final eval on the full gallery (match the grid's 2000-page gallery for comparability).
    fe = min(args.final_eval_pages, len(eval_idx))
    rf = evaluate(reader, args.cache_dir, rows, eval_idx[:fe], cropper, device,
                  max_crops=args.eval_max_crops)
    (args.out / "final_eval.json").write_text(json.dumps({
        "recall": rf["recall"], "sanity": rf["sanity"], "best_r10_during_train": best,
        "frozen_mae_baseline_r10": 0.036, "trainable_params": int(n_train_p),
        "num_train_pages": int(len(train_idx)), "eval_pages": int(fe),
    }, indent=2))
    print(f"FINAL R@10={rf['recall']['10']:.3f} (frozen-MAE before=0.036, best={best:.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

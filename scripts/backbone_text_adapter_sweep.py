#!/usr/bin/env python3
r"""Backbone + text adapter capacity sweep.

Hypothesis: Frozen backbones can be aligned to text space using small adapters
trained on (page-screenshot, page-title) pairs.

Pipeline:
  1. Sample N_train + N_eval pages.
  2. Encode each page-screenshot with frozen backbone (I-JEPA, CLIP, DINOv2).
  3. Encode each page-title with a frozen text encoder (BERT-base).
  4. Fit an adapter via InfoNCE.
  5. Eval on held-out pages.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from visword.data import manifest as M


ADAPTER_KINDS = ("linear", "bottleneck", "mlp", "deep_mlp", "low_rank")
BACKBONES = ("ijepa", "clip", "dinov2_cls", "dinov2_mean")


@dataclass(frozen=True)
class AdapterSpec:
    kind: str
    rank: int | None = None
    hidden_dim: int | None = None

    @property
    def label(self) -> str:
        if self.kind == "low_rank":
            return f"low_rank_r{self.rank}"
        if self.kind == "bottleneck":
            return f"bottleneck_r{self.rank}"
        if self.kind in {"mlp", "deep_mlp"} and self.hidden_dim is not None:
            return f"{self.kind}_h{self.hidden_dim}"
        return self.kind


def _sample_pages(cache_dir: Path, n_train: int, n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    manifest = M.read_manifest(cache_dir)
    rows = manifest["rows"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    skip = 3000
    picks = perm[skip : skip + n_train + n_eval]
    train_rows = [rows[i] for i in picks[:n_train]]
    eval_rows  = [rows[i] for i in picks[n_train:]]
    return train_rows, eval_rows


def _read_text(row: dict, cache_dir: Path, source: str = "title") -> str:
    title = row.get("title", "")
    if source == "title":
        return title
    text_path = cache_dir / row.get("text_path", "")
    if text_path.exists():
        body = text_path.read_text()[:200]
        return f"{title}. {body}"
    return title


def _center_crop_pil(im: Image.Image, target: int = 224) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side)).resize(
        (target, target), Image.BILINEAR)


@torch.no_grad()
def encode_images(rows: list[dict], cache_dir: Path, backbone_name: str,
                  device: torch.device, batch_size: int = 16) -> np.ndarray:
    from visword.models.zeroshot import ZeroShotIJepa, ZeroShotCLIPImage, ZeroShotDINOv2
    from visword.data.light_dataset import default_transform
    
    if backbone_name == "ijepa":
        enc = ZeroShotIJepa(cfg=None).to(device).eval()
    elif backbone_name == "clip":
        enc = ZeroShotCLIPImage(cfg=None).to(device).eval()
    elif backbone_name == "dinov2_cls":
        from visword.config import Config, BackboneConfig
        # Minimal dummy config for DINOv2
        cfg = Config.model_construct(backbone=BackboneConfig(arch="dinov2_vitb14", feature_dim=768))
        enc = ZeroShotDINOv2(cfg=cfg, mode="cls").to(device).eval()
    elif backbone_name == "dinov2_mean":
        from visword.config import Config, BackboneConfig
        cfg = Config.model_construct(backbone=BackboneConfig(arch="dinov2_vitb14", feature_dim=768))
        enc = ZeroShotDINOv2(cfg=cfg, mode="mean_patch").to(device).eval()
    else:
        raise ValueError(f"unknown backbone {backbone_name}")

    tf = default_transform()
    out: list[np.ndarray] = []
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i : i + batch_size]
        ts = []
        for r in batch_rows:
            with Image.open(cache_dir / r["image_path"]) as im:
                im = im.convert("RGB")
                im = _center_crop_pil(im, 224)
                ts.append(tf(im))
        x = torch.stack(ts).to(device)
        z = enc(x).cpu().numpy()
        out.append(z)
        if (i // batch_size) % 10 == 0:
            print(f"  {backbone_name} encode: {i + len(batch_rows)}/{len(rows)}", flush=True)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def encode_bert_titles(rows: list[dict], cache_dir: Path,
                       device: torch.device, source: str = "title",
                       batch_size: int = 32) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased",
                                      use_safetensors=True).to(device).eval()
    out: list[np.ndarray] = []
    for i in range(0, len(rows), batch_size):
        texts = [_read_text(r, cache_dir, source) for r in rows[i : i + batch_size]]
        enc = tok(texts, padding="max_length", truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        v = F.normalize(v, dim=-1)
        out.append(v.cpu().numpy())
    return np.concatenate(out, axis=0)


def info_nce(z_v: torch.Tensor, z_t: torch.Tensor, tau: float) -> torch.Tensor:
    z_v = F.normalize(z_v, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    logits = z_v @ z_t.T / tau
    labels = torch.arange(z_v.size(0), device=z_v.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def build_text_adapter(kind: str, d_in: int, d_out: int,
                       rank: int = 64, hidden_dim: int = 2048) -> nn.Module:
    if kind == "linear":
        return nn.Linear(d_in, d_out, bias=False)
    if kind == "low_rank":
        return nn.Sequential(
            nn.Linear(d_in, rank, bias=False),
            nn.Linear(rank, d_out, bias=False),
        )
    if kind == "bottleneck":
        return nn.Sequential(
            nn.Linear(d_in, rank),
            nn.GELU(),
            nn.Linear(rank, d_out),
        )
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_out),
        )
    if kind == "deep_mlp":
        return nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_out),
        )
    raise ValueError(f"unknown adapter kind {kind!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_adapter(adapter: nn.Module, x: torch.Tensor, y: torch.Tensor,
                  steps: int = 3000, batch: int = 256, lr: float = 1e-3,
                  tau: float = 0.07, weight_decay: float = 1e-4,
                  device: torch.device = torch.device("cpu")) -> dict:
    N = x.shape[0]
    adapter = adapter.to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    x = x.to(device)
    y = y.to(device)
    losses: list[float] = []
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), device=device)
        z_v = adapter(x[idx])
        z_t = y[idx]
        loss = info_nce(z_v, z_t, tau)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_f = float(loss.detach().cpu())
        losses.append(loss_f)
        if step % 200 == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}  loss {loss_f:.4f}", flush=True)
    return {
        "final": losses[-1] if losses else None,
        "min": min(losses) if losses else None,
        "steps": steps,
    }


def text_to_image_retrieval(z_v_eval: torch.Tensor, z_t_eval: torch.Tensor,
                            k_values: list[int]) -> dict:
    z_v = F.normalize(z_v_eval, dim=-1)
    z_t = F.normalize(z_t_eval, dim=-1)
    sim = z_t @ z_v.T
    Q, P = sim.shape
    max_k = min(max(k_values), P)
    correct = (sim.topk(max_k, dim=1).indices == torch.arange(Q).unsqueeze(1))
    out = {}
    for k in k_values:
        out[str(k)] = float(correct[:, :min(k, max_k)].any(dim=1).float().mean())
    diag = sim.diag()
    off = (sim.sum(dim=1) - diag) / max(1, sim.shape[1] - 1)
    return {
        "recall": out,
        "sanity": {
            "diag_sim_mean": float(diag.mean()),
            "off_sim_mean": float(off.mean()),
            "gap": float(diag.mean() - off.mean()),
        },
    }


def _cache_name(backbone: str, split: str, n: int, text_source: str, seed: int) -> str:
    return f"feats_{backbone}_{split}_n{n}_{text_source}_seed{seed}.npz"


def load_or_encode_features(args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_rows, eval_rows = _sample_pages(
        args.cache_dir, args.n_train, args.n_eval, args.seed)
    print(f"sampled: train={len(train_rows)}, eval={len(eval_rows)}", flush=True)

    cache_train = args.out_dir / _cache_name(args.backbone, "train", args.n_train, args.text_source, args.seed)
    cache_eval = args.out_dir / _cache_name(args.backbone, "eval", args.n_eval, args.text_source, args.seed)

    if cache_train.exists():
        d = np.load(cache_train)
        x_train, y_train = d["x"], d["y"]
        print(f"loaded train cache: x={x_train.shape}, y={y_train.shape}")
    else:
        t0 = time.time()
        x_train = encode_images(train_rows, args.cache_dir, args.backbone, device)
        print(f"  {args.backbone} train: {x_train.shape} in {time.time()-t0:.1f}s")
        t0 = time.time()
        y_train = encode_bert_titles(train_rows, args.cache_dir, device,
                                     source=args.text_source)
        print(f"  bert train:  {y_train.shape} in {time.time()-t0:.1f}s")
        np.savez(cache_train, x=x_train, y=y_train)

    if cache_eval.exists():
        d = np.load(cache_eval)
        x_eval, y_eval = d["x"], d["y"]
        print(f"loaded eval cache: x={x_eval.shape}, y={y_eval.shape}")
    else:
        t0 = time.time()
        x_eval = encode_images(eval_rows, args.cache_dir, args.backbone, device)
        print(f"  {args.backbone} eval: {x_eval.shape} in {time.time()-t0:.1f}s")
        t0 = time.time()
        y_eval = encode_bert_titles(eval_rows, args.cache_dir, device,
                                    source=args.text_source)
        print(f"  bert eval:  {y_eval.shape} in {time.time()-t0:.1f}s")
        np.savez(cache_eval, x=x_eval, y=y_eval)

    return x_train, y_train, x_eval, y_eval


def no_adapter_baseline(x_eval: np.ndarray, y_eval: np.ndarray) -> dict:
    d_t = y_eval.shape[1]
    if x_eval.shape[1] >= d_t:
        x_base = x_eval[:, :d_t]
    else:
        x_base = np.pad(x_eval, ((0, 0), (0, d_t - x_eval.shape[1])))
    return text_to_image_retrieval(
        torch.from_numpy(x_base.astype(np.float32)),
        torch.from_numpy(y_eval.astype(np.float32)),
        k_values=[1, 5, 10, 20])


def default_sweep_specs(hidden_dim: int) -> list[AdapterSpec]:
    return [
        AdapterSpec("linear"),
        AdapterSpec("low_rank", rank=16),
        AdapterSpec("low_rank", rank=64),
        AdapterSpec("low_rank", rank=256),
        AdapterSpec("bottleneck", rank=64),
        AdapterSpec("bottleneck", rank=256),
        AdapterSpec("mlp", hidden_dim=hidden_dim),
        AdapterSpec("deep_mlp", hidden_dim=hidden_dim),
    ]


def run_one_adapter(spec: AdapterSpec, x_train: np.ndarray, y_train: np.ndarray,
                    x_eval: np.ndarray, y_eval: np.ndarray, baseline: dict,
                    args: argparse.Namespace, device: torch.device) -> dict:
    d_in = int(x_train.shape[1])
    d_out = int(y_train.shape[1])
    rank = spec.rank if spec.rank is not None else args.rank
    hidden_dim = spec.hidden_dim if spec.hidden_dim is not None else args.hidden_dim
    adapter = build_text_adapter(
        spec.kind, d_in=d_in, d_out=d_out, rank=rank, hidden_dim=hidden_dim)
    param_count = count_parameters(adapter)
    print(f"\n=== training {spec.label} ({param_count:,} params) ===", flush=True)
    t0 = time.time()
    train_loss = train_adapter(
        adapter,
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
        steps=args.steps,
        batch=args.batch_size,
        lr=args.lr,
        tau=args.tau,
        weight_decay=args.weight_decay,
        device=device,
    )
    train_seconds = time.time() - t0
    
    adapter = adapter.to(device).eval()
    with torch.no_grad():
        z_v_adapted = adapter(torch.from_numpy(x_eval.astype(np.float32)).to(device)).cpu()
    adapted_score = text_to_image_retrieval(
        z_v_adapted,
        torch.from_numpy(y_eval.astype(np.float32)),
        k_values=[1, 5, 10, 20])
    print(json.dumps(adapted_score, indent=2), flush=True)

    report = {
        "adapter": {
            "kind": spec.kind,
            "label": spec.label,
            "parameter_count": param_count,
        },
        "backbone": args.backbone,
        "train_loss": train_loss,
        "adapted": adapted_score,
        "improvement_R10": adapted_score["recall"]["10"] - baseline["recall"]["10"],
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    p.add_argument("--backbone", choices=BACKBONES, default="ijepa")
    p.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "wiki_ss")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-source", choices=["title", "title_body"], default="title")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "runs" / "backbone_text_adapter_sweep")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_eval, y_eval = load_or_encode_features(args, device)
    base_score = no_adapter_baseline(x_eval, y_eval)
    
    specs = default_sweep_specs(args.hidden_dim) if args.sweep else [AdapterSpec("linear")]
    reports = [run_one_adapter(spec, x_train, y_train, x_eval, y_eval, base_score, args, device) for spec in specs]
    
    summary = {
        "backbone": args.backbone,
        "baseline": base_score,
        "adapters": reports
    }
    out_path = args.out_dir / f"summary_{args.backbone}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote summary to {out_path}")
    return 0


if __name__ == "__main__":
    main()

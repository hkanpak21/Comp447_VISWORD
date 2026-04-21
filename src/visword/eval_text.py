"""Text-only and cross-modal Phase-2 eval.

Three retrieval protocols on the anchor triplet set:

  * ``text_query_text_pool`` (rows 16–18, CLIP text-to-text): encode the
    anchor's text (title or title+snippet), encode each pool image's text,
    cosine retrieve.
  * ``image_query_text_pool`` (row 20, CLIP cross-modal): encode the
    anchor IMAGE with CLIP image branch, encode pool TEXTS with CLIP text
    branch, cosine retrieve in the shared CLIP embedding space.

All three read the anchor metadata (title + visible_text_snippet) from
``data/wiki_ss_anchors/metadata.jsonl``.

Writes ``phase2_recall.json`` in the same schema as ``eval_phase2`` so
rows tabulate side-by-side with image-only models.

Phase 1 is skipped for these rows: Phase 1 queries are small crops with
no text, so text retrieval is meaningless there.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

# Install DNS shim BEFORE importing transformers/sentence_transformers/open_clip
# — compute nodes SERVFAIL huggingface.co on the internal resolver.
from visword.hf_dns_shim import install as _install_dns_shim  # noqa: E402
_install_dns_shim()

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from visword.config import Config, config_hash
from visword.eval_phase2 import load_val_triplets
from visword.seed import seed_everything
from visword.train import resolve_config


def _load_anchor_metadata(anchors_cache_dir: Path) -> dict[str, dict]:
    """Return image_fname → metadata row (title, visible_text_snippet, etc.)."""
    meta = {}
    path = Path(anchors_cache_dir) / "metadata.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        meta[r["image_path"]] = r
    return meta


def _text_for(row: dict | None, mode: str = "title") -> str:
    if row is None:
        return ""
    title = (row.get("title", "") or "").replace("_", " ")
    if mode == "title":
        return title
    if mode == "title_body":
        snip = row.get("visible_text_snippet", "") or ""
        return (title + "\n" + snip)[:2000]
    return title


# ---------------------------------------------------------------------------
# Encoder builders
# ---------------------------------------------------------------------------


def _build_bert_encoder(device: torch.device, max_length: int = 64) -> Callable[[list[str]], torch.Tensor]:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    # use_safetensors forces safetensors checkpoint, avoiding a transformers
    # 4.57 + huggingface_hub 0.34 interop path that returns None for
    # checkpoint_files when the repo only publishes .safetensors.
    bert = AutoModel.from_pretrained("bert-base-uncased", use_safetensors=True).to(device).eval()

    @torch.no_grad()
    def encode(texts: list[str]) -> torch.Tensor:
        enc = tok(texts, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        out = bert(**enc).last_hidden_state           # (B, T, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(pooled, p=2, dim=-1).cpu()

    return encode


def _build_minilm_encoder(device: torch.device) -> Callable[[list[str]], torch.Tensor]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device).eval()

    @torch.no_grad()
    def encode(texts: list[str]) -> torch.Tensor:
        e = model.encode(texts, convert_to_tensor=True, device=str(device))
        return F.normalize(e, p=2, dim=-1).cpu()

    return encode


def _build_clip_encoders(device: torch.device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    @torch.no_grad()
    def encode_text(texts: list[str]) -> torch.Tensor:
        tokens = tokenizer(texts).to(device)
        e = model.encode_text(tokens).float()
        return F.normalize(e, p=2, dim=-1).cpu()

    @torch.no_grad()
    def encode_image(image_paths: list[Path]) -> torch.Tensor:
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
        x = torch.stack(imgs).to(device)
        e = model.encode_image(x).float()
        return F.normalize(e, p=2, dim=-1).cpu()

    return encode_text, encode_image


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _phase2_recall_from_sim(triplets, anchors_meta, anchor_emb_fn, pool_emb_fn,
                            anchors_root: Path, *, k_values, max_triplets,
                            text_mode: str, protocol: str):
    """Run Phase-2 retrieval given encode functions."""
    scores = {k: 0 for k in k_values}
    same_sim: list[float] = []
    diff_sim: list[float] = []
    n_valid = 0
    for t in triplets[:max_triplets or len(triplets)]:
        pos = list(t.get("positives", []))
        neg = list(t.get("negatives", []))
        anc = t.get("anchor", "")
        if not pos or not neg:
            continue
        # For image_query_text_pool we need anchor image to exist on disk
        if protocol == "image_query_text_pool":
            if not (anchors_root / "images" / anc).exists():
                continue

        a = anchor_emb_fn(t)
        if a is None:
            continue
        p_embeds = pool_emb_fn(pos + neg)
        if p_embeds is None or p_embeds.shape[0] < 2:
            continue
        n_valid += 1
        sim = (a @ p_embeds.T).squeeze(0)
        n_pos = len(pos)
        for k in k_values:
            topk = sim.topk(min(k, len(sim))).indices
            if any(i < n_pos for i in topk.tolist()):
                scores[k] += 1
        same_sim.extend(sim[:n_pos].tolist())
        diff_sim.extend(sim[n_pos:].tolist())
    for k in scores:
        scores[k] /= max(n_valid, 1)
    return {
        "num_triplets": n_valid,
        "num_anchors": len(triplets),
        "recall": {str(k): float(scores[k]) for k in k_values},
        "protocol": protocol,
        "text_mode": text_mode,
        "sanity": {
            "same_sim_mean": float(np.mean(same_sim)) if same_sim else 0.0,
            "diff_sim_mean": float(np.mean(diff_sim)) if diff_sim else 0.0,
            "gap": float(np.mean(same_sim) - np.mean(diff_sim)) if same_sim and diff_sim else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--run-name", required=True)
    p.add_argument("--encoder", required=True,
                   choices=["bert_title", "bert_title_body", "minilm", "clip_text",
                            "clip_cross_modal"])
    p.add_argument("--max-triplets", type=int, default=None)
    args = p.parse_args(argv)

    cfg = resolve_config(args.config, [])
    seed_everything(cfg.train.seed)
    anchors_root = Path(cfg.data.anchors_cache_dir)
    anchors_meta = _load_anchor_metadata(anchors_root)
    triplets = load_val_triplets(anchors_root)

    project_root = Path(__file__).resolve().parents[2]
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    run_dir = project_root / "runs" / f"{ts}_textzs_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    (run_dir / "provenance.json").write_text(json.dumps({
        "run_name": args.run_name, "encoder": args.encoder, "zeroshot": True,
        "config_hash": config_hash(cfg),
    }, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_values = list(cfg.eval.k_values)

    if args.encoder.startswith("bert") or args.encoder == "minilm":
        if args.encoder == "bert_title":
            encode_text = _build_bert_encoder(device); text_mode = "title"
        elif args.encoder == "bert_title_body":
            encode_text = _build_bert_encoder(device, max_length=256); text_mode = "title_body"
        else:
            encode_text = _build_minilm_encoder(device); text_mode = "title"

        def anchor_fn(t): return encode_text([_text_for(anchors_meta.get(t["anchor"]), text_mode)])
        def pool_fn(paths): return encode_text([_text_for(anchors_meta.get(p), text_mode) for p in paths])
        protocol = "text_query_text_pool"

    elif args.encoder == "clip_text":
        encode_text, _ = _build_clip_encoders(device); text_mode = "title"
        def anchor_fn(t): return encode_text([_text_for(anchors_meta.get(t["anchor"]), text_mode)])
        def pool_fn(paths): return encode_text([_text_for(anchors_meta.get(p), text_mode) for p in paths])
        protocol = "text_query_text_pool"

    elif args.encoder == "clip_cross_modal":
        encode_text, encode_image = _build_clip_encoders(device); text_mode = "title"
        def anchor_fn(t):
            return encode_image([anchors_root / "images" / t["anchor"]])
        def pool_fn(paths):
            return encode_text([_text_for(anchors_meta.get(p), text_mode) for p in paths])
        protocol = "image_query_text_pool"

    else:
        raise SystemExit(f"unknown encoder {args.encoder}")

    res = _phase2_recall_from_sim(
        triplets, anchors_meta, anchor_fn, pool_fn, anchors_root,
        k_values=k_values, max_triplets=args.max_triplets,
        text_mode=text_mode, protocol=protocol,
    )
    (run_dir / "phase2_recall.json").write_text(json.dumps(res, indent=2))
    sys.stdout.write(f"{args.run_name}: P2 R@1={res['recall']['1']:.3f}  "
                     f"R@10={res['recall']['10']:.3f}  n={res['num_triplets']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

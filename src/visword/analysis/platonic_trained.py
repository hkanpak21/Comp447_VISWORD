"""Extended Platonic alignment: include our TRAINED SALAD-main and CLS-main
checkpoints alongside the frozen encoders. Answers the question:

  Does contrastive fine-tuning on Wikipedia screenshots push DINOv2's image
  features AWAY from text-encoder spaces (consistent with layout-fingerprinting)
  or TOWARD them (consistent with acquiring linguistic structure)?

Same 500-sample protocol as `platonic_alignment.py`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from visword.hf_dns_shim import install as _install_dns_shim
_install_dns_shim()

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from visword.analysis.platonic_alignment import (
    cka, procrustes_distance, mutual_knn,
    encode_bert_title, encode_minilm_title, encode_clip_text,
    encode_dinov2, encode_clip_image,
)
from visword.config import Config
from visword.data.light_dataset import default_transform


def _load_config_and_ckpt(run_dir: Path) -> tuple[Config, dict]:
    cfg_dict = yaml.safe_load(open(run_dir / "config.resolved.yaml"))
    cfg = Config(**cfg_dict)
    ckpt = torch.load(run_dir / "checkpoints" / "best_phase1.pt", map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
    return cfg, state


def _build_trained_model(cfg: Config, state: dict, device):
    if cfg.model_kind == "salad":
        from visword.models.dinov2_salad import DINOv2SALAD
        model = DINOv2SALAD(cfg)
    elif cfg.model_kind == "cls":
        from visword.models.dinov2_cls import DINOv2CLS
        model = DINOv2CLS(cfg)
    elif cfg.model_kind == "linear_probe":
        from visword.models.zeroshot import DINOv2LinearProbe
        model = DINOv2LinearProbe(cfg)
    else:
        raise SystemExit(f"unsupported trained model_kind={cfg.model_kind!r}")
    model.load_state_dict(state)
    return model.to(device).eval()


def encode_trained(run_dir: Path, image_paths, device, batch_size=4):
    cfg, state = _load_config_and_ckpt(run_dir)
    model = _build_trained_model(cfg, state, device)
    tf = default_transform()
    embeds = []
    for i in range(0, len(image_paths), batch_size):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in image_paths[i:i+batch_size]]).to(device)
        with torch.no_grad():
            e = model(batch).float()
        embeds.append(e.cpu().numpy())
    return np.concatenate(embeds)


def main() -> int:
    # Same sample as platonic_alignment.py
    cache_dir = Path("/scratch/hkanpak21/VISWORD/data/wiki_ss")
    manifest = json.load(open(cache_dir / "manifest.json"))
    rows = manifest["rows"]
    np.random.seed(42)
    import os
    n_samples = int(os.environ.get("PLATONIC_N", "500"))
    picks = np.random.choice(len(rows), min(n_samples, len(rows)), replace=False)
    sampled = [rows[i] for i in picks]
    img_paths = [cache_dir / r["image_path"] for r in sampled]
    titles = [(r["title"] or "").replace("_", " ") or "untitled" for r in sampled]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, n=500")

    # Trained checkpoints (final versions at 10k train / 3 epochs — salad-main, cls-main)
    salad_main_dir = Path("/scratch/hkanpak21/VISWORD/runs/2026-04-19_201330_324858ad_salad-main_138f")
    cls_main_dir   = Path("/scratch/hkanpak21/VISWORD/runs/2026-04-19_213919_324858ad_cls-main_db61")
    linear_probe_dir = Path("/scratch/hkanpak21/VISWORD/runs/2026-04-21_195057_6d06122e_row07-linear-probe_385b")

    encoders: dict[str, np.ndarray] = {}
    t0 = time.time()

    print("encoding SALAD-main (trained)...", flush=True)
    encoders["salad_main"] = encode_trained(salad_main_dir, img_paths, device)
    print(f"  {time.time()-t0:.0f}s  dim={encoders['salad_main'].shape[1]}")

    print("encoding CLS-main (trained)...", flush=True)
    encoders["cls_main"] = encode_trained(cls_main_dir, img_paths, device)
    print(f"  {time.time()-t0:.0f}s  dim={encoders['cls_main'].shape[1]}")

    print("encoding linear probe (trained)...", flush=True)
    encoders["linear_probe"] = encode_trained(linear_probe_dir, img_paths, device)
    print(f"  {time.time()-t0:.0f}s  dim={encoders['linear_probe'].shape[1]}")

    print("encoding DINOv2 zero-shot (reference)...", flush=True)
    encoders["dinov2_zeroshot"] = encode_dinov2(img_paths, device)
    print(f"  {time.time()-t0:.0f}s")

    print("encoding CLIP image (reference)...", flush=True)
    encoders["clip_image"] = encode_clip_image(img_paths, device)
    print(f"  {time.time()-t0:.0f}s")

    print("encoding BERT titles...", flush=True)
    encoders["bert"] = encode_bert_title(titles, device)
    print(f"  {time.time()-t0:.0f}s")

    print("encoding MiniLM titles...", flush=True)
    encoders["minilm"] = encode_minilm_title(titles, device)
    print(f"  {time.time()-t0:.0f}s")

    print("encoding CLIP text...", flush=True)
    encoders["clip_text"] = encode_clip_text(titles, device)
    print(f"  {time.time()-t0:.0f}s")

    # Compute alignment matrix
    names = list(encoders.keys())
    print(f"\n=== Pairwise alignment (n=500) ===")
    results = {"n_samples": len(sampled), "encoders": {n: {"dim": encoders[n].shape[1]} for n in names},
               "pairs": {}}

    for i, a in enumerate(names):
        for b in names[i+1:]:
            cka_v = float(cka(encoders[a], encoders[b], debiased=True))
            proc = procrustes_distance(encoders[a], encoders[b])
            knn_10 = mutual_knn(encoders[a], encoders[b], k=10)
            results["pairs"][f"{a}__{b}"] = {
                "cka_debiased": cka_v,
                "procrustes": proc,
                "mutual_knn_10": knn_10,
            }
            print(f"  {a:18s} ↔ {b:18s}: CKA={cka_v:+.3f}  Procrustes={proc:.3f}  knn@10={knn_10:.3f}")

    out_dir = Path(f"/scratch/hkanpak21/VISWORD/runs/platonic_trained_{time.strftime('%Y-%m-%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(results, indent=2))
    print(f"\n→ {out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

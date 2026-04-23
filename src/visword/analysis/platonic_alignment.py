"""Platonic Representation Hypothesis test (paper/study.md §7.2).

Encodes the same set of pages with frozen vision + text encoders and
reports alignment metrics:
  - debiased HSIC / CKA (Murphy et al., ICLR 2024 Re-Align Workshop)
  - Procrustes distance
  - mutual-kNN overlap (Huh et al., ICML 2024 Position Oral)

Output: runs/platonic_alignment_<TS>/report.json

CPU-only; each encoder runs on whatever device is available.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from visword.hf_dns_shim import install as _install_dns_shim
_install_dns_shim()

import numpy as np
import torch
import torch.nn.functional as F


def hsic_biased(K, L):
    """Biased HSIC estimator on centered Gram matrices."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return (Kc * Lc).sum() / (n - 1) ** 2


def hsic_unbiased(K, L):
    """Debiased HSIC (Song et al. 2007). Removes diagonal bias."""
    n = K.shape[0]
    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))
    term1 = (K * L).sum()
    term2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
    term3 = 2 * (K.sum(0) @ L.sum(0)) / (n - 2)
    return (term1 + term2 - term3) / (n * (n - 3))


def cka(X, Y, debiased=True):
    """Linear CKA(X, Y). X, Y are (n, d) feature matrices."""
    K = X @ X.T
    L = Y @ Y.T
    f = hsic_unbiased if debiased else hsic_biased
    return f(K, L) / np.sqrt(f(K, K) * f(L, L))


def procrustes_distance(X, Y):
    """Procrustes (orthogonal-invariant) distance between X and Y after
    mean-centering and unit-Frobenius normalisation. Returns in [0, 2].
    """
    X = X - X.mean(0, keepdims=True); X = X / np.linalg.norm(X)
    Y = Y - Y.mean(0, keepdims=True); Y = Y / np.linalg.norm(Y)
    # Optimal rotation R = argmin || X R - Y ||_F via SVD of X.T @ Y
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    R = U @ Vt
    return float(np.linalg.norm(X @ R - Y))


def mutual_knn(X, Y, k=10):
    """Fraction of shared nearest neighbours in the top-k between X and Y."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    Sx = Xn @ Xn.T
    Sy = Yn @ Yn.T
    np.fill_diagonal(Sx, -np.inf)
    np.fill_diagonal(Sy, -np.inf)
    topk_x = np.argpartition(-Sx, k, axis=1)[:, :k]
    topk_y = np.argpartition(-Sy, k, axis=1)[:, :k]
    overlap = [len(set(a) & set(b)) / k for a, b in zip(topk_x, topk_y)]
    return float(np.mean(overlap))


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def encode_dinov2(paths, device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    from PIL import Image
    embeds = []
    for i in range(0, len(paths), 16):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+16]]).to(device)
        with torch.no_grad():
            e = F.normalize(model(batch).float(), p=2, dim=-1)
        embeds.append(e.cpu().numpy())
    return np.concatenate(embeds)


def encode_clip_image(paths, device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.to(device).eval()
    from PIL import Image
    embeds = []
    for i in range(0, len(paths), 16):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+16]]).to(device)
        with torch.no_grad():
            e = F.normalize(model.encode_image(batch).float(), p=2, dim=-1)
        embeds.append(e.cpu().numpy())
    return np.concatenate(embeds)


def encode_bert_title(titles, device):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased", use_safetensors=True).to(device).eval()
    embeds = []
    for i in range(0, len(titles), 32):
        enc = tok(titles[i:i+32], padding=True, truncation=True, max_length=64,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = bert(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1)
        embeds.append(F.normalize(pooled, p=2, dim=-1).cpu().numpy())
    return np.concatenate(embeds)


def encode_minilm_title(titles, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device).eval()
    with torch.no_grad():
        e = model.encode(titles, convert_to_tensor=True, device=str(device), batch_size=32)
    return F.normalize(e, p=2, dim=-1).cpu().numpy()


def encode_clip_text(titles, device):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    embeds = []
    for i in range(0, len(titles), 32):
        toks = tokenizer(titles[i:i+32]).to(device)
        with torch.no_grad():
            e = F.normalize(model.encode_text(toks).float(), p=2, dim=-1)
        embeds.append(e.cpu().numpy())
    return np.concatenate(embeds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path,
                        default="/scratch/hkanpak21/VISWORD/data/wiki_ss")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--encoders", nargs="+",
                        default=["dinov2", "clip_image", "bert", "minilm", "clip_text"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load a stratified sample of the cache
    manifest = json.load(open(args.cache_dir / "manifest.json"))
    rows = manifest["rows"]
    np.random.seed(42)
    picks = np.random.choice(len(rows), min(args.n_samples, len(rows)), replace=False)
    sampled = [rows[i] for i in picks]
    img_paths = [args.cache_dir / r["image_path"] for r in sampled]
    titles = [(r["title"] or "").replace("_", " ") or "untitled" for r in sampled]
    print(f"sampled {len(sampled)} pages")

    # Encode
    encoders = {}
    t0 = time.time()
    if "dinov2" in args.encoders:
        print("encoding DINOv2...", flush=True)
        encoders["dinov2"] = encode_dinov2(img_paths, device); print(f"  {time.time()-t0:.0f}s")
    if "clip_image" in args.encoders:
        print("encoding CLIP image...", flush=True)
        encoders["clip_image"] = encode_clip_image(img_paths, device); print(f"  {time.time()-t0:.0f}s")
    if "bert" in args.encoders:
        print("encoding BERT title...", flush=True)
        encoders["bert"] = encode_bert_title(titles, device); print(f"  {time.time()-t0:.0f}s")
    if "minilm" in args.encoders:
        print("encoding MiniLM title...", flush=True)
        encoders["minilm"] = encode_minilm_title(titles, device); print(f"  {time.time()-t0:.0f}s")
    if "clip_text" in args.encoders:
        print("encoding CLIP text...", flush=True)
        encoders["clip_text"] = encode_clip_text(titles, device); print(f"  {time.time()-t0:.0f}s")

    # Compute alignment matrix
    names = list(encoders.keys())
    print(f"\nComputing alignment across {len(names)} encoders on n={len(sampled)}")
    results = {"n_samples": len(sampled), "encoders": {}, "pairs": {}}
    for n in names:
        results["encoders"][n] = {"dim": encoders[n].shape[1]}

    for i, a in enumerate(names):
        for b in names[i+1:]:
            cka_v = float(cka(encoders[a], encoders[b], debiased=True))
            proc = procrustes_distance(encoders[a], encoders[b])
            knn_5 = mutual_knn(encoders[a], encoders[b], k=5)
            knn_10 = mutual_knn(encoders[a], encoders[b], k=10)
            results["pairs"][f"{a}__{b}"] = {
                "cka_debiased": cka_v,
                "procrustes": proc,
                "mutual_knn_5": knn_5,
                "mutual_knn_10": knn_10,
            }
            print(f"  {a:12s} ↔ {b:12s}: CKA={cka_v:.3f}  Procrustes={proc:.3f}  "
                  f"knn@5={knn_5:.3f}  knn@10={knn_10:.3f}")

    out_dir = Path(f"/scratch/hkanpak21/VISWORD/runs/platonic_alignment_{time.strftime('%Y-%m-%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(results, indent=2))
    print(f"\n→ {out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

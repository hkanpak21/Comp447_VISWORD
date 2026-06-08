"""One-time: convert microsoft/dit-base's pytorch_model.bin -> model.safetensors in the
HF cache. DiT ships only the old .bin format, which this transformers version refuses to
torch.load unless torch>=2.6 (the offline env is older). We load the already-downloaded
weights and re-save them as safetensors so AutoModel.from_pretrained works. Run once on a
node with the model cached (login node, after prefetch_doc_models.py microsoft/dit-base).
"""
import glob
import os

import torch
from safetensors.torch import save_file


def main() -> int:
    snaps = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--dit-base/snapshots/*"))
    if not snaps:
        print("dit-base not in cache — run scripts/prefetch_doc_models.py microsoft/dit-base first")
        return 1
    snap = snaps[0]
    binp, stp = os.path.join(snap, "pytorch_model.bin"), os.path.join(snap, "model.safetensors")
    if os.path.exists(stp):
        print("already converted:", stp)
        return 0
    sd = torch.load(binp, map_location="cpu", weights_only=True)
    clean = {k: v.clone().contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    save_file(clean, stp, metadata={"format": "pt"})
    print(f"converted {len(clean)} tensors -> {stp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Dedicated I-JEPA cross-modal (Text-Target) pre-training script."""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from visword.config import Config
from visword.data import manifest as M
from visword.data.cropper import NonOverlappingCropper
from visword.data.light_dataset import LightWikiScreenshotDataset
from visword.eval_phase1 import phase1_recall
from visword.models.ijepa_masks import MaskCollator
from visword.models.ijepa_predictor import apply_masks
from visword.models.ijepa_text_predictor import VisionTransformerTextPredictor
from visword.paths import PROJECT_ROOT, expand_env
from visword.reporting.jsonl_logger import JsonlLogger, read_jsonl
from visword.reporting.plots import plot_train_curves
from visword.reporting.run_dir import create_run_dir
from visword.seed import seed_everything


def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size()
    return 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _split_indices(
    num_rows: int, num_train: int, num_eval: int, seed: int
) -> tuple[list[int], list[int]]:
    if num_train + num_eval > num_rows:
        raise SystemExit(
            f"data.num_train_samples + data.num_eval_samples = {num_train + num_eval} "
            f"exceeds cache size {num_rows}. Re-run prefetch with a larger target."
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_rows)
    return perm[:num_train].tolist(), perm[num_train : num_train + num_eval].tolist()


class IJepaEvalWrapper(nn.Module):
    """Wraps context encoder for standard Phase-1/Phase-2 retrieval evaluation."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.model = encoder

    @property
    def descriptor_dim(self) -> int:
        return 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        feats = out.last_hidden_state.mean(dim=1).float()
        return F.normalize(feats, p=2, dim=-1)


class ContextEncoderWrapper(nn.Module):
    """Wraps context encoder to perform masking inside forward() for DDP."""
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, imgs: torch.Tensor, masks_enc: list[torch.Tensor]) -> torch.Tensor:
        embs = self.encoder.embeddings(imgs, interpolate_pos_encoding=True)
        x_ctxt = apply_masks(embs, masks_enc)
        enc_out = self.encoder.encoder(x_ctxt)
        z = self.encoder.layernorm(enc_out.last_hidden_state)
        return z


class JepaTextCollator:
    def __init__(self, mask_collator: MaskCollator) -> None:
        self.mask_collator = mask_collator

    def __call__(self, batch: list[tuple[torch.Tensor, str, int]]) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        images = [item[0].squeeze(0) for item in batch]
        texts = [item[1] for item in batch]
        collated_images, masks_enc, _ = self.mask_collator(images)
        return collated_images, texts, masks_enc


def _save_ckpt(
    path: Path,
    encoder: nn.Module,
    predictor: nn.Module,
    *,
    step: int,
    recall10: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "step": step,
            "phase1_recall@10": recall10,
        },
        path,
    )


def _eval_and_log(
    eval_model: nn.Module,
    eval_ds: LightWikiScreenshotDataset,
    cfg: Config,
    logger: JsonlLogger,
    step: int,
    device: torch.device,
) -> float:
    eval_model.eval()
    result = phase1_recall(eval_model, eval_ds, k_values=cfg.eval.k_values, device=device)
    row: dict[str, Any] = {
        "eval_step": step,
        "phase1_num_crops": result["num_crops"],
        "phase1_num_pages": result["num_pages"],
        "phase1_sanity_gap": result["sanity"]["gap"],
        "phase1_monotonic": result["sanity"]["monotonic"],
    }
    for k, v in result["recall"].items():
        row[f"phase1_recall@{k}"] = v
    logger.log(row)
    return float(result["recall"].get("10", 0.0))


def _lr_schedule(total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(round(warmup_ratio * total_steps)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return lr_lambda


def main(argv: list[str] | None = None) -> int:
    local_rank, world_size = setup_ddp()
    is_main_process = (local_rank == 0)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--run-name", default=None, type=str)
    p.add_argument("--runs-root", default=None, type=Path,
                   help="override runs/ location (used by tests).")
    args = p.parse_args(argv)

    # Resolve config
    from visword.train import resolve_config as _resolve_config
    cfg = _resolve_config(args.config, args.overrides)
    seed_everything(cfg.train.seed)

    # ---- Data cache sanity checks --------
    cache_dir = Path(cfg.data.wiki_ss_cache_dir)

    if is_main_process:
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"ERROR: data cache is empty or missing at {cache_dir}.\n"
                f"Run scripts/prefetch_data.py first."
            )
        if not M.verify_fingerprint(cache_dir):
            raise SystemExit(
                f"ERROR: data cache fingerprint mismatch at {cache_dir}. Re-run prefetch."
            )

    if dist.is_initialized():
        dist.barrier()

    fingerprint = (cache_dir / ".fingerprint").read_text().strip()

    # ---- Run directory --------
    run_dir = None
    if is_main_process:
        run_dir = create_run_dir(
            cfg,
            run_name=args.run_name,
            data_fingerprint=fingerprint,
            runs_root=args.runs_root,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data split & datasets --------
    manifest = M.read_manifest(cache_dir)
    train_idx, eval_idx = _split_indices(
        manifest["num_rows"],
        cfg.data.num_train_samples,
        cfg.data.num_eval_samples,
        cfg.train.seed,
    )

    cropper = NonOverlappingCropper(
        crop_size=cfg.cropper.crop_size,
        overlap=cfg.cropper.overlap,
        min_text_ratio=cfg.cropper.min_text_ratio,
        target_size=cfg.cropper.target_size,
    )
    # return_text=True, text_source from config
    train_ds = LightWikiScreenshotDataset(
        cache_dir,
        indices=train_idx,
        cropper=cropper,
        k_per_page=1,
        seed=cfg.train.seed,
        return_text=True,
        text_source=cfg.ijepa.text_source,
    )

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True) if world_size > 1 else None

    eval_ds = None
    if is_main_process:
        eval_cropper = NonOverlappingCropper(
            crop_size=cfg.cropper.crop_size,
            overlap=cfg.cropper.overlap,
            min_text_ratio=0.0,  # keep all for eval
            target_size=cfg.cropper.target_size,
        )
        eval_ds = LightWikiScreenshotDataset(
            cache_dir,
            indices=eval_idx,
            cropper=eval_cropper,
            k_per_page=2,
            seed=cfg.train.seed + 1,
            return_text=False,
        )

    # Setup context mask collator
    mask_collator = MaskCollator(
        input_size=(cfg.cropper.target_size, cfg.cropper.target_size),
        patch_size=14,  # ViT-H/14
        enc_mask_scale=cfg.ijepa.enc_mask_scale,
        pred_mask_scale=cfg.ijepa.pred_mask_scale,
        aspect_ratio=cfg.ijepa.aspect_ratio,
        nenc=cfg.ijepa.num_enc_masks,
        npred=1,  # unused but required by MaskCollator signature
        allow_overlap=cfg.ijepa.allow_overlap,
        min_keep=cfg.ijepa.min_keep,
    )
    collator = JepaTextCollator(mask_collator)

    loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        drop_last=True,
    )

    # ---- Build Model components --------
    # Frozen-except-last-N Context Encoder
    from visword.models.ijepa_backbone import IJepaBackbone
    backbone = IJepaBackbone(num_trainable_blocks=cfg.backbone.num_trainable_blocks)
    base_encoder = backbone.model.to(device)
    base_encoder.gradient_checkpointing_enable()

    # Frozen BERT text encoder and tokenizer
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased", use_safetensors=True).to(device)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False

    # Vision-to-Text Predictor
    num_patches = (cfg.cropper.target_size // 14) ** 2
    predictor = VisionTransformerTextPredictor(
        num_patches=num_patches,
        max_text_tokens=cfg.ijepa.max_text_tokens,
        embed_dim=1280,
        predictor_embed_dim=cfg.ijepa.pred_emb_dim,
        depth=cfg.ijepa.pred_depth,
        num_heads=12,
        target_dim=768,  # BERT representations
    ).to(device)

    context_encoder = ContextEncoderWrapper(base_encoder)

    if dist.is_initialized():
        context_encoder = DDP(context_encoder, device_ids=[local_rank], find_unused_parameters=True)
        predictor = DDP(predictor, device_ids=[local_rank], find_unused_parameters=False)

    # Setup Optimizer & schedulers
    groups = [
        {"params": [p for p in context_encoder.parameters() if p.requires_grad], "lr": cfg.train.lr_backbone, "name": "backbone"},
        {"params": list(predictor.parameters()), "lr": cfg.train.lr_head, "name": "predictor"},
    ]
    optim = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)

    total_steps = max(1, len(loader) * cfg.train.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, _lr_schedule(total_steps, cfg.train.warmup_ratio)
    )

    # Logger
    logger = JsonlLogger(run_dir / "metrics.jsonl") if is_main_process else None
    best_r10 = -1.0
    global_step = 0
    t_start = time.time()

    # Evaluator model wrapper (evaluates vision encoder only)
    eval_model = IJepaEvalWrapper(base_encoder) if is_main_process else None

    try:
        for epoch in range(cfg.train.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            context_encoder.train()
            predictor.train()

            for imgs, texts, masks_enc in loader:
                mask_collator.step()

                # Move visual inputs to device
                imgs = imgs.to(device, non_blocking=True)
                masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
                B = len(imgs)

                # 1. Forward through frozen BERT to get linguistic targets
                inputs = tokenizer(
                    texts,
                    padding="max_length",
                    max_length=cfg.ijepa.max_text_tokens,
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    bert_out = bert(**inputs)
                    # (B, T, 768)
                    true_bert_embeds = bert_out.last_hidden_state

                # 2. Context Vision Encoder forward pass (grads on trainable blocks via DDP)
                z = context_encoder(imgs, masks_enc)

                # 3. Predictor forward pass
                # Outputs predicted text embeddings: (B * nenc, T, 768)
                pred_embeds = predictor(z, masks_enc, target_len=cfg.ijepa.max_text_tokens)

                # 4. Compute masked Smooth L1 Loss against true BERT embeddings
                # Repeat target representations for each context mask
                nenc = len(masks_enc)
                targets = true_bert_embeds.repeat(nenc, 1, 1)
                attn_mask = inputs["attention_mask"].repeat(nenc, 1)  # (B * nenc, T)

                # Smooth L1 loss
                raw_loss = F.smooth_l1_loss(pred_embeds, targets, reduction="none")  # (B * nenc, T, 768)
                # Apply attention mask to ignore padding tokens
                mask = attn_mask.unsqueeze(-1)  # (B * nenc, T, 1)
                loss = (raw_loss * mask).sum() / (mask.sum() * targets.size(-1) + 1e-8)

                # Backward and optimization
                optim.zero_grad(set_to_none=True)
                loss.backward()

                if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                    trainable_params = [p for p in context_encoder.parameters() if p.requires_grad] + list(predictor.parameters())
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.train.grad_clip)

                optim.step()
                scheduler.step()

                # Log metrics on main process
                if is_main_process:
                    row = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": float(loss.detach().cpu()),
                        "lr_bb": float(optim.param_groups[0]["lr"]),
                        "lr_pred": float(optim.param_groups[1]["lr"]),
                        "gpu_mem_gb": round(torch.cuda.memory_allocated() / (1024**3), 3) if torch.cuda.is_available() else 0.0,
                        "wall_time_s": round(time.time() - t_start, 2),
                    }
                    logger.log(row)

                    sys.stdout.write(
                        f"\rstep {global_step}/{total_steps}  loss {float(loss):.4f}  "
                        f"lr_bb {optim.param_groups[0]['lr']:.2e}  lr_pred {optim.param_groups[1]['lr']:.2e}"
                    )
                    sys.stdout.flush()

                    # Periodic evaluation
                    if (cfg.train.eval_every_steps > 0
                            and global_step > 0
                            and global_step % cfg.train.eval_every_steps == 0):
                        r10 = _eval_and_log(eval_model, eval_ds, cfg, logger, global_step, device)
                        if r10 > best_r10:
                            best_r10 = r10
                            _save_ckpt(
                                run_dir / "checkpoints" / "best_phase1.pt",
                                base_encoder, predictor.module if isinstance(predictor, DDP) else predictor,
                                step=global_step, recall10=r10,
                            )
                        context_encoder.train()
                        predictor.train()

                global_step += 1

        if is_main_process:
            # End of training evaluation
            r10 = _eval_and_log(eval_model, eval_ds, cfg, logger, global_step, device)
            if r10 > best_r10:
                best_r10 = r10
                _save_ckpt(
                    run_dir / "checkpoints" / "best_phase1.pt",
                    base_encoder, predictor.module if isinstance(predictor, DDP) else predictor,
                    step=global_step, recall10=r10,
                )
            _save_ckpt(
                run_dir / "checkpoints" / "last.pt",
                base_encoder, predictor.module if isinstance(predictor, DDP) else predictor,
                step=global_step, recall10=r10,
            )

    finally:
        if is_main_process:
            logger.close()
        cleanup_ddp()

    if is_main_process:
        rows = read_jsonl(run_dir / "metrics.jsonl")
        plot_train_curves(rows, run_dir / "train_curves.png")
        sys.stdout.write("\n")
        sys.stdout.write(f"run_dir: {run_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

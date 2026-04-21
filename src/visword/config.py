"""Pydantic v2 config models + canonical hashing.

Schema mirrors PROJECT_SPEC.md §3.1 exactly. Two runs with the same canonical
hash are guaranteed to share identical resolved config.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    wiki_ss_cache_dir: Path
    anchors_cache_dir: Path
    num_train_samples: int = 10_000
    num_eval_samples: int = 1_000


class CropperConfig(_Strict):
    crop_size: int = 490
    overlap: float = 0.0
    min_text_ratio: float = 0.05
    target_size: int = 224


class BackboneConfig(_Strict):
    arch: Literal["dinov2_vitb14", "dinov2_vits14"] = "dinov2_vitb14"
    num_trainable_blocks: int = 4
    feature_dim: int = 768


class SaladConfig(_Strict):
    num_clusters: int = 64
    cluster_dim: int = 128
    token_dim: int = 256
    sinkhorn_iters: int = 3
    ablation: Literal["full", "token_only", "vlad_only", "softmax_assign"] = "full"


class TrainConfig(_Strict):
    loss: Literal["infonce", "multisim", "triplet"] = "multisim"
    temperature: float = 0.07
    k_per_page: int = 4
    batch_size: int = 32
    epochs: int = 3
    lr_backbone: float = 1e-5
    lr_head: float = 5e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42
    eval_every_steps: int = 100
    num_diag_batches: int = 3


class EvalConfig(_Strict):
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 20])
    phase1_max_pages: int = 500
    phase2_max_queries: int = 200


class Config(_Strict):
    experiment_name: str
    model_kind: Literal[
        "cls", "salad",
        "linear_probe",
        "zeroshot_dinov2_cls", "zeroshot_dinov2_mean",
        "zeroshot_clip_image", "zeroshot_imagenet_vit",
    ] = "salad"
    data: DataConfig
    cropper: CropperConfig = Field(default_factory=CropperConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    salad: SaladConfig = Field(default_factory=SaladConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def canonical_dump(cfg: Config | dict[str, Any]) -> str:
    """Canonical JSON of a config: sorted keys, no whitespace, JSON-safe types."""
    if isinstance(cfg, Config):
        data = cfg.model_dump(mode="json")
    else:
        data = _to_jsonable(cfg)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def config_hash(cfg: Config | dict[str, Any]) -> str:
    """SHA-1 of the canonical JSON dump (spec §3.3)."""
    return hashlib.sha1(canonical_dump(cfg).encode("utf-8")).hexdigest()

"""Run-directory layout + provenance capture (PROJECT_SPEC.md §11).

The invariant we want: ``runs/<id>/`` alone is enough to reconstruct the
experiment — config, code SHAs, env, data cache fingerprint, metrics,
plots and checkpoints all live under it. No hidden global state.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from visword.config import Config, config_hash
from visword.paths import PROJECT_ROOT


RUNS_ROOT = PROJECT_ROOT / "runs"


# ---------------------------------------------------------------------------
# git + SALAD SHA helpers
# ---------------------------------------------------------------------------


def _run_git(args: list[str], cwd: Path) -> str | None:
    """Return ``git <args>`` output in ``cwd``, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(cwd), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _visword_sha() -> str:
    return _run_git(["rev-parse", "HEAD"], PROJECT_ROOT) or "unknown"


def _salad_sha() -> str:
    """Parse the SALAD commit SHA out of ``third_party/salad/SETUP.md``."""
    setup = PROJECT_ROOT / "third_party" / "salad" / "SETUP.md"
    if not setup.exists():
        return "unknown"
    text = setup.read_text()
    m = re.search(r"`([0-9a-f]{40})`", text)
    return m.group(1) if m else "unknown"


def _salad_module_file_hash() -> str:
    """SHA-256 of the file that defines OfficialSALAD — detects hot-patching."""
    from visword.models.salad_bridge import OfficialSALAD
    try:
        path = Path(sys.modules[OfficialSALAD.__module__].__file__).resolve()
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Slug / ID helpers
# ---------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(s: str, max_len: int = 32) -> str:
    s = _SLUG_RE.sub("-", s.lower()).strip("-")
    return (s or "run")[:max_len]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def run_dir_name(cfg: Config, *, run_name: str | None = None) -> str:
    """``<ts>_<visword_sha8>_<slug>_<cfg_hash4>/``"""
    slug = slugify(run_name or cfg.experiment_name)
    return f"{_utc_timestamp()}_{_visword_sha()[:8]}_{slug}_{config_hash(cfg)[:4]}"


# ---------------------------------------------------------------------------
# Provenance + run dir creation
# ---------------------------------------------------------------------------


def _gpu_info() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def _torch_info() -> tuple[str, str]:
    try:
        import torch
        return torch.__version__, torch.version.cuda or "n/a"
    except Exception:
        return "unknown", "n/a"


def build_provenance(cfg: Config, *, data_fingerprint: str | None = None) -> dict:
    torch_ver, cuda_ver = _torch_info()
    return {
        "visword_git_sha": _visword_sha(),
        "salad_vendor_sha": _salad_sha(),
        "salad_sha_in_bridge": _salad_module_file_hash(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_ver,
        "cuda": cuda_ver,
        "gpu": _gpu_info(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "config_hash": config_hash(cfg),
        "data_fingerprint": data_fingerprint or "",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def create_run_dir(
    cfg: Config,
    *,
    run_name: str | None = None,
    data_fingerprint: str | None = None,
    runs_root: Path | None = None,
) -> Path:
    """Create ``<runs_root>/<run_dir_name>/`` with config + provenance."""
    root = Path(runs_root) if runs_root else RUNS_ROOT
    root.mkdir(parents=True, exist_ok=True)

    run_dir = root / run_dir_name(cfg, run_name=run_name)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    (run_dir / "interpret").mkdir(parents=True, exist_ok=True)

    # Resolved config (with Path-types turned into strings).
    with (run_dir / "config.resolved.yaml").open("w") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=True)

    # Provenance — everything needed to reproduce this run.
    (run_dir / "provenance.json").write_text(
        json.dumps(build_provenance(cfg, data_fingerprint=data_fingerprint), indent=2)
    )
    return run_dir


__all__ = [
    "RUNS_ROOT",
    "build_provenance",
    "config_hash",
    "create_run_dir",
    "run_dir_name",
    "slugify",
]

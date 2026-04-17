"""Single source of truth for the official SALAD aggregator and DINOv2 backbone.

Per PROJECT_SPEC.md §5.2: this module manipulates ``sys.path`` so the vendored
``third_party/salad/`` package becomes importable. Do NOT import
``models.aggregators.salad`` or ``models.backbones.dinov2`` from anywhere else
in the project — go through this bridge so there is exactly one path.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SALAD_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "salad"
if not _SALAD_ROOT.exists():
    raise RuntimeError(
        f"third_party/salad not found at {_SALAD_ROOT}. "
        f"Run scripts/vendor_salad.sh to vendor it."
    )
if str(_SALAD_ROOT) not in sys.path:
    sys.path.insert(0, str(_SALAD_ROOT))

# Now safe to import — the vendored package's `models/` is on sys.path.
from models.aggregators.salad import SALAD as OfficialSALAD  # noqa: E402
from models.backbones.dinov2 import DINOv2 as OfficialDINOv2  # noqa: E402

try:
    from models.aggregators.salad import log_otp_solver  # noqa: E402
except ImportError:
    log_otp_solver = None

__all__ = ["OfficialSALAD", "OfficialDINOv2", "log_otp_solver"]

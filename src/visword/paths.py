"""Central path resolution and ${VAR} substitution for YAML configs."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def expand_env(value: Any, env: dict[str, str] | None = None) -> Any:
    """Recursively replace ``${VAR}`` substrings in ``value`` from ``env`` / os.environ."""
    if env is None:
        env = dict(os.environ)
        env.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
        env.setdefault("DATA_DIR", str(PROJECT_ROOT / "data"))

    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            if key not in env:
                raise KeyError(f"Unresolved variable ${{{key}}} in config string")
            return env[key]

        return _VAR_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, env) for v in value]
    return value

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coerce(value: str) -> Any:
    """Turn a command-line string into bool/int/float/null/str."""
    s = value.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``["window.mirror=false", "plots.hz=5"]`` style overrides to a copy of ``cfg``."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key=value, got {ov!r}")
        path, val = ov.split("=", 1)
        parts = [p for p in path.strip().split(".") if p]
        if not parts:
            raise ValueError(f"empty key in override {ov!r}")
        cur = cfg
        for p in parts[:-1]:
            if not isinstance(cur.get(p), dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = coerce(val)
    return cfg

from __future__ import annotations
import json, copy
from typing import Any, Dict, List

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def _coerce(value: str) -> Any:
    # crude coercion for -o overrides
    s = value.strip()
    if s.lower() in ("true","false"):
        return s.lower() == "true"
    try:
        if "." in s: return float(s)
        return int(s)
    except Exception:
        return s

def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    overrides like ["window.mirror=false", "plots.hz=5", "asr.enabled=false"]
    """
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov: continue
        path, val = ov.split("=", 1)
        parts = [p for p in path.strip().split(".") if p]
        cur = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = _coerce(val)
    return cfg

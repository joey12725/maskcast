from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple


def now() -> float:
    return time.time()


def pin_to_core(core: int, log: logging.Logger) -> None:
    """Pin the calling thread to one CPU (Linux only; a no-op elsewhere)."""
    try:
        cores = sorted(os.sched_getaffinity(0))
        if not cores:
            return
        core_id = cores[min(core, len(cores) - 1)]
        os.sched_setaffinity(threading.get_native_id(), {core_id})
        log.info("Pinned to CPU %d", core_id)
    except Exception:
        pass


def parse_window_arg(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse ``"640x480"`` or ``"640,480"`` into a (w, h) tuple."""
    if not s:
        return None
    for sep in ("x", ","):
        if sep in s.lower():
            try:
                w, h = s.lower().split(sep)
                return int(w), int(h)
            except ValueError:
                return None
    return None


def fit_within(w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int]:
    """Scale (w, h) down to fit max_w x max_h, keeping aspect ratio."""
    if w <= max_w and h <= max_h:
        return w, h
    scale = min(max_w / max(1, w), max_h / max(1, h))
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))

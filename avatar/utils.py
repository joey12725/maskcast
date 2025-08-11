from __future__ import annotations
import time, logging
from typing import Optional, Tuple
from pathlib import Path
from .optional_deps import cv2

def now() -> float:
    return time.time()

def parse_window_arg(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s: return None
    try:
        if "x" in s.lower():
            w, h = s.lower().split("x"); return int(w), int(h)
        if "," in s:
            w, h = s.split(","); return int(w), int(h)
    except Exception:
        return None
    return None

def autodetect_camera_size(cam_index: int, log: logging.Logger) -> tuple[int, int]:
    if cv2 is None:
        return (640, 480)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    ok, frame = cap.read()
    if ok and frame is not None:
        h, w = frame.shape[:2]
    cap.release()
    log.info("Auto camera size: %dx%d", w, h)
    return w, h

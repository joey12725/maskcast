"""Synthetic tracker for ``--demo``: no camera, no models, same render path.

It writes the exact state keys the real VideoWorker and Reporter write, so the
renderer, WebRTC stream and LED mask output all behave as they would live.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .utils import now

# (label, valence, arousal, seconds). Keyword flashes are prefixed with "!".
DEFAULT_SCRIPT: Sequence[Tuple[str, float, float, float]] = (
    ("neutral", 0.0, 0.0, 2.0),
    ("happy", 0.4, 0.1, 1.6),
    ("happy", 0.8, 0.8, 1.4),
    ("surprise", 0.1, 0.8, 1.4),
    ("!amazing", 0.0, 0.0, 1.2),
    ("sad", -0.6, -0.3, 1.6),
    ("angry", -0.7, 0.7, 1.6),
    ("fear", -0.3, 0.6, 1.3),
    ("disgust", -0.4, 0.2, 1.3),
    ("contempt", -0.2, 0.0, 1.3),
    ("!adorable", 0.0, 0.0, 1.2),
    ("!wink", 0.0, 0.0, 1.0),
    ("!sleepy", 0.0, 0.0, 1.2),
)


def script_at(t: float, script: Sequence[Tuple[str, float, float, float]] = DEFAULT_SCRIPT):
    """Return the script entry active at ``t`` seconds (the script loops)."""
    total = sum(s[3] for s in script)
    t = t % total
    for entry in script:
        if t < entry[3]:
            return entry
        t -= entry[3]
    return script[-1]


def face_path(t: float) -> Tuple[float, float]:
    """A Lissajous wander so the gaze has something to follow."""
    return 0.8 * math.sin(t * 0.9), 0.5 * math.sin(t * 1.7 + 0.6)


class DemoDriver(threading.Thread):
    def __init__(
        self, state: Dict[str, Any], hz: float = 30.0, script: Optional[List[Tuple[str, float, float, float]]] = None
    ):
        super().__init__(daemon=True, name="Demo")
        self.state = state
        self.dt = 1.0 / hz
        self.script = script or list(DEFAULT_SCRIPT)
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        t0 = now()
        while not self._halt.is_set():
            t = now() - t0
            label, v, a, _ = script_at(t, self.script)
            nx, ny = face_path(t)
            # Drop the face briefly every ~11s to show hold-and-return.
            has_face = (t % 11.0) < 9.5
            upd: Dict[str, Any] = {"has_face": has_face, "valence": v, "arousal": a, "frame_size": (640, 480)}
            if has_face:
                upd.update(face_norm=(nx, ny), last_seen_face_norm=(nx, ny), last_face_ts=now())
            if label.startswith("!"):
                upd.update(flash_label=label[1:], flash_until=now() + 0.2)
            else:
                upd.update(emotion_label=label, emotion_probs={label: 1.0})
            self.state.update(upd)
            self._halt.wait(self.dt)

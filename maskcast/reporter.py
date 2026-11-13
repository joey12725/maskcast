from __future__ import annotations

import logging
import threading
import time
from typing import Any, List, Optional

from .expressions import canon, match_keyword
from .tuning import Tuning
from .utils import now


class Reporter(threading.Thread):
    """Watches the transcript for keywords, logs a status line and feeds the plotter."""

    def __init__(
        self,
        state: dict,
        asr: Any,
        log: logging.Logger,
        tuning: Tuning,
        q: Optional[object] = None,
        hz: float = 2.0,
        keywords: Optional[List[str]] = None,
    ):
        super().__init__(daemon=True, name="Reporter")
        self.state = state
        self.asr = asr
        self.log = log.getChild("Reporter")
        self.q = q
        self.dt = 1.0 / max(0.1, hz)
        self.running = True
        self.tuning = tuning
        self.keywords = keywords or []

    def _check_keywords(self, window: str) -> None:
        kw = match_keyword(window, self.keywords)
        if kw is None:
            return
        # Only (re)arm the flash if it has expired or the keyword changed
        if self.state.get("flash_label") != kw or now() > float(self.state.get("flash_until", 0.0)):
            self.state["flash_label"] = kw
            self.state["flash_until"] = now() + float(self.tuning.keyword_flash_s)

    def run(self):
        while self.running:
            s = self.state
            emo = canon(s.get("emotion_label", "neutral"))
            v, a = float(s.get("valence", 0.0)), float(s.get("arousal", 0.0))
            pos_ms = float(s.get("pos_latency_ms", 0.0))
            emo_ms = float(s.get("emotion_latency_ms", 0.0))
            rtc_ms = float(s.get("remote_latency_ms", 0.0))
            asr_ms = float(getattr(self.asr, "last_latency_ms", 0.0)) if self.asr else 0.0
            mic_db = float(getattr(self.asr, "last_rms_dbfs", -120.0)) if self.asr else -120.0
            window = str(getattr(self.asr, "window_text_value", "")) if self.asr else ""
            if window:
                self._check_keywords(window)
            nx, ny = s.get("face_norm", (0.0, 0.0))

            self.log.info(
                "face=%s emotion=%s v=%.2f a=%.2f | latency pos=%.0fms emo=%.0fms asr=%.0fms rtc=%.0fms | "
                "mic=%.1f dBFS | heard=%r",
                "yes" if s.get("has_face") else "no",
                emo,
                v,
                a,
                pos_ms,
                emo_ms,
                asr_ms,
                rtc_ms,
                mic_db,
                window,
            )

            if self.q is not None:
                try:
                    self.q.put_nowait(
                        {
                            "ts": now(),
                            "valence": v,
                            "arousal": a,
                            "lat_pos": pos_ms,
                            "lat_emo": emo_ms,
                            "lat_asr": asr_ms,
                            "lat_rtc": rtc_ms,
                            "nx": float(nx),
                            "ny": float(ny),
                            "probs": s.get("emotion_probs", {}),
                            "va_fallback": bool(s.get("va_fallback_active", False)),
                            "text": window,
                        }
                    )
                except Exception:
                    pass
            time.sleep(self.dt)

    def stop(self):
        self.running = False

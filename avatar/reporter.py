from __future__ import annotations
import threading, time, logging
from typing import Optional, Any, Dict, List
from .tuning import canon, KEYWORD_ALIASES, Tuning
from .utils import now

class Reporter(threading.Thread):
    def __init__(self, state: dict, asr: Any, log: logging.Logger, tuning: Tuning,
                 q: Optional[object] = None, hz: float = 2.0, keywords: Optional[List[str]] = None):
        super().__init__(daemon=True, name="Reporter")
        self.state = state
        self.asr = asr
        self.log = log.getChild("Reporter")
        self.q = q
        self.dt = 1.0 / max(0.1, hz)
        self.running = True
        self.tuning = tuning
        self.keywords = keywords or []

    def run(self):
        while self.running:
            fc = self.state.get("face_center", (None, None))
            has_face = self.state.get("has_face", False)
            emo = canon(self.state.get("emotion_label", "neutral"))
            v = float(self.state.get("valence", 0.0))
            a = float(self.state.get("arousal", 0.0))
            pos_ms = float(self.state.get("pos_latency_ms", 0.0))
            emo_ms = float(self.state.get("emotion_latency_ms", 0.0))
            rtc_ms = float(self.state.get("remote_latency_ms", 0.0))
            trans_ms = float(getattr(self.asr, "last_latency_ms", 0.0)) if self.asr else 0.0
            mic_db = float(getattr(self.asr, "last_rms_dbfs", -120.0)) if self.asr else -120.0
            window = str(getattr(self.asr, "window_text_value", "")) if self.asr else ""
            window_low = window.lower()
            if window_low:
                for kw in self.keywords:
                    aliases = KEYWORD_ALIASES.get(kw, [])
                    if any(a in window_low for a in aliases):
                        self.state["flash_label"] = kw
                        self.state["flash_until"] = now() + float(getattr(self.tuning, "keyword_flash_s", 0.0))
                        break
            nx, ny = self.state.get("face_norm", (0.0, 0.0))
            probs = self.state.get("emotion_probs", {})

            print(
                f"position ({fc[0]}, {fc[1]}), emotion: {emo}, valence: {v:.2f}, arousal: {a:.2f}, "
                f"{'face detected' if has_face else 'no face detected'}, "
                f"current transcription window: \"{window}\", "
                f"emotion latency: {emo_ms:.0f}ms, pos latency: {pos_ms:.0f}ms, rtc latency: {rtc_ms:.0f}ms, transcription latency: {trans_ms:.0f}ms, "
                f"mic level: {mic_db:.1f} dBFS {'[OK]' if mic_db > -55 else '[low]'}"
            )

            if self.q is not None:
                try:
                    self.q.put_nowait({
                        "ts": now(),
                        "valence": v, "arousal": a,
                        "lat_pos": pos_ms, "lat_emo": emo_ms, "lat_asr": trans_ms, "lat_rtc": rtc_ms,
                        "nx": float(nx), "ny": float(ny),
                        "probs": probs,
                        "text": window,
                    })
                except Exception:
                    pass
            time.sleep(self.dt)

    def stop(self):
        self.running = False

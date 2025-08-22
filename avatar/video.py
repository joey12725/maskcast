from __future__ import annotations
import threading, logging, time
from collections import deque
from typing import Optional, Tuple, Dict, Any
from queue import Queue
from .optional_deps import cv2
from .detectors import FaceDetector
from .emotions import EmotionEngine
from .tuning import Tuning, canon
from .utils import now, pin_to_core

class VideoWorker(threading.Thread):
    def __init__(self, cam_index: int, tuning: Tuning, state: Dict[str, Any], log: logging.Logger,
                 detector_mode: str, emotion_backend: str, mirror: bool, frame_queue: Optional[Queue] = None):
        super().__init__(daemon=True, name="Video")
        self.cam_index = cam_index
        self.tuning = tuning
        self.state = state
        self.log = log.getChild("Video")
        self.detector = FaceDetector(detector_mode, self.log)
        self.emotion = EmotionEngine(emotion_backend, self.log)
        self.cap = None
        self.stop_flag = False
        self._prev_norm = (0.0, 0.0)
        self._prev_ts = now()
        self.mirror = mirror
        self.frame_queue = frame_queue

    def stop(self):
        self.stop_flag = True

    def run(self):
        pin_to_core(1, self.log)
        if self.frame_queue is None:
            if cv2 is None:
                raise RuntimeError("OpenCV not installed")
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.cam_index}")
            self.log.info("Video capture opened (index=%d)", self.cam_index)

        fps_t0, frames = now(), 0
        while not self.stop_flag:
            pos_t0 = now()
            if self.frame_queue is None:
                ok, bgr = self.cap.read()
                if not ok or bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                try:
                    item = self.frame_queue.get(timeout=0.5)
                except Exception:
                    continue
                rgb = item[0] if isinstance(item, tuple) else item
            # Cap resolution to fit within 720x1080 while preserving aspect ratio
            H, W = rgb.shape[:2]
            if (W > 720 or H > 1080) and cv2 is not None:
                scale = min(720 / max(1, W), 1080 / max(1, H))
                new_w = max(1, int(round(W * scale)))
                new_h = max(1, int(round(H * scale)))
                try:
                    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    H, W = rgb.shape[:2]
                except Exception:
                    pass
            self.state["frame_size"] = (W, H)

            faces = self.detector.detect(rgb)
            main = max(faces, key=lambda f: f[2] * f[3]) if faces else None

            if main:
                x, y, w, h, score = main
                cx = x + w / 2
                cy = y + h / 2
                nx = (cx - W / 2) / (W / 2)
                ny = (cy - H / 2) / (H / 2)
                if self.mirror:
                    nx = -nx

                ts = now(); dt = max(1e-3, ts - self._prev_ts)
                vx = (nx - self._prev_norm[0]) / dt
                vy = (ny - self._prev_norm[1]) / dt
                self._prev_norm, self._prev_ts = (nx, ny), ts

                last_emo_ts = self.state.get("last_emo_ts", 0.0)
                if now() - last_emo_ts >= 1.0 / max(0.1, self.tuning.emotion_hz):
                    label, probs, v, a, emo_latency_ms = self.emotion.predict(rgb, (x, y, w, h))
                    probs = {canon(k): float(vv) for k, vv in probs.items()}
                    self.state["emotion_probs"] = self._ema_probs(self.state.get("emotion_probs", {}), probs, self.tuning.emotion_smooth_alpha)
                    final_label = canon(max(self.state["emotion_probs"].items(), key=lambda kv: kv[1])[0])
                    # Fallback VA: if recognizer didn't provide VA and we have non-trivial probs, derive from probs
                    va_fallback = False
                    if abs(float(v)) < 1e-6 and abs(float(a)) < 1e-6:
                        nonzero_probs = [p for p in self.state["emotion_probs"].values() if p > 1e-3]
                        if len(nonzero_probs) >= 2:
                            happy_p = float(self.state["emotion_probs"].get("happy", 0.0))
                            sad_p = float(self.state["emotion_probs"].get("sad", 0.0))
                            v = happy_p - sad_p
                            # Relative certainty: top-1 minus top-2 probability
                            try:
                                sorted_probs = sorted(self.state["emotion_probs"].values(), reverse=True)
                                a = float(sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0))
                            except Exception:
                                a = 0.0
                            va_fallback = True
                        else:
                            va_fallback = False
                    else:
                        va_fallback = False
                    self.state.update({
                        "emotion_label": final_label,
                        "valence": float(v), "arousal": float(a),
                        "last_emo_ts": now(),
                        "emotion_latency_ms": float(emo_latency_ms),
                        "va_fallback_active": bool(va_fallback),
                    })

                self.state.update({
                    "has_face": True,
                    "face_bbox": main,
                    "face_center": (int(cx), int(cy)),
                    "face_norm": (float(nx), float(ny)),
                    "last_seen_face_norm": (float(nx), float(ny)),
                    "face_norm_vel": (float(vx), float(vy)),
                    "last_face_ts": now(),
                })
            else:
                self.state["has_face"] = False
                self.state["face_norm_vel"] = (0.0, 0.0)

            self.state["pos_latency_ms"] = (now() - pos_t0) * 1000.0

            frames += 1
            if now() - fps_t0 >= 2.0:
                fps = frames / max(1e-6, (now() - fps_t0))
                self.log.info("~%.1f FPS | face=%s | W×H=%s", fps, "yes" if self.state.get("has_face") else "no", self.state.get("frame_size"))
                fps_t0, frames = now(), 0

        if self.cap is not None:
            self.cap.release()
            self.log.info("Video capture closed.")

    @staticmethod
    def _ema_probs(old: dict, new: dict, alpha: float) -> dict:
        keys = set(list(old.keys()) + list(new.keys()))
        out = {}
        for k in keys:
            o = old.get(k, 0.0); n = new.get(k, 0.0)
            out[k] = alpha * n + (1 - alpha) * o
        return out

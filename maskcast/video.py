from __future__ import annotations

import logging
import threading
from queue import Queue
from typing import Any, Dict, Optional

from .detectors import FaceDetector
from .emotions import EmotionEngine
from .expressions import canon, derive_valence_arousal, ema_probs, top_label
from .optional_deps import cv2
from .tuning import Tuning
from .utils import fit_within, now, pin_to_core


class VideoWorker(threading.Thread):
    def __init__(
        self,
        cam_index: int,
        tuning: Tuning,
        state: Dict[str, Any],
        log: logging.Logger,
        detector_mode: str,
        emotion_backend: str,
        mirror: bool,
        frame_queue: Optional[Queue] = None,
    ):
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
            new_w, new_h = fit_within(W, H, 720, 1080)
            if (new_w, new_h) != (W, H) and cv2 is not None:
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                H, W = rgb.shape[:2]
            self.state["frame_size"] = (W, H)

            faces = [f for f in self.detector.detect(rgb) if f[4] >= self.tuning.min_face_conf]
            main = max(faces, key=lambda f: f[2] * f[3]) if faces else None

            if main:
                x, y, w, h, score = main
                cx = x + w / 2
                cy = y + h / 2
                nx = (cx - W / 2) / (W / 2)
                ny = (cy - H / 2) / (H / 2)
                if self.mirror:
                    nx = -nx

                ts = now()
                dt = max(1e-3, ts - self._prev_ts)
                vx = (nx - self._prev_norm[0]) / dt
                vy = (ny - self._prev_norm[1]) / dt
                self._prev_norm, self._prev_ts = (nx, ny), ts

                last_emo_ts = self.state.get("last_emo_ts", 0.0)
                if now() - last_emo_ts >= 1.0 / max(0.1, self.tuning.emotion_hz):
                    _, probs, v, a, emo_latency_ms = self.emotion.predict(rgb, (x, y, w, h))
                    probs = {canon(k): float(vv) for k, vv in probs.items()}
                    smoothed = ema_probs(self.state.get("emotion_probs", {}), probs, self.tuning.emotion_smooth_alpha)
                    self.state["emotion_probs"] = smoothed
                    final_label = top_label(smoothed)
                    # No VA head output: estimate valence/arousal from the class probabilities
                    va_fallback = False
                    if abs(float(v)) < 1e-6 and abs(float(a)) < 1e-6:
                        derived = derive_valence_arousal(smoothed)
                        if derived is not None:
                            v, a = derived
                            va_fallback = True
                    self.state.update(
                        {
                            "emotion_label": final_label,
                            "valence": float(v),
                            "arousal": float(a),
                            "last_emo_ts": now(),
                            "emotion_latency_ms": float(emo_latency_ms),
                            "va_fallback_active": bool(va_fallback),
                        }
                    )

                self.state.update(
                    {
                        "has_face": True,
                        "face_bbox": main,
                        "face_center": (int(cx), int(cy)),
                        "face_norm": (float(nx), float(ny)),
                        "last_seen_face_norm": (float(nx), float(ny)),
                        "face_norm_vel": (float(vx), float(vy)),
                        "last_face_ts": now(),
                    }
                )
            else:
                self.state["has_face"] = False
                self.state["face_norm_vel"] = (0.0, 0.0)

            self.state["pos_latency_ms"] = (now() - pos_t0) * 1000.0

            frames += 1
            if now() - fps_t0 >= 2.0:
                fps = frames / max(1e-6, (now() - fps_t0))
                self.log.info(
                    "~%.1f FPS | face=%s | W×H=%s",
                    fps,
                    "yes" if self.state.get("has_face") else "no",
                    self.state.get("frame_size"),
                )
                fps_t0, frames = now(), 0

        if self.cap is not None:
            self.cap.release()
            self.log.info("Video capture closed.")

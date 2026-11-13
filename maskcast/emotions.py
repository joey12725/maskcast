from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from .expressions import CANONICAL, canon
from .hw import pick_torch_device
from .optional_deps import EmotiEffLibRecognizerBase, EmotiEffLibRecognizerTorch, torch
from .utils import now


class EmotionEngine:
    def __init__(self, backend: str, log: logging.Logger):
        self.log = log.getChild("Emotion")
        self.available = False
        self.device = "cpu"
        self.recognizer: Optional[EmotiEffLibRecognizerBase] = None
        self.labels = list(CANONICAL)
        if backend != "emotieff":
            self.log.info("Emotion disabled (backend=%s)", backend)
            return
        if EmotiEffLibRecognizerTorch is None:
            self.log.warning("EmotiEffLib Torch not available; neutral fallback.")
            return
        try:
            dev = pick_torch_device()
            self.device = dev

            # Handle PyTorch 2.6+ weights_only issue for EmotiEffLib
            if torch is not None:
                # Import the required class and add it to safe globals
                try:
                    from timm.models.efficientnet import EfficientNet

                    torch.serialization.add_safe_globals([EfficientNet])
                except ImportError:
                    pass  # Fall back to weights_only=False approach

                # Also patch torch.load as a fallback for any direct calls
                _orig_torch_load = torch.load

                def _patched_load(*args, **kwargs):
                    if "weights_only" not in kwargs:
                        kwargs["weights_only"] = False
                    return _orig_torch_load(*args, **kwargs)

                torch.load = _patched_load
                try:
                    self.recognizer = EmotiEffLibRecognizerTorch("enet_b0_8_va_mtl", device=dev)
                finally:
                    torch.load = _orig_torch_load
            else:
                self.recognizer = EmotiEffLibRecognizerTorch("enet_b0_8_va_mtl", device=dev)
            if hasattr(self.recognizer, "idx_to_emotion_class"):
                raw = [
                    self.recognizer.idx_to_emotion_class[i] for i in sorted(self.recognizer.idx_to_emotion_class.keys())
                ]  # type: ignore
                self.labels = sorted({canon(s) for s in raw if canon(s)})
            self.available = True
            self.log.info("Loaded EmotiEffLib 'enet_b0_8_va_mtl' on %s", dev)
        except Exception as e:
            self.log.warning("EmotiEff init failed: %s (neutral fallback)", e)

    def emotion_names(self):
        return list(sorted(set(self.labels) | set(CANONICAL)))

    def predict(self, frame_rgb: Any, bbox_xywh: Optional[Tuple[int, int, int, int]]):
        t0 = now()
        if not self.available or bbox_xywh is None:
            return "neutral", {"neutral": 1.0}, 0.0, 0.0, (now() - t0) * 1000.0
        x, y, w, h = bbox_xywh
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(frame_rgb.shape[1], x + w)
        y1 = min(frame_rgb.shape[0], y + h)
        face = frame_rgb[y0:y1, x0:x1]
        if face.size == 0:
            return "neutral", {"neutral": 1.0}, 0.0, 0.0, (now() - t0) * 1000.0
        try:
            preds, scores = self.recognizer.predict_emotions(face, logits=False)  # type: ignore
            probs_vec = scores[0]
            v, a = 0.0, 0.0
            if probs_vec.shape[-1] >= 10:
                v = float(probs_vec[-2])
                a = float(probs_vec[-1])
                probs_vec = probs_vec[:-2]
            if hasattr(self.recognizer, "idx_to_emotion_class"):
                idxmap = self.recognizer.idx_to_emotion_class  # type: ignore
                probs_map_raw = {str(idxmap[i]).lower(): float(probs_vec[i]) for i in range(len(probs_vec))}
            else:
                probs_map_raw = {}
            probs = {}
            for k, vprob in probs_map_raw.items():
                ck = canon(k)
                probs[ck] = max(vprob, probs.get(ck, 0.0))
            label = canon(preds[0])
            if not label:
                label = "neutral"
            return label, probs, v, a, (now() - t0) * 1000.0
        except Exception as e:
            self.log.debug("Emotion prediction failed: %s", e)
            return "neutral", {"neutral": 1.0}, 0.0, 0.0, (now() - t0) * 1000.0

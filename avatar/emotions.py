from __future__ import annotations
import logging
from typing import Optional, Tuple, Dict, Any
from .optional_deps import torch, EmotiEffLibRecognizerBase, EmotiEffLibRecognizerTorch
from .tuning import CANONICAL, canon
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
            dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            self.device = dev
            self.recognizer = EmotiEffLibRecognizerTorch("enet_b0_8_va_mtl", device=dev)
            if hasattr(self.recognizer, "idx_to_emotion_class"):
                raw = [self.recognizer.idx_to_emotion_class[i] for i in sorted(self.recognizer.idx_to_emotion_class.keys())]  # type: ignore
                self.labels = sorted({canon(s) for s in raw if canon(s)})
            self.available = True
            self.log.info("Loaded EmotiEffLib 'enet_b0_8_va_mtl' on %s", dev)
        except Exception as e:
            self.log.warning("EmotiEff init failed: %s (neutral fallback)", e)

    def emotion_names(self):
        return list(sorted(set(self.labels) | set(CANONICAL)))

    def predict(self, frame_rgb: Any, bbox_xywh: Optional[Tuple[int,int,int,int]]):
        t0 = now()
        if not self.available or bbox_xywh is None:
            return "neutral", {"neutral": 1.0}, 0.0, 0.0, (now() - t0) * 1000.0
        x, y, w, h = bbox_xywh
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(frame_rgb.shape[1], x + w); y1 = min(frame_rgb.shape[0], y + h)
        face = frame_rgb[y0:y1, x0:x1]
        if face.size == 0:
            return "neutral", {"neutral": 1.0}, 0.0, 0.0, (now() - t0) * 1000.0
        try:
            preds, scores = self.recognizer.predict_emotions(face, logits=False)  # type: ignore
            probs_vec = scores[0]
            v, a = 0.0, 0.0
            if probs_vec.shape[-1] >= 10:
                v = float(probs_vec[-2]); a = float(probs_vec[-1])
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
        except Exception:
            return "neutral", {"neutral": 1.0}, 0.0, 0.0, (now() - t0) * 1000.0

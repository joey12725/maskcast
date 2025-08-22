from __future__ import annotations
import logging, os
from typing import List, Tuple, Any
from .optional_deps import cv2
from .hw import onnxruntime_providers

class FaceDetector:
    """
    - mode='insightface' tries to use InsightFace SCRFD via onnxruntime (CUDA/CPU).
    - fallback is OpenCV Haar cascade.
    """
    def __init__(self, mode: str, log: logging.Logger):
        self.mode = mode
        self.log = log.getChild("FaceDetector")
        self.impl = None
        self.is_gpu = False
        if mode == "insightface":
            self._init_insightface()
        else:
            self._init_cv2_haar()

    def _init_insightface(self):
        try:
            import onnxruntime as ort  # noqa
            import insightface as IF
        except Exception as e:
            self.log.warning("InsightFace unavailable (%s); falling back to Haar.", e)
            return self._init_cv2_haar()
        providers = onnxruntime_providers()
        try:
            fa = IF.app.FaceAnalysis(name='buffalo_l', providers=providers)
            fa.prepare(ctx_id=0, det_size=(640, 640))
            try:
                sess_prov = fa.models['detection'].session.get_providers()  # type: ignore
            except Exception:
                sess_prov = []
            self.is_gpu = any(('CUDAExecutionProvider' in str(p)) or ('DmlExecutionProvider' in str(p)) or ('CoreMLExecutionProvider' in str(p)) or ('MPSExecutionProvider' in str(p)) or ('ROCMExecutionProvider' in str(p)) for p in sess_prov)
            self.impl = fa
            self.log.info("Using InsightFace SCRFD (%s) providers=%s", "GPU" if self.is_gpu else "CPU", ",".join(map(str, providers)))
        except Exception as e:
            self.log.warning("InsightFace init failed (%s); falling back to Haar.", e)
            return self._init_cv2_haar()

    def _init_cv2_haar(self):
        if cv2 is None:
            raise RuntimeError("OpenCV not installed; cannot use Haar fallback.")
        haar_path = None
        try:
            base = getattr(cv2, "data").haarcascades  # type: ignore[attr-defined]
            haar_path = base + "haarcascade_frontalface_default.xml"
        except Exception:
            pass
        if not haar_path or not os.path.isfile(haar_path):
            for p in (
                "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            ):
                if os.path.isfile(p):
                    haar_path = p
                    break
        if not haar_path or not os.path.isfile(haar_path):
            raise RuntimeError("OpenCV Haar cascade not found; please install OpenCV data files.")
        self.haar = cv2.CascadeClassifier(haar_path)
        self.impl = "haar"
        self.log.warning("Using Haar cascade fallback: %s", haar_path)

    def detect(self, rgb: Any) -> List[Tuple[int,int,int,int,float]]:
        faces = []
        if isinstance(self.impl, str) and self.impl == "haar":
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            dets = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            for (x, y, ww, hh) in dets:
                faces.append((int(x), int(y), int(ww), int(hh), 0.9))
            return faces

        # InsightFace
        try:
            res = self.impl.get(rgb)  # type: ignore
            for d in res or []:
                x1, y1, x2, y2 = map(int, d.bbox)
                faces.append((x1, y1, x2 - x1, y2 - y1, float(d.det_score)))
        except Exception:
            pass
        return faces

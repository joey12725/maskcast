from __future__ import annotations

import logging
import threading
import time
from collections import deque
from queue import Queue
from typing import Optional

from .hw import faster_whisper_device_and_compute
from .optional_deps import WhisperModel, np, sd
from .utils import now, pin_to_core


class WhisperASR(threading.Thread):
    def __init__(
        self, model_name: str, log: logging.Logger, device_preference: str = "auto", pcm_queue: Optional[Queue] = None
    ):
        super().__init__(daemon=True, name="ASR-Whisper")
        self.log = log.getChild("ASR")
        self.model_name = model_name
        self.running = False
        self.last_latency_ms = 0.0
        self.last_rms_dbfs = -120.0
        self.window_text_value = ""
        self.device_preference = device_preference
        self.pcm_queue = pcm_queue
        self.available = WhisperModel is not None and np is not None and (sd is not None or self.pcm_queue is not None)

    def window_text(self) -> str:
        return self.window_text_value

    def _pick_device_and_compute(self):
        if self.device_preference == "cpu":
            return "cpu", "int8"
        return faster_whisper_device_and_compute("auto")

    def run(self):
        pin_to_core(2, self.log)
        if not self.available:
            self.log.warning("Whisper not available (sounddevice/faster-whisper/numpy missing)")
            return
        device, compute_type = self._pick_device_and_compute()
        try:
            model = WhisperModel(self.model_name, device=device, compute_type=compute_type)  # type: ignore
            self.log.info("faster-whisper model=%s on %s (%s)", self.model_name, device.upper(), compute_type)
        except Exception as e:
            self.log.warning("Failed to load faster-whisper: %s", e)
            return

        sr = 16000
        block = int(sr * 0.5)
        ring = deque(maxlen=sr * 6)  # 6s window
        self.running = True

        if self.pcm_queue is not None:
            self.log.info("ASR queue mode started (%.1f kHz).", sr / 1000)
            while self.running:
                try:
                    pcm = self.pcm_queue.get(timeout=0.1)
                except Exception:
                    continue
                pcm = np.array(pcm, dtype=np.float32)
                rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0  # type: ignore
                self.last_rms_dbfs = -120.0 if rms <= 1e-9 else 20.0 * np.log10(max(1e-3, rms) / 32768.0)  # type: ignore
                ring.extend(pcm)
                if len(ring) >= ring.maxlen and self.last_rms_dbfs > -55.0:
                    wav = np.array(ring, dtype=np.float32) / 32768.0  # type: ignore
                    t1 = now()
                    try:
                        segments, _ = model.transcribe(wav, language="en", vad_filter=True, word_timestamps=True)
                        texts = [seg.text.strip() for seg in segments if (seg.end - seg.start) > 0.3]
                        if texts:
                            self.window_text_value = " ".join(texts)
                    except Exception:
                        pass
                    self.last_latency_ms = (now() - t1) * 1000.0
                    for _ in range(len(ring) // 3):
                        ring.popleft()
            self.log.info("ASR queue reader stopping…")
            return

        def cb(indata, frames, time_info, status):
            if not self.running:
                return
            pcm = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32)  # type: ignore
            rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0  # type: ignore
            self.last_rms_dbfs = -120.0 if rms <= 1e-9 else 20.0 * np.log10(max(1e-3, rms) / 32768.0)  # type: ignore
            ring.extend(pcm)
            if len(ring) >= ring.maxlen and self.last_rms_dbfs > -55.0:
                wav = np.array(ring, dtype=np.float32) / 32768.0  # type: ignore
                t1 = now()
                try:
                    segments, _ = model.transcribe(wav, language="en", vad_filter=True, word_timestamps=True)
                    texts = [seg.text.strip() for seg in segments if (seg.end - seg.start) > 0.3]
                    if texts:
                        self.window_text_value = " ".join(texts)
                except Exception:
                    pass
                self.last_latency_ms = (now() - t1) * 1000.0
                for _ in range(len(ring) // 3):
                    ring.popleft()

        try:
            stream = sd.RawInputStream(samplerate=sr, blocksize=block, dtype="int16", channels=1, callback=cb)  # type: ignore
        except Exception as e:
            self.log.warning("Audio init failed: %s", e)
            return

        with stream:
            self.log.info("ASR stream started (%.1f kHz).", sr / 1000)
            while self.running:
                time.sleep(0.05)
        self.log.info("ASR stream stopping…")

    def stop(self):
        self.running = False

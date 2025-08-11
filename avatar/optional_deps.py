# Safe, optional imports shared by modules. Everything else imports from here.
from __future__ import annotations

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import pygame  # type: ignore
except Exception:
    pygame = None  # type: ignore

try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None  # type: ignore

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    from emotiefflib.facial_analysis import (
        EmotiEffLibRecognizerBase,  # type: ignore
        EmotiEffLibRecognizerTorch,  # type: ignore
    )
except Exception:
    EmotiEffLibRecognizerBase = object  # type: ignore
    EmotiEffLibRecognizerTorch = None  # type: ignore

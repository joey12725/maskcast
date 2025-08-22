from __future__ import annotations
import os, platform, shutil
from typing import List, Tuple

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

def is_apple_silicon() -> bool:
    try:
        return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")
    except Exception:
        return False

def pick_torch_device(preference: str = "auto") -> str:
    if preference == "cpu":
        return "cpu"
    # Prefer CUDA when available (covers NVIDIA and ROCm builds that present as CUDA)
    if torch is not None and hasattr(torch, "cuda"):
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    # Apple Metal Performance Shaders backend
    if torch is not None and hasattr(torch, "mps"):
        try:
            if torch.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"

def onnxruntime_providers(prefer_gpu: bool = True) -> List[str]:
    try:
        import onnxruntime as ort  # type: ignore
        available = list(ort.get_available_providers())
    except Exception:
        # Fallback if ORT not installed yet
        return ["CPUExecutionProvider"]

    preferred_order: List[str] = []

    # macOS: CoreML and/or MPS providers
    if platform.system() == "Darwin":
        if "CoreMLExecutionProvider" in available:
            preferred_order.append("CoreMLExecutionProvider")
        if "MPSExecutionProvider" in available:
            preferred_order.append("MPSExecutionProvider")

    # Windows: DirectML, CUDA
    if platform.system() == "Windows":
        if "DmlExecutionProvider" in available:
            preferred_order.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in available:
            preferred_order.append("CUDAExecutionProvider")

    # Linux/Other: CUDA, ROCm
    if "CUDAExecutionProvider" in available:
        preferred_order.append("CUDAExecutionProvider")
    if "ROCMExecutionProvider" in available:
        preferred_order.append("ROCMExecutionProvider")

    # Always include CPU as a fallback
    preferred_order.append("CPUExecutionProvider")

    # Keep only those that are truly available, preserve order
    ordered = [p for p in preferred_order if p in available] or ["CPUExecutionProvider"]
    return ordered

def faster_whisper_device_and_compute(preference: str = "auto") -> Tuple[str, str]:
    # Device selection: try CUDA, then auto/metal on Apple, else CPU
    device = "cpu"
    if preference == "cpu":
        device = "cpu"
    else:
        dev = pick_torch_device("auto")
        if dev == "cuda":
            device = "cuda"
        elif is_apple_silicon():
            # Let CTranslate2 decide the best backend (Metal if available)
            device = "auto"
        else:
            device = "cpu"

    # Compute type: FP16 when on CUDA, INT8 otherwise (Apple Silicon has issues with FP16 in CTranslate2)
    if device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8"
    return device, compute_type



#!/usr/bin/env python3
"""One-shot installer for the full ML stack, with the right wheels per platform.

macOS (Apple Silicon): Torch with MPS, ONNX Runtime with CoreML.
Windows: ONNX Runtime DirectML (AMD/Intel/NVIDIA), Torch CPU or CUDA.
Linux: Torch CPU/CUDA, ONNX Runtime CPU/CUDA/ROCm.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys

PY = sys.executable


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def pip_install(pkg: str) -> bool:
    code = run([PY, "-m", "pip", "install", "--upgrade", pkg])
    return code == 0


def main() -> int:
    system = platform.system()
    machine = platform.machine().lower()
    is_arm64 = machine in ("arm64", "aarch64")

    print(f"Detected OS={system} arch={machine}")

    # Optional system libs (best effort)
    if system == "Darwin":
        if shutil.which("brew"):
            print("Installing system libs via Homebrew (best effort): ffmpeg, portaudio…")
            run(["brew", "install", "ffmpeg", "portaudio"])
    elif system == "Linux":
        if shutil.which("apt-get"):
            print("Installing system libs via apt-get (best effort): ffmpeg, portaudio19-dev…")
            run(["sudo", "apt-get", "update"])  # noqa: S603
            run(["sudo", "apt-get", "install", "-y", "ffmpeg", "portaudio19-dev"])
        elif shutil.which("pacman"):
            run(["sudo", "pacman", "-S", "--noconfirm", "ffmpeg", "portaudio"])  # noqa: S603
    elif system == "Windows":
        # Optional: install ffmpeg if choco is available
        if shutil.which("choco"):
            run(["choco", "install", "-y", "ffmpeg"])

    base_packages = [
        "pip",
        "setuptools",
        "wheel",
        "numpy",
        "opencv-python",
        "pygame",
        "sounddevice",
        "aiohttp",
        "aiortc",
        "av",
        "bleak",
        "pycryptodome",
        "matplotlib",
        "pillow",
        "faster-whisper",
        "insightface",
        "emotiefflib",
    ]

    # Install packaging tools first
    for pkg in ["pip", "setuptools", "wheel"]:
        pip_install(pkg)

    # Torch: default wheels include MPS support on macOS; CPU on Windows/Linux unless CUDA/ROCm available
    pip_install("torch")

    # EmotiEffLib compatibility: pin timm to a known-good version
    # Newer timm releases introduce breaking changes that affect EmotiEffLib's EfficientNet usage
    pip_install("timm==0.9.16")

    # ONNX Runtime per-OS
    if system == "Windows":
        # DirectML enables GPU acceleration on AMD/NVIDIA/Intel via D3D12
        pip_install("onnxruntime-directml")
    else:
        # Try standard onnxruntime first
        if not pip_install("onnxruntime") and system == "Darwin" and is_arm64:
            # Fallback for older environments
            pip_install("onnxruntime-silicon")

    # Remaining packages (with some special handling)
    ok_any = True
    for pkg in base_packages:
        if pkg in ("insightface", "emotiefflib"):
            # Install without deps to avoid NumPy conflicts with Homebrew
            code = run([PY, "-m", "pip", "install", "--upgrade", "--no-deps", pkg])
            ok_any = (code == 0) and ok_any
        else:
            ok_any = pip_install(pkg) and ok_any

    # Verify environment
    try:
        import onnxruntime as ort  # type: ignore

        print("onnxruntime providers:", getattr(ort, "get_available_providers", lambda: [])())
    except Exception as e:
        print("onnxruntime not available:", e)

    try:
        import torch  # type: ignore

        devs = []
        try:
            if torch.cuda.is_available():
                devs.append("cuda")
        except Exception:
            pass
        try:
            if hasattr(torch, "mps") and torch.mps.is_available():
                devs.append("mps")
        except Exception:
            pass
        print("torch version:", getattr(torch, "__version__", "unknown"))
        print("torch devices:", devs or ["cpu"])
    except Exception as e:
        print("torch not available:", e)

    try:
        import timm  # type: ignore

        print("timm version:", getattr(timm, "__version__", "unknown"))
    except Exception as e:
        print("timm not available:", e)

    try:
        from emotiefflib.facial_analysis import EmotiEffLibRecognizerTorch  # type: ignore

        print("EmotiEffLibRecognizerTorch available:", EmotiEffLibRecognizerTorch is not None)
    except Exception as e:
        print("emotiefflib import failed:", e)

    print("\nDone. You can now run: python -m maskcast")
    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())

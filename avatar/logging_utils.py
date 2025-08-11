from __future__ import annotations
import logging, os, platform, sys

def setup_logging(debug: bool, trace: bool) -> None:
    level = logging.WARNING
    if debug:
        level = logging.INFO
    if trace:
        level = logging.DEBUG
    fmt = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

def _add_cuda_search_paths_windows(log: logging.Logger) -> None:
    if os.name != "nt": return
    added = []
    candidates = set()
    for base in (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Program Files\NVIDIA\CUDNN",
    ):
        if os.path.isdir(base):
            for root, dirs, files in os.walk(base):
                if os.path.basename(root).lower() == "bin":
                    candidates.add(root)
    try:
        import site, glob
        for sp in site.getsitepackages():
            for p in glob.glob(os.path.join(sp, "nvidia", "cudnn", "*", "bin")):
                if os.path.isdir(p):
                    candidates.add(p)
    except Exception:
        pass
    for p in sorted(candidates):
        try:
            os.add_dll_directory(p)
            added.append(p)
        except Exception:
            pass
    if added:
        log.info("CUDA/cuDNN search paths added: %s", "; ".join(added))
    else:
        log.info("No additional CUDA/cuDNN paths added (Windows)")

def env_dump(logger: logging.Logger) -> None:
    import importlib
    logger.info("===== Environment =====")
    logger.info("Python: %s", sys.version.split()[0])
    logger.info("OS: %s | %s", platform.system(), platform.platform())
    for mod, label in (("numpy","NumPy"), ("cv2","OpenCV"), ("pygame","Pygame"), ("torch","Torch")):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", getattr(getattr(m, "version", None), "ver", "?"))
            logger.info("%s: %s", label, ver)
        except Exception:
            logger.info("%s: <not installed>", label)
    try:
        from faster_whisper import WhisperModel  # noqa
        logger.info("faster-whisper: available")
    except Exception:
        logger.info("faster-whisper: <not installed>")
    try:
        import onnxruntime as ort
        logger.info("onnxruntime providers: %s", ort.get_available_providers())
    except Exception:
        logger.info("onnxruntime: <not installed>")
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        ins = [d for d in devs if d.get("max_input_channels", 0) > 0]
        logger.info("sounddevice: %d input devs", len(ins))
    except Exception as e:
        logger.info("sounddevice query failed or not installed: %s", e)
    logger.info("=======================")

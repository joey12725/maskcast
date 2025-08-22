from __future__ import annotations
import multiprocessing as mp
import logging, time, queue
from pathlib import Path
from typing import Dict, Any, Tuple

from .logging_utils import setup_logging, env_dump, _add_cuda_search_paths_windows
from .tuning import Tuning, CANONICAL
from .assets import list_emotions_present
from .utils import parse_window_arg, autodetect_camera_size
from .optional_deps import cv2
from .video import VideoWorker
from .asr import WhisperASR
from .render import Renderer
from .reporter import Reporter
from .plotter import plotter_process

def run(config: Dict[str, Any]) -> None:
    # Logging
    setup_logging(debug=config["logging"].get("debug", False), trace=config["logging"].get("trace", False))
    log = logging.getLogger("Avatar")

    # Windows CUDA paths
    _add_cuda_search_paths_windows(log)
    env_dump(log)

    # Tuning
    tuning = Tuning(**config.get("tuning", {}))

    eyes_dir = Path(config["assets"]["eyes"])
    mouths_dir = Path(config["assets"]["mouths"])

    # Shared state
    state: Dict[str, Any] = {
        "has_face": False,
        "face_center": (None, None),
        "face_norm": (0.0, 0.0),
        "last_seen_face_norm": (0.0, 0.0),
        "face_norm_vel": (0.0, 0.0),
        "emotion_label": "neutral",
        "emotion_probs": {"neutral": 1.0},
        "valence": 0.0,
        "arousal": 0.0,
        "emotion_latency_ms": 0.0,
        "pos_latency_ms": 0.0,
        "remote_latency_ms": 0.0,
        "frame_size": None,
        "last_face_ts": 0.0,
        "flash_label": None,
        "flash_until": 0.0,
    }

    # Which emotions do we have assets for?
    emotions_present, keywords = list_emotions_present(eyes_dir)
    print(f"[Setup] Eyes found; emotions: {emotions_present or ['<none>']}; keywords: {keywords or ['<none>']}")

    remote_cfg = config.get("remote", {})
    remote_srv = None
    frame_q = None
    audio_q = None
    if remote_cfg.get("enabled", False):
        try:
            from .remote import WebRTCServer  # type: ignore
        except Exception as e:
            log.error("Remote server requested but not available: %s", e)
        else:
            frame_q = queue.Queue(maxsize=1)
            audio_q = queue.Queue(maxsize=50)
            remote_srv = WebRTCServer(
                frame_q,
                audio_q,
                queue.Queue(maxsize=1),
                state,
                host=remote_cfg.get("host", "0.0.0.0"),
                port=int(remote_cfg.get("port", 8080)),
            )
            remote_srv.start()

    cam_index = int(config.get("camera", 0))
    detector_mode = config["backends"]["detector"]
    emotion_backend = config["backends"]["emotion"]
    mirror = bool(config["window"].get("mirror", True))
    # macOS camera permission preflight: open and immediately close capture once in main thread
    if cv2 is not None:
        try:
            cap_test = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
            _ = cap_test.isOpened()
            cap_test.release()
        except Exception:
            pass

    vid = VideoWorker(cam_index, tuning, state, log, detector_mode, emotion_backend, mirror=mirror, frame_queue=frame_q)
    vid.start()

    # ASR
    asr = None
    if config["asr"].get("enabled", True) and config["asr"]["engine"] == "whisper":
        asr = WhisperASR(config["asr"].get("whisper_model", "large-v3"), log, device_preference=config["asr"].get("device","auto"), pcm_queue=audio_q)
        asr.start()

    # Window size (cap to 720x1080)
    w, h = (640, 480)
    for _ in range(40):  # ~2s
        if state.get("frame_size"):
            w, h = state["frame_size"]
            break
        time.sleep(0.05)
    # Apply cap preserving aspect ratio
    try:
        if w > 720 or h > 1080:
            scale = min(720 / max(1, w), 1080 / max(1, h))
            w = max(1, int(round(w * scale)))
            h = max(1, int(round(h * scale)))
    except Exception:
        pass

    size_cfg = config["window"].get("size", "auto")
    if isinstance(size_cfg, str):
        if size_cfg.lower() != "auto":
            maybe = parse_window_arg(size_cfg)
            if maybe: w, h = maybe
    elif isinstance(size_cfg, (list, tuple)) and len(size_cfg) == 2:
        w, h = int(size_cfg[0]), int(size_cfg[1])
    # Enforce cap after config overrides
    try:
        if w > 720 or h > 1080:
            scale = min(720 / max(1, w), 1080 / max(1, h))
            w = max(1, int(round(w * scale)))
            h = max(1, int(round(h * scale)))
    except Exception:
        pass
    log.info("Window size: %dx%d", w, h)

    # Renderer (pygame in main thread)
    renderer = Renderer((w, h), tuning, state, log, eyes_dir, mouths_dir, frame_cb=(remote_srv.send_frame if remote_srv else None))

    # Plot process + reporter
    plot_q = None
    plot_proc = None
    plot_start_evt = None
    plot_stop_evt = None
    reporter = None

    if config["plots"].get("enabled", False):
        import numpy as np
        try:
            _ = np  # noqa
        except Exception:
            log.warning("NumPy not available; plots disabled.")
        else:
            plot_q = mp.Queue(maxsize=1000)
            plot_start_evt = mp.Event()
            plot_stop_evt = mp.Event()
            label_list = list(CANONICAL)
            plot_proc = mp.Process(
                target=plotter_process,
                args=(plot_q, label_list, int(config["plots"].get("seconds", 180)), float(config["plots"].get("hz", 10.0)), plot_start_evt, plot_stop_evt),
                daemon=False
            )
            plot_proc.start()
            if plot_start_evt is not None:
                plot_start_evt.wait(timeout=3.0)

    reporter = Reporter(state, asr, log, tuning, q=plot_q, hz=float(config["plots"].get("hz", 10.0)), keywords=keywords)
    reporter.start()

    try:
        renderer.loop()
    finally:
        log.info("Starting shutdown sequence…")
        # Stop producers first
        if reporter: reporter.stop()
        if vid: vid.stop()
        if asr: asr.stop()
        if remote_srv: remote_srv.stop()

        # Ask plotter proc to exit
        if plot_stop_evt is not None:
            plot_stop_evt.set()

        # Join threads
        for t, name in ((asr, "ASR"), (reporter, "Reporter"), (vid, "Video")):
            try:
                if t and hasattr(t, "is_alive") and t.is_alive():
                    log.info("Waiting for %s thread…", name)
                    t.join(timeout=2.5)
            except Exception:
                pass

        # Join plot process
        if plot_proc is not None and plot_proc.is_alive():
            log.info("Waiting for plotter process…")
            plot_proc.join(timeout=3.0)
            if plot_proc.is_alive():
                log.warning("Plotter process did not exit cleanly; terminating.")
                plot_proc.terminate()

        log.info("Shutdown complete.")

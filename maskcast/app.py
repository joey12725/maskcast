from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .assets import list_emotions_present
from .expressions import CANONICAL
from .logging_utils import _add_cuda_search_paths_windows, env_dump, setup_logging
from .tuning import Tuning
from .utils import fit_within, parse_window_arg

MAX_W, MAX_H = 720, 1080


def initial_state() -> Dict[str, Any]:
    """Shared blackboard. Workers write; the renderer and reporter read."""
    return {
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


def window_size(config: Dict[str, Any], frame_size: Optional[tuple]) -> tuple:
    """Window follows the camera aspect unless ``window.size`` says otherwise."""
    w, h = frame_size or (640, 480)
    size_cfg = config.get("window", {}).get("size", "auto")
    if isinstance(size_cfg, str) and size_cfg.lower() != "auto":
        w, h = parse_window_arg(size_cfg) or (w, h)
    elif isinstance(size_cfg, (list, tuple)) and len(size_cfg) == 2:
        w, h = int(size_cfg[0]), int(size_cfg[1])
    return fit_within(w, h, MAX_W, MAX_H)


def _fan_out(callbacks: List[Callable]) -> Optional[Callable]:
    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]

    def send(frame):
        for cb in callbacks:
            cb(frame)

    return send


def run(config: Dict[str, Any], demo: bool = False) -> None:
    setup_logging(debug=config["logging"].get("debug", False), trace=config["logging"].get("trace", False))
    log = logging.getLogger("maskcast")
    _add_cuda_search_paths_windows(log)
    env_dump(log)

    tuning = Tuning.from_dict(config.get("tuning", {}))
    eyes_dir = Path(config["assets"]["eyes"])
    mouths_dir = Path(config["assets"]["mouths"])
    state = initial_state()

    emotions_present, keywords = list_emotions_present(eyes_dir)
    print(f"maskcast: {len(emotions_present)} emotions + {len(keywords)} keyword flashes from {eyes_dir}")

    frame_outputs: List[Callable] = []
    workers: List[Any] = []

    # WebRTC: browser camera + mic in, rendered face out
    frame_q = audio_q = None
    remote_cfg = config.get("remote", {})
    if remote_cfg.get("enabled", False):
        try:
            from .remote import WebRTCServer
        except Exception as e:
            log.error("Remote server requested but not available: %s", e)
        else:
            frame_q, audio_q = queue.Queue(maxsize=1), queue.Queue(maxsize=50)
            srv = WebRTCServer(
                frame_q,
                audio_q,
                queue.Queue(maxsize=1),
                state,
                host=remote_cfg.get("host", "127.0.0.1"),
                port=int(remote_cfg.get("port", 8080)),
            )
            srv.start()
            workers.append(srv)
            frame_outputs.append(srv.send_frame)

    # Bluetooth LED mask
    mask_cfg = config.get("mask", {})
    if mask_cfg.get("enabled", False):
        try:
            from .mask import MaskOutput

            led = MaskOutput(
                int(mask_cfg.get("width", 45)), int(mask_cfg.get("height", 60)), mask_cfg.get("name_prefix", "Mask")
            )
        except Exception as e:
            log.error("LED mask requested but not available: %s", e)
        else:
            led.start()
            workers.append(led)
            frame_outputs.append(led.send)

    asr = None
    if demo:
        from .demo import DemoDriver

        driver = DemoDriver(state)
        driver.start()
        workers.append(driver)
    else:
        from .asr import WhisperASR
        from .optional_deps import cv2
        from .video import VideoWorker

        cam_index = int(config.get("camera", 0))
        if cv2 is not None and frame_q is None:
            # macOS: trigger the camera permission prompt from the main thread
            try:
                cv2.VideoCapture(cam_index, cv2.CAP_ANY).release()
            except Exception:
                pass
        vid = VideoWorker(
            cam_index,
            tuning,
            state,
            log,
            config["backends"]["detector"],
            config["backends"]["emotion"],
            mirror=bool(config["window"].get("mirror", True)),
            frame_queue=frame_q,
        )
        vid.start()
        workers.append(vid)

        asr_cfg = config.get("asr", {})
        if asr_cfg.get("enabled", True) and asr_cfg.get("engine", "whisper") == "whisper":
            asr = WhisperASR(
                asr_cfg.get("whisper_model", "small.en"),
                log,
                device_preference=asr_cfg.get("device", "auto"),
                pcm_queue=audio_q,
            )
            asr.start()
            workers.append(asr)

    # Give the camera ~2s to report its frame size
    for _ in range(40):
        if state.get("frame_size"):
            break
        time.sleep(0.05)
    w, h = window_size(config, state.get("frame_size"))
    log.info("Window size: %dx%d", w, h)

    from .render import Renderer

    renderer = Renderer((w, h), tuning, state, log, eyes_dir, mouths_dir, frame_cb=_fan_out(frame_outputs))

    plot_proc = plot_stop_evt = plot_q = None
    plots_cfg = config.get("plots", {})
    if plots_cfg.get("enabled", False):
        from .plotter import plotter_process

        plot_q, plot_start_evt, plot_stop_evt = mp.Queue(maxsize=1000), mp.Event(), mp.Event()
        plot_proc = mp.Process(
            target=plotter_process,
            args=(
                plot_q,
                list(CANONICAL),
                int(plots_cfg.get("seconds", 180)),
                float(plots_cfg.get("hz", 10.0)),
                plot_start_evt,
                plot_stop_evt,
            ),
            daemon=False,
        )
        plot_proc.start()
        plot_start_evt.wait(timeout=3.0)

    from .reporter import Reporter

    reporter = Reporter(state, asr, log, tuning, q=plot_q, hz=float(plots_cfg.get("hz", 2.0)), keywords=keywords)
    reporter.start()
    workers.append(reporter)

    try:
        renderer.loop()
    finally:
        log.info("Shutting down")
        for wk in workers:
            try:
                wk.stop()
            except Exception:
                pass
        if plot_stop_evt is not None:
            plot_stop_evt.set()
        for wk in workers:
            if wk.is_alive():
                wk.join(timeout=2.5)
        if plot_proc is not None:
            plot_proc.join(timeout=3.0)
            if plot_proc.is_alive():
                plot_proc.terminate()
        log.info("Shutdown complete.")

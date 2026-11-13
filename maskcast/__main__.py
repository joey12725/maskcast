from __future__ import annotations

import argparse
import multiprocessing as mp
import os

from .config import apply_overrides, load_config

# Quiet down third-party startup noise
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="maskcast", description="Real-time expression-driven anime face mask")
    ap.add_argument("--config", "-c", default="configs/default.json", help="path to config JSON")
    ap.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value by dotted path, e.g. -o window.mirror=false",
    )
    ap.add_argument(
        "--demo",
        action="store_true",
        help="drive the face from a scripted performance instead of the camera (no ML deps needed)",
    )
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    mp.set_start_method("spawn", force=True)

    from .app import run

    try:
        run(cfg, demo=args.demo)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

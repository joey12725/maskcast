#!/usr/bin/env python3
"""Record docs/demo.gif by running the real renderer headless in --demo mode.

No camera or ML models are involved: the DemoDriver plays a scripted
performance through the same state the live trackers write.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from PIL import Image  # noqa: E402

from maskcast.app import initial_state  # noqa: E402
from maskcast.demo import DEFAULT_SCRIPT, DemoDriver  # noqa: E402
from maskcast.render import Renderer  # noqa: E402
from maskcast.tuning import Tuning  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--seconds", type=float, default=sum(s[3] for s in DEFAULT_SCRIPT))
    ap.add_argument("--fps", type=int, default=20, help="GIF frame rate")
    ap.add_argument("--size", default="480x360")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))

    state = initial_state()
    driver = DemoDriver(state)
    driver.start()
    renderer = Renderer(
        (w, h), Tuning(), state, logging.getLogger("record"), Path("assets/eyes"), Path("assets/mouths"), fps=60
    )

    frames = []
    needed = int(args.seconds * args.fps)
    next_t = {"t": 0.0}

    def grab(frame):
        # Sample on wall-clock time so the GIF plays back at real speed
        t = time.monotonic()
        if t >= next_t["t"]:
            next_t["t"] = (next_t["t"] or t) + 1.0 / args.fps
            frames.append(Image.fromarray(frame))
            if len(frames) >= needed:
                renderer.stop()

    renderer.frame_cb = grab
    renderer.loop()
    driver.stop()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.quantize(colors=32, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

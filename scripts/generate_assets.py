#!/usr/bin/env python3
"""Regenerate the procedural eye/mouth art in assets/.

python scripts/generate_assets.py               # pink, into ./assets
python scripts/generate_assets.py --color 80ffd0 --out my-assets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maskcast.artgen import PINK, generate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="assets", help="output folder (eyes/ and mouths/ are created inside)")
    ap.add_argument("--color", default="%02x%02x%02x" % PINK, help="hex colour, e.g. ff69b4")
    args = ap.parse_args()
    color = tuple(int(args.color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    paths = generate(Path(args.out), color)
    print(f"wrote {len(paths)} images to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

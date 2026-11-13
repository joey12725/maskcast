#!/usr/bin/env python3
"""Render docs/expressions.png: every expression as a face tile."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from maskcast.artgen import EYES, MOUTHS, draw_eyes, draw_mouth  # noqa: E402

BG = (10, 10, 14)
TILE = (240, 220)
COLS = 6


def face_tile(name: str) -> Image.Image:
    tile = Image.new("RGB", TILE, BG)
    eyes = draw_eyes(name).resize((168, 63), Image.LANCZOS)
    mouth = draw_mouth(name if name in MOUTHS else "neutral").resize((100, 50), Image.LANCZOS)
    for img, (x, y) in ((eyes, (36, 40)), (mouth, (70, 115))):
        mask = img.convert("L").point(lambda v: 255 if v > 10 else 0)
        tile.paste(img, (x, y), mask)
    d = ImageDraw.Draw(tile)
    try:
        font = ImageFont.load_default(size=17)
    except TypeError:
        font = ImageFont.load_default()
    d.text((TILE[0] / 2, 196), name, fill=(170, 170, 185), font=font, anchor="mm")
    return tile


def main() -> int:
    names = list(EYES)
    rows = (len(names) + COLS - 1) // COLS
    sheet = Image.new("RGB", (TILE[0] * COLS, TILE[1] * rows), BG)
    for i, name in enumerate(names):
        sheet.paste(face_tile(name), ((i % COLS) * TILE[0], (i // COLS) * TILE[1]))
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/expressions.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

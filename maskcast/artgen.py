"""Procedural face art: every eye and mouth shape maskcast ships with.

All shapes are drawn from primitives (pills, arcs, polygons) in a single
colour on black, the classic LED-face look. The renderer keys out the black,
so the PNGs double as a template for your own art: draw on black, name the
file after an emotion, done.

Run ``python scripts/generate_assets.py`` to rebuild ``assets/``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont

PINK = (255, 105, 180)
BLACK = (0, 0, 0)
SS = 4  # supersampling factor for smooth edges

EYES_SIZE = (640, 240)
MOUTH_SIZE = (400, 200)
EYE_X = (160, 480)  # centre of left / right eye
EYE_Y = 120

Draw = ImageDraw.ImageDraw
Pt = Tuple[float, float]


# ------------------------------------------------------------- primitives
class Pen:
    """Thin wrapper that works in final-image pixels and scales by SS."""

    def __init__(self, size: Tuple[int, int], color=PINK):
        self.size = size
        self.img = Image.new("RGB", (size[0] * SS, size[1] * SS), BLACK)
        self.d = ImageDraw.Draw(self.img)
        self.color = color

    def _s(self, *vals: float) -> List[float]:
        return [v * SS for v in vals]

    def pill(self, cx, cy, w, h, color=None, r=None):
        r = min(w, h) * 0.42 if r is None else r
        self.d.rounded_rectangle(
            self._s(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), radius=r * SS, fill=color or self.color
        )

    def ellipse(self, cx, cy, rx, ry, color=None, width=0):
        box = self._s(cx - rx, cy - ry, cx + rx, cy + ry)
        if width:
            self.d.ellipse(box, outline=color or self.color, width=int(width * SS))
        else:
            self.d.ellipse(box, fill=color or self.color)

    def poly(self, pts: Iterable[Pt], color=None):
        self.d.polygon([(x * SS, y * SS) for x, y in pts], fill=color or self.color)

    def line(self, pts: Iterable[Pt], width: float, color=None, round_caps: bool = True):
        pts = [(x * SS, y * SS) for x, y in pts]
        c = color or self.color
        self.d.line(pts, fill=c, width=int(width * SS), joint="curve")
        if round_caps:
            r = width * SS / 2
            for x, y in (pts[0], pts[-1]):
                self.d.ellipse((x - r, y - r, x + r, y + r), fill=c)

    def arc(self, cx, cy, rx, ry, start, end, width, color=None):
        """Arc as a polyline so it gets round caps. Angles in degrees, 0 = +x, 90 = down."""
        n = max(8, int(abs(end - start) / 4))
        pts = [
            (
                cx + rx * math.cos(math.radians(start + (end - start) * i / n)),
                cy + ry * math.sin(math.radians(start + (end - start) * i / n)),
            )
            for i in range(n + 1)
        ]
        self.line(pts, width, color)

    def text(self, cx, cy, s: str, size: int, color=None):
        try:
            font = ImageFont.load_default(size=size * SS)
        except TypeError:  # Pillow < 10.1
            font = ImageFont.load_default()
        self.d.text((cx * SS, cy * SS), s, font=font, fill=color or self.color, anchor="mm")

    def done(self) -> Image.Image:
        return self.img.resize(self.size, Image.LANCZOS)


def heart(p: Pen, cx, cy, s):
    r = s * 0.3
    p.ellipse(cx - r * 0.95, cy - r * 0.4, r, r)
    p.ellipse(cx + r * 0.95, cy - r * 0.4, r, r)
    p.poly([(cx - s * 0.58, cy - r * 0.05), (cx + s * 0.58, cy - r * 0.05), (cx, cy + s * 0.62)])


def star(p: Pen, cx, cy, r_out, r_in=None, n=5, rot=-90):
    r_in = r_out * 0.45 if r_in is None else r_in
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = math.radians(rot + i * 180 / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    p.poly(pts)


def spiral(p: Pen, cx, cy, r, turns=2.6, width=11):
    n = 160
    pts = []
    for i in range(n + 1):
        t = i / n
        a = t * turns * 2 * math.pi
        pts.append((cx + r * t * math.cos(a), cy + r * t * math.sin(a)))
    p.line(pts, width)


# ------------------------------------------------------------------- eyes
EyeFn = Callable[[Pen, float, int], None]  # (pen, centre x, side -1/+1)
W, H = 120, 150  # base eye size


def eye_neutral(p, x, side):
    p.pill(x, EYE_Y, W, H)


def eye_happy(p, x, side):
    p.arc(x, EYE_Y + 30, 62, 60, 200, 340, 26)


def eye_happy_big(p, x, side):  # happy, high valence + arousal
    p.arc(x, EYE_Y + 20, 70, 72, 195, 345, 30)
    star(p, x + side * 70, EYE_Y - 70, 18)


def eye_sad(p, x, side):
    p.pill(x, EYE_Y + 10, W, H * 0.9)
    inner, outer = x - side * 90, x + side * 90
    # brow cut: high on the inside, low on the outside
    p.poly([(inner, EYE_Y - 120), (outer, EYE_Y - 120), (outer, EYE_Y - 5), (inner, EYE_Y - 75)], BLACK)


def eye_sad_low(p, x, side):  # sad, low valence + low arousal: droopy, closed
    p.arc(x, EYE_Y - 10, 58, 40, 20, 160, 22)
    p.ellipse(x - side * 30, EYE_Y + 75, 11, 16)  # tear


def eye_angry(p, x, side):
    p.pill(x, EYE_Y + 10, W, H * 0.9)
    p.poly(
        [
            (x - side * 90, EYE_Y - 120),
            (x + side * 90, EYE_Y - 120),
            (x + side * 90, EYE_Y - 55),
            (x - side * 90, EYE_Y + 10),
        ],
        BLACK,
    )


def eye_angry_hot(p, x, side):  # angry, low valence + high arousal: slits
    p.poly(
        [
            (x - side * 70, EYE_Y + 10),
            (x + side * 70, EYE_Y - 45),
            (x + side * 55, EYE_Y + 45),
            (x - side * 55, EYE_Y + 45),
        ]
    )


def eye_surprise(p, x, side):
    p.ellipse(x, EYE_Y, 72, 80, width=20)
    p.ellipse(x, EYE_Y, 26, 30)


def eye_surprise_big(p, x, side):
    p.ellipse(x, EYE_Y, 84, 92, width=16)
    p.ellipse(x, EYE_Y, 16, 18)


def eye_fear(p, x, side):
    p.ellipse(x, EYE_Y, 66, 74, width=14)
    p.ellipse(x + side * 6, EYE_Y + 6, 14, 14)
    p.line([(x - side * 70, EYE_Y - 105), (x + side * 55, EYE_Y - 92)], 12)


def eye_disgust(p, x, side):
    p.pill(x, EYE_Y + 35, W * 1.05, H * 0.45)
    p.line([(x - 60, EYE_Y - 40), (x + 60, EYE_Y - 30 + side * 10)], 14)


def eye_contempt(p, x, side):
    p.pill(x, EYE_Y + 25, W, H * 0.6)
    p.poly([(x - 80, EYE_Y - 100), (x + 80, EYE_Y - 100), (x + 80, EYE_Y + 10), (x - 80, EYE_Y + 10)], BLACK)
    p.line([(x - 62, EYE_Y + 6), (x + 62, EYE_Y + 6)], 14)


def eye_heart(p, x, side):
    heart(p, x, EYE_Y, 190)


def eye_star(p, x, side):
    star(p, x, EYE_Y + 5, 92)


def eye_x(p, x, side):
    p.line([(x - 55, EYE_Y - 55), (x + 55, EYE_Y + 55)], 26)
    p.line([(x - 55, EYE_Y + 55), (x + 55, EYE_Y - 55)], 26)
    p.poly([(x, EYE_Y + 70), (x + 14, EYE_Y + 92), (x, EYE_Y + 114), (x - 14, EYE_Y + 92)])


def eye_shades(p, x, side):
    p.poly(
        [
            (x - 95, EYE_Y - 45),
            (x + 95, EYE_Y - 45),
            (x + 80, EYE_Y + 40),
            (x + 30, EYE_Y + 55),
            (x - 30, EYE_Y + 55),
            (x - 80, EYE_Y + 40),
        ]
    )
    if side < 0:  # bridge between the lenses
        p.line([(x + 90, EYE_Y - 36), (EYE_X[1] - 90, EYE_Y - 36)], 16, round_caps=False)


def eye_demon(p, x, side):
    p.poly([(x - side * 85, EYE_Y - 5), (x + side * 80, EYE_Y - 60), (x + side * 60, EYE_Y + 50)])
    p.ellipse(x + side * 30, EYE_Y + 5, 8, 26, color=BLACK)


def eye_dollar(p, x, side):
    p.text(x, EYE_Y, "$", 210)


def eye_chevron(p, x, side):
    # ">" on the left, "<" on the right
    tip = x + side * -50
    p.line([(x + side * 50, EYE_Y - 60), (tip, EYE_Y), (x + side * 50, EYE_Y + 60)], 26)


def eye_pog(p, x, side):
    p.ellipse(x, EYE_Y, 80, 88)
    p.ellipse(x - 22, EYE_Y - 28, 22, 22, color=BLACK)
    p.ellipse(x + 26, EYE_Y + 24, 10, 10, color=BLACK)


def eye_question(p, x, side):
    p.text(x, EYE_Y, "?", 210)


def eye_sleepy(p, x, side):
    p.arc(x, EYE_Y - 10, 60, 32, 15, 165, 20)
    if side > 0:
        p.text(x + 90, EYE_Y - 80, "z", 70)
        p.text(x + 125, EYE_Y - 115, "z", 46)


def eye_spiral(p, x, side):
    spiral(p, x, EYE_Y, 82, turns=2.6 if side < 0 else -2.6, width=12)


def eye_wink(p, x, side):
    if side < 0:
        p.pill(x, EYE_Y, W, H)
    else:
        p.line([(x + 50, EYE_Y - 45), (x - 45, EYE_Y), (x + 50, EYE_Y + 45)], 24)


EYES: Dict[str, EyeFn] = {
    "neutral": eye_neutral,
    "happy": eye_happy,
    "happy_v+_a+": eye_happy_big,
    "sad": eye_sad,
    "sad_v-_a-": eye_sad_low,
    "angry": eye_angry,
    "angry_v-_a+": eye_angry_hot,
    "surprise": eye_surprise,
    "surprise_v0_a+": eye_surprise_big,
    "fear": eye_fear,
    "disgust": eye_disgust,
    "contempt": eye_contempt,
    # keyword flashes
    "adorable": eye_heart,
    "amazing": eye_star,
    "clown": eye_x,
    "cool": eye_shades,
    "demon": eye_demon,
    "money": eye_dollar,
    "owo": eye_chevron,
    "pog": eye_pog,
    "question": eye_question,
    "sleepy": eye_sleepy,
    "wild": eye_spiral,
    "wink": eye_wink,
}


# ----------------------------------------------------------------- mouths
MX, MY = MOUTH_SIZE[0] / 2, MOUTH_SIZE[1] / 2


def mouth_neutral(p):
    p.pill(MX, MY, 170, 26)


def mouth_smile(p):
    p.arc(MX, MY - 40, 110, 80, 25, 155, 24)


def mouth_grin(p):
    p.d.chord([v * SS for v in (MX - 130, MY - 120, MX + 130, MY + 80)], 0, 180, fill=PINK)


def mouth_frown(p):
    p.arc(MX, MY + 60, 100, 70, 205, 335, 24)


def mouth_angry(p):
    p.poly([(MX - 120, MY - 20), (MX + 120, MY - 20), (MX + 95, MY + 45), (MX - 95, MY + 45)])
    for i in range(5):  # teeth gaps
        x = MX - 80 + i * 40
        p.line([(x, MY - 20), (x, MY + 45)], 6, BLACK, round_caps=False)


def mouth_o(p):
    p.ellipse(MX, MY, 48, 60, width=20)


def mouth_o_big(p):
    p.ellipse(MX, MY, 58, 74)


def mouth_wave(p, amp=16, width=20):
    pts = [(MX - 120 + i * 6, MY + amp * math.sin(i * 6 / 240 * 4 * math.pi)) for i in range(41)]
    p.line(pts, width)


def mouth_zigzag(p):
    pts = [(MX - 120 + i * 30, MY + (18 if i % 2 else -18)) for i in range(9)]
    p.line(pts, 18)


def mouth_smirk(p):
    p.line([(MX - 100, MY + 10), (MX + 40, MY + 10)], 20)
    p.arc(MX + 40, MY - 25, 55, 35, 20, 90, 20)


def mouth_cat(p):
    p.arc(MX - 36, MY - 10, 36, 34, 10, 170, 18)
    p.arc(MX + 36, MY - 10, 36, 34, 10, 170, 18)


def mouth_fangs(p):
    mouth_grin(p)
    for dx in (-70, 70):
        p.poly([(MX + dx - 18, MY - 21), (MX + dx + 18, MY - 21), (MX + dx, MY + 18)], BLACK)


def mouth_tongue(p):
    p.arc(MX, MY - 40, 110, 80, 25, 155, 22)
    p.d.chord([v * SS for v in (MX + 10, MY + 5, MX + 80, MY + 85)], 0, 180, fill=PINK)


def mouth_small_o(p):
    p.ellipse(MX + 30, MY + 10, 22, 26, width=14)


MOUTHS: Dict[str, Callable[[Pen], None]] = {
    "neutral": mouth_neutral,
    "happy": mouth_smile,
    "happy_v+_a+": mouth_grin,
    "sad": mouth_frown,
    "sad_v-_a-": lambda p: mouth_wave(p, amp=8, width=16),
    "angry": mouth_angry,
    "angry_v-_a+": mouth_zigzag,
    "surprise": mouth_o,
    "surprise_v0_a+": mouth_o_big,
    "fear": mouth_wave,
    "disgust": mouth_zigzag,
    "contempt": mouth_smirk,
    "adorable": mouth_cat,
    "amazing": mouth_grin,
    "clown": mouth_grin,
    "cool": mouth_smirk,
    "demon": mouth_fangs,
    "money": mouth_grin,
    "owo": mouth_cat,
    "pog": mouth_o_big,
    "question": lambda p: mouth_wave(p, amp=10, width=16),
    "sleepy": mouth_small_o,
    "wild": mouth_zigzag,
    "wink": mouth_tongue,
}


# ------------------------------------------------------------------ build
def draw_eyes(name: str, color=PINK) -> Image.Image:
    p = Pen(EYES_SIZE, color)
    for x, side in zip(EYE_X, (-1, 1), strict=False):
        EYES[name](p, x, side)
    return p.done()


def draw_mouth(name: str, color=PINK) -> Image.Image:
    p = Pen(MOUTH_SIZE, color)
    MOUTHS[name](p)
    return p.done()


def generate(out_dir: Path, color=PINK) -> List[Path]:
    """Write every eye and mouth PNG under ``out_dir/{eyes,mouths}``."""
    written = []
    for sub, table, fn in (("eyes", EYES, draw_eyes), ("mouths", MOUTHS, draw_mouth)):
        d = Path(out_dir) / sub
        d.mkdir(parents=True, exist_ok=True)
        for name in table:
            path = d / f"{name}.png"
            fn(name, color).save(path, optimize=True)
            written.append(path)
    return written

"""Shipped art covers every emotion and keyword, and the renderer can draw it."""

import logging
import os
from pathlib import Path

import pytest

from maskcast.assets import list_emotions_present
from maskcast.expressions import CANONICAL, KEYWORD_ALIASES

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def test_shipped_assets_cover_vocabulary():
    emotions, keywords = list_emotions_present(ASSETS / "eyes")
    assert set(emotions) == set(CANONICAL)
    assert set(keywords) == set(KEYWORD_ALIASES)
    eyes = {p.stem for p in (ASSETS / "eyes").glob("*.png")}
    mouths = {p.stem for p in (ASSETS / "mouths").glob("*.png")}
    assert eyes == mouths


def test_artgen_matches_shipped_assets(tmp_path):
    pytest.importorskip("PIL")
    from maskcast.artgen import EYES, MOUTHS, generate

    assert set(EYES) == set(MOUTHS)
    written = generate(tmp_path)
    assert {p.name for p in written if p.parent.name == "eyes"} == {p.name for p in (ASSETS / "eyes").glob("*.png")}


def test_renderer_draws_headless():
    pytest.importorskip("pygame")
    np = pytest.importorskip("numpy")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from maskcast.app import initial_state
    from maskcast.render import BACKGROUND, Renderer
    from maskcast.tuning import Tuning

    state = initial_state()
    r = Renderer((320, 240), Tuning(), state, logging.getLogger("test"), ASSETS / "eyes", ASSETS / "mouths")
    assert set(r.eyes) >= set(CANONICAL) | set(KEYWORD_ALIASES)
    state.update(emotion_label="happy", valence=0.9, arousal=0.9)
    r.draw_frame(1 / 60)
    frame = r.frame_array()
    assert frame.shape == (240, 320, 3)
    assert (frame != np.array(BACKGROUND)).any()
    r.stop()

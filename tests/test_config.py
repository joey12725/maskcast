import json
import logging
from pathlib import Path

import pytest

from maskcast.app import window_size
from maskcast.config import apply_overrides, coerce, load_config
from maskcast.tuning import Tuning
from maskcast.utils import fit_within, parse_window_arg

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "raw,val",
    [
        ("true", True),
        ("False", False),
        ("3", 3),
        ("2.5", 2.5),
        ("1e-3", 0.001),
        ("null", None),
        ("insightface", "insightface"),
        ("127.0.0.1", "127.0.0.1"),
    ],
)
def test_coerce(raw, val):
    assert coerce(raw) == val


def test_apply_overrides_nested_and_non_mutating():
    cfg = {"window": {"mirror": True}, "plots": {"hz": 10.0}}
    out = apply_overrides(cfg, ["window.mirror=false", "plots.hz=5", "remote.port=9000"])
    assert out == {"window": {"mirror": False}, "plots": {"hz": 5}, "remote": {"port": 9000}}
    assert cfg["window"]["mirror"] is True


def test_apply_overrides_rejects_garbage():
    with pytest.raises(ValueError):
        apply_overrides({}, ["window.mirror"])
    with pytest.raises(ValueError):
        apply_overrides({}, ["=1"])


def test_default_config_maps_onto_tuning(caplog):
    cfg = load_config(str(ROOT / "configs" / "default.json"))
    with caplog.at_level(logging.WARNING):
        t = Tuning.from_dict(cfg["tuning"])
    assert "unknown" not in caplog.text
    assert t.arousal_high_threshold == cfg["tuning"]["arousal_high_threshold"]
    assert cfg["remote"]["host"] == "127.0.0.1"
    json.dumps(cfg)  # stays serialisable


def test_tuning_ignores_unknown_keys(caplog):
    with caplog.at_level(logging.WARNING):
        t = Tuning.from_dict({"gaze_omega": 3.0, "bogus": 1})
    assert t.gaze_omega == 3.0
    assert "bogus" in caplog.text


@pytest.mark.parametrize(
    "raw,val", [("640x480", (640, 480)), ("800,600", (800, 600)), ("nope", None), ("", None), ("axb", None)]
)
def test_parse_window_arg(raw, val):
    assert parse_window_arg(raw) == val


def test_fit_within_keeps_aspect():
    assert fit_within(640, 480, 720, 1080) == (640, 480)
    assert fit_within(1920, 1080, 720, 1080) == (720, 405)


def test_window_size():
    assert window_size({"window": {"size": "auto"}}, (1280, 720)) == (720, 405)
    assert window_size({"window": {"size": "400x300"}}, (1280, 720)) == (400, 300)
    assert window_size({"window": {"size": [300, 300]}}, None) == (300, 300)
    assert window_size({"window": {}}, None) == (640, 480)

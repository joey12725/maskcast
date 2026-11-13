import math

import pytest

from maskcast.motion import Ema2D, SpringDamper2D, gaze_target, norm_to_px
from maskcast.tuning import Tuning

T = Tuning()


def test_norm_to_px_uses_per_direction_limits():
    assert norm_to_px((1.0, 1.0), T) == (T.gaze_limit_px_right, T.gaze_limit_px_down)
    assert norm_to_px((-1.0, -1.0), T) == (-T.gaze_limit_px_left, -T.gaze_limit_px_up)
    assert norm_to_px((5.0, -5.0), T, energy=1.5) == (T.gaze_limit_px_right, -T.gaze_limit_px_up)


def test_gaze_follows_then_holds_then_returns():
    seen = (0.5, 0.5)
    live = gaze_target(True, seen, seen, 0.0, T)
    assert live == norm_to_px(seen, T)
    assert gaze_target(False, (0, 0), seen, T.face_hold_s * 0.5, T) == live
    mid = gaze_target(False, (0, 0), seen, T.face_hold_s + T.return_to_center_s / 2, T)
    assert mid == pytest.approx((live[0] / 2, live[1] / 2))
    assert gaze_target(False, (0, 0), seen, 60.0, T) == (0.0, 0.0)


def test_spring_settles_on_target():
    s = SpringDamper2D.from_tuning(T)
    for _ in range(600):
        s.step((30.0, -20.0), 1 / 60)
    assert s.pos == pytest.approx((30.0, -20.0), abs=1.0)


def test_spring_clamps_velocity():
    s = SpringDamper2D(omega=50, zeta=0.1, max_vel=100, max_acc=1e9)
    for _ in range(10):
        s.step((10_000.0, 0.0), 1 / 60)
        assert math.hypot(*s.vel) <= 100 + 1e-6


def test_spring_deadzone_stops_jitter():
    s = SpringDamper2D(deadzone=2.0)
    s.step((1.0, 0.0), 1 / 60)
    assert s.pos == (0.0, 0.0)


def test_ema2d():
    e = Ema2D(alpha=0.5)
    e.update((2.0, 4.0))
    assert e.update((2.0, 4.0)) == (1.5, 3.0)

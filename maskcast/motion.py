"""Gaze motion: where the eyes want to look, and how they get there.

Pure Python on 2-tuples so it can be tested (and reasoned about) without
numpy or a window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

from .tuning import Tuning

Vec = Tuple[float, float]


def _clamp_norm(x: float, y: float, limit: float) -> Vec:
    n = math.hypot(x, y)
    if n > limit > 0:
        s = limit / n
        return x * s, y * s
    return x, y


def norm_to_px(norm: Vec, tuning: Tuning, energy: float = 1.0) -> Vec:
    """Map a face position in [-1, 1] to a gaze offset in pixels.

    Each direction has its own travel limit so the eyes can look further down
    than up, which reads more naturally on a face-shaped display.
    """
    x, y = norm
    lim_l, lim_r = tuning.gaze_limit_px_left, tuning.gaze_limit_px_right
    lim_u, lim_d = tuning.gaze_limit_px_up, tuning.gaze_limit_px_down
    tx = max(-lim_l, min(lim_r, x * (lim_r if x >= 0 else lim_l) * energy))
    ty = max(-lim_u, min(lim_d, y * (lim_d if y >= 0 else lim_u) * energy))
    return tx, ty


def gaze_target(
    has_face: bool, face_norm: Vec, last_seen_norm: Vec, face_age_s: float, tuning: Tuning, energy: float = 1.0
) -> Vec:
    """Target gaze offset given tracker state.

    While a face is visible the eyes follow it. When it is lost they hold the
    last position for ``face_hold_s``, then ease back to centre over
    ``return_to_center_s``.
    """
    if has_face:
        return norm_to_px(face_norm, tuning, energy)
    if face_age_s <= tuning.face_hold_s:
        return norm_to_px(last_seen_norm, tuning, energy)
    fade = tuning.return_to_center_s
    if face_age_s <= tuning.face_hold_s + fade:
        t = (face_age_s - tuning.face_hold_s) / max(1e-6, fade)
        bx, by = norm_to_px(last_seen_norm, tuning, energy)
        return (1.0 - t) * bx, (1.0 - t) * by
    return 0.0, 0.0


@dataclass
class Ema2D:
    alpha: float
    value: Vec = (0.0, 0.0)

    def update(self, sample: Vec) -> Vec:
        a = self.alpha
        self.value = ((1 - a) * self.value[0] + a * sample[0], (1 - a) * self.value[1] + a * sample[1])
        return self.value


@dataclass
class SpringDamper2D:
    """Second-order spring-damper that chases a target.

    ``omega`` is the natural frequency (rad/s) and ``zeta`` the damping ratio.
    Under 1.0 the eyes overshoot slightly, which looks alive; at 1.0 they
    settle as fast as possible without overshoot. Acceleration and velocity
    are clamped so a sudden jump in the target never teleports the eyes.
    """

    omega: float = 10.0
    zeta: float = 0.75
    max_vel: float = 500.0
    max_acc: float = 2500.0
    deadzone: float = 0.75
    pos: Vec = field(default=(0.0, 0.0))
    vel: Vec = field(default=(0.0, 0.0))

    @classmethod
    def from_tuning(cls, t: Tuning) -> "SpringDamper2D":
        return cls(
            omega=t.gaze_omega,
            zeta=t.gaze_zeta,
            max_vel=t.gaze_max_vel_px,
            max_acc=t.gaze_max_acc_px,
            deadzone=t.gaze_deadzone_px,
        )

    def step(self, target: Vec, dt: float) -> Vec:
        ex, ey = target[0] - self.pos[0], target[1] - self.pos[1]
        if math.hypot(ex, ey) < self.deadzone:
            ex = ey = 0.0
        w2, c = self.omega * self.omega, 2.0 * self.zeta * self.omega
        ax, ay = _clamp_norm(w2 * ex - c * self.vel[0], w2 * ey - c * self.vel[1], self.max_acc)
        vx, vy = _clamp_norm(self.vel[0] + ax * dt, self.vel[1] + ay * dt, self.max_vel)
        self.vel = (vx, vy)
        self.pos = (self.pos[0] + vx * dt, self.pos[1] + vy * dt)
        return self.pos

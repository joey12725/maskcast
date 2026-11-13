from __future__ import annotations

import dataclasses
import logging
from typing import Any, Mapping

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Tuning:
    """Every feel-related knob in one place. Override any field from config."""

    # Detection
    min_face_conf: float = 0.6

    # Layout (fractions of the window)
    eyes_max_w_scale: float = 0.55
    mouth_max_h_scale: float = 0.30
    mouth_max_y_scale: float = 0.14
    base_pos_x: float = 0.5
    base_pos_y: float = 0.42
    mouth_offset_y_scale: float = 0.27

    # Asset loading: near-black pixels become transparent, then the art is cropped
    asset_black_threshold: int = 10
    asset_trim_margin_px: int = 0
    asset_transition_s: float = 0.12

    # Gaze travel limits, in pixels from the rest position
    gaze_limit_px_left: int = 40
    gaze_limit_px_right: int = 40
    gaze_limit_px_up: int = 28
    gaze_limit_px_down: int = 32

    # Gaze dynamics (spring-damper)
    gaze_deadzone_px: float = 0.75
    gaze_omega: float = 10.0  # natural frequency, rad/s; higher = snappier
    gaze_zeta: float = 0.75  # damping ratio; 1.0 = critically damped
    gaze_max_vel_px: float = 500.0
    gaze_max_acc_px: float = 2500.0
    face_pos_ema_alpha: float = 0.12

    # What happens when the face leaves frame
    face_hold_s: float = 1.0
    return_to_center_s: float = 0.6

    # Emotion classifier
    emotion_hz: float = 2.0
    emotion_smooth_alpha: float = 0.35

    # Valence/arousal buckets for picking ``<emotion>_v?_a?`` asset variants
    valence_high_threshold: float = 0.5
    valence_low_threshold: float = -0.5
    arousal_high_threshold: float = 0.15
    arousal_low_threshold: float = -0.15

    # Speech
    keyword_flash_s: float = 1.0

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "Tuning":
        names = {f.name for f in dataclasses.fields(cls)}
        known = {k: v for k, v in (values or {}).items() if k in names}
        unknown = sorted(set(values or {}) - names)
        if unknown:
            log.warning("Ignoring unknown tuning keys: %s", ", ".join(unknown))
        return cls(**known)

    def va_thresholds(self) -> dict:
        return {
            "v_high": self.valence_high_threshold,
            "v_low": self.valence_low_threshold,
            "a_high": self.arousal_high_threshold,
            "a_low": self.arousal_low_threshold,
        }

from __future__ import annotations
import dataclasses
from typing import Optional, Tuple

@dataclasses.dataclass
class Tuning:
    min_face_conf: float = 0.6
    eyes_max_w_scale: float = 1.0/3.0
    mouth_max_h_scale: float = 0.60
    mouth_max_y_scale: float = 0.16
    game_window_xy_dim: Optional[Tuple[int, int]] = None
    gaze_pd_kp: float = 0.22
    gaze_max_step_px: float = 14.0
    gaze_limit_px_left: int = 40
    gaze_limit_px_right: int = 40
    gaze_limit_px_up: int = 28
    gaze_limit_px_down: int = 32
    face_hold_s: float = 1.0
    return_to_center_s: float = 0.6
    emotion_hz: float = 2.0
    emotion_smooth_alpha: float = 0.35
    # Thresholds for selecting valence/arousal-specific assets
    valence_high_threshold: float = 0.5
    valence_low_threshold: float = -0.5
    arousal_high_threshold: float = 0.5
    arousal_low_threshold: float = -0.5
    base_pos_x: float = 0.5
    base_pos_y: float = 0.55
    mouth_offset_y_scale: float = 0.24
    transcript_max_words: int = 5
    keyword_flash_s: float = 2.0
    gaze_deadzone_px = 0.75
    gaze_omega = 10         # higher = snappier
    gaze_zeta  = 0.75        # 1.0 = critically damped
    gaze_max_vel_px = 500.0
    gaze_max_acc_px = 2500.0
    face_pos_ema_alpha = 0.12
    face_vel_ema_alpha = 0.25

# Canonical sets and mappings
CANONICAL = ("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")

EMOTION_ALIASES = {
    "anger": "angry", "angry": "angry",
    "contempt": "contempt",
    "disgust": "disgust",
    "fear": "fear", "afraid": "fear", "scared": "fear",
    "happiness": "happy", "happy": "happy",
    "neutral": "neutral", "default": "neutral",
    "sadness": "sad", "sad": "sad",
    "surprise": "surprise", "surprised": "surprise",
}

EMOTION_TO_ENERGY = {
    "happy": 1.0, "angry": 1.0, "surprise": 1.0,
    "neutral": 0.6, "sad": 0.4, "disgust": 0.6, "fear": 0.8, "contempt": 0.6
}


KEYWORD_ALIASES = {
    "adorable": [
        "adorable", "cute", "love", "lovely", "aww", "precious", "sweet", "kawaii", "so cute", "too cute"
    ],
    "amazing": [
        "amazing", "awesome", "wow", "incredible", "epic", "unbelievable", "no way"
    ],
    "clown": [
        "clown", "clowning", "bozo", "goofy", "cringe", "what a clown"
    ],
    "cool": [
        "cool", "smooth", "chill", "deal with it", "sunglasses", "swagger", "too cool"
    ],
    "demon": [
        "demon", "devil", "oni", "possessed", "cursed", "evil"
    ],
    "money": [
        "money", "rich", "cash", "dollar", "jackpot", "make it rain", "cha-ching", "expensive"
    ],
    "owo": [
        "owo", "uwu", "nya", "nyan"
    ],
    "pog": [
        "pog", "poggers", "hype", "big win", "let's go", "so hype", "congrats", "nailed it"
    ],
    "question": [
        "question", "excuse me", "confused", "confusion", "what the"
    ],
    "sleepy": [
        "sleepy", "tired", "zzz", "yawn", "nap", "snooze", "bored", "boring", "eepy"
    ],
    "wild": [
        "wild", "crazy", "unhinged", "chaos", "trippy"
    ],
    "wink": [
        "wink", "if you know what i mean", "ba dum tss", "badum tss", "rimshot", "gotcha"
    ]
}

def canon(label: str) -> str:
    s = (label or "").lower().strip()
    return EMOTION_ALIASES.get(s, s)

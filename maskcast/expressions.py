"""Expression vocabulary and the pure logic that maps model output to art.

Nothing in here touches a camera, a GPU or a window, so it is all unit-tested.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")

# The eight classes EmotiEffLib's enet_b0_8_va_mtl head predicts.
CANONICAL = ("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")

EMOTION_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "contempt": "contempt",
    "disgust": "disgust",
    "fear": "fear",
    "afraid": "fear",
    "scared": "fear",
    "happiness": "happy",
    "happy": "happy",
    "neutral": "neutral",
    "default": "neutral",
    "sadness": "sad",
    "sad": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}

# How lively the gaze is for each emotion (scales the gaze travel).
EMOTION_TO_ENERGY = {
    "happy": 1.0,
    "angry": 1.0,
    "surprise": 1.0,
    "neutral": 0.6,
    "sad": 0.4,
    "disgust": 0.6,
    "fear": 0.8,
    "contempt": 0.6,
}

# Spoken phrases that flash a one-off expression. The key is the asset name;
# a keyword only fires if an eyes asset with that name exists.
KEYWORD_ALIASES: Dict[str, List[str]] = {
    "adorable": ["adorable", "cute", "love", "lovely", "aww", "precious", "sweet", "kawaii", "so cute", "too cute"],
    "amazing": ["amazing", "awesome", "wow", "incredible", "epic", "unbelievable", "no way"],
    "clown": ["clown", "clowning", "bozo", "goofy", "cringe", "what a clown"],
    "cool": ["cool", "smooth", "chill", "deal with it", "sunglasses", "swagger", "too cool"],
    "demon": ["demon", "devil", "oni", "possessed", "cursed", "evil"],
    "money": ["money", "rich", "cash", "dollar", "jackpot", "make it rain", "cha-ching", "expensive"],
    "owo": ["owo", "uwu", "nya", "nyan"],
    "pog": ["pog", "poggers", "hype", "big win", "let's go", "so hype", "congrats", "nailed it"],
    "question": ["question", "excuse me", "confused", "confusion", "what the"],
    "sleepy": ["sleepy", "tired", "zzz", "yawn", "nap", "snooze", "bored", "boring", "eepy"],
    "wild": ["wild", "crazy", "unhinged", "chaos", "trippy"],
    "wink": ["wink", "if you know what i mean", "ba dum tss", "badum tss", "rimshot", "gotcha"],
}


def canon(label: Optional[str]) -> str:
    """Normalise a model or file label to its canonical emotion name."""
    s = (label or "").lower().strip()
    return EMOTION_ALIASES.get(s, s)


def canonical_asset_name(name: str) -> str:
    """Canonicalise an asset stem, keeping any ``_v?_a?`` suffix.

    ``surprised_v0_a+`` -> ``surprise_v0_a+``; ``Scared`` -> ``fear``.
    """
    stem = name.lower().strip()
    base, sep, rest = stem.partition("_v")
    return canon(base) + (sep + rest if sep else "")


def level_code(value: float, high: float, low: float) -> str:
    """Bucket a valence/arousal value into ``+``, ``-`` or ``0``."""
    if value >= high:
        return "+"
    if value <= low:
        return "-"
    return "0"


def asset_candidates(
    label: str,
    valence: float,
    arousal: float,
    v_high: float = 0.5,
    v_low: float = -0.5,
    a_high: float = 0.5,
    a_low: float = -0.5,
) -> List[str]:
    """Asset names to try for an expression, most specific first.

    ``happy`` at high valence and arousal tries ``happy_v+_a+``, then
    ``happy``, then ``neutral``.
    """
    v = level_code(valence, v_high, v_low)
    a = level_code(arousal, a_high, a_low)
    names = [f"{label}_v{v}_a{a}", label]
    if label != "neutral":
        names.append("neutral")
    return names


def select_asset(
    assets: Mapping[str, T], label: str, valence: float, arousal: float, **thresholds: float
) -> Tuple[str, Optional[T], bool]:
    """Pick the best asset for ``label`` at the current valence/arousal.

    Returns ``(name, asset, used_fallback)``; ``asset`` is ``None`` and the
    name is ``"<none>"`` when nothing usable exists.
    """
    for i, name in enumerate(asset_candidates(label, valence, arousal, **thresholds)):
        if name in assets:
            return name, assets[name], i > 0
    return "<none>", None, True


def match_keyword(
    text: str, keywords: Iterable[str], aliases: Mapping[str, Sequence[str]] = KEYWORD_ALIASES
) -> Optional[str]:
    """Return the first keyword whose alias appears in the transcript window."""
    low = (text or "").lower()
    if not low:
        return None
    for kw in keywords:
        if any(alias in low for alias in aliases.get(kw, [kw])):
            return kw
    return None


def split_assets(stems: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Split asset stems into (emotions present, keyword flashes present)."""
    emo_set = set(CANONICAL) | set(EMOTION_ALIASES)
    stems = [s.lower() for s in stems]
    emotions = sorted({canon(s) for s in stems if s in emo_set})
    keywords = sorted(s for s in stems if s not in emo_set and "_v" not in s)
    return emotions, keywords


def ema_probs(old: Mapping[str, float], new: Mapping[str, float], alpha: float) -> Dict[str, float]:
    """Exponential moving average over two sparse probability maps."""
    keys = set(old) | set(new)
    return {k: alpha * new.get(k, 0.0) + (1.0 - alpha) * old.get(k, 0.0) for k in keys}


def derive_valence_arousal(probs: Mapping[str, float]) -> Optional[Tuple[float, float]]:
    """Estimate valence/arousal from class probabilities.

    Used when the recogniser has no VA head. Valence is P(happy) - P(sad);
    arousal is the margin between the top two classes (a confident, decisive
    expression reads as high energy). Returns ``None`` if the distribution is
    too degenerate to say anything.
    """
    nonzero = [p for p in probs.values() if p > 1e-3]
    if len(nonzero) < 2:
        return None
    valence = float(probs.get("happy", 0.0)) - float(probs.get("sad", 0.0))
    top = sorted(probs.values(), reverse=True)
    arousal = float(top[0] - top[1])
    return valence, arousal


def top_label(probs: Mapping[str, float], default: str = "neutral") -> str:
    if not probs:
        return default
    return canon(max(probs.items(), key=lambda kv: kv[1])[0])

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
from .tuning import CANONICAL, EMOTION_ALIASES, canon

def list_emotions_present(eyes_dir: Path) -> Tuple[List[str], List[str]]:
    names = [f.stem.lower() for f in eyes_dir.glob("*.png")] if eyes_dir.exists() else []
    emo_set = set(CANONICAL) | set(EMOTION_ALIASES.keys())
    emotions_present = sorted({canon(n) for n in names if n in emo_set})
    keywords = sorted([n for n in names if n not in emo_set])
    return emotions_present, keywords

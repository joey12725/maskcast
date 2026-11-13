from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .expressions import split_assets


def list_emotions_present(eyes_dir: Path) -> Tuple[List[str], List[str]]:
    """Scan an eyes folder: which emotions have art, and which keyword flashes."""
    stems = [f.stem for f in eyes_dir.glob("*.png")] if eyes_dir.exists() else []
    return split_assets(stems)

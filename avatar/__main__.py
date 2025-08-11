from __future__ import annotations
import argparse, json, os, multiprocessing as mp
from .config import load_config, apply_overrides
from .app import run as run_app

# Keep the same environment quieting defaults as before
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TQDM_NOPOS", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

def main():
    ap = argparse.ArgumentParser(description="Desktop Avatar — modular app")
    ap.add_argument("--config", "-c", type=str, default="configs/default.json", help="Path to config JSON")
    ap.add_argument("-o", "--override", action="append", default=[], help="Override config (dotted path), e.g. -o window.mirror=false")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    mp.set_start_method("spawn", force=True)
    run_app(cfg)

if __name__ == "__main__":
    main()

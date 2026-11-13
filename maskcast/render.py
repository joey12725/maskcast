from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .expressions import EMOTION_TO_ENERGY, canon, canonical_asset_name, select_asset
from .motion import Ema2D, SpringDamper2D, gaze_target
from .optional_deps import np, pygame
from .tuning import Tuning
from .utils import now

BACKGROUND = (10, 10, 14)


class _Layer:
    """One cross-fading slot (eyes or mouth)."""

    def __init__(self) -> None:
        self.cur = None
        self.prev = None
        self.changed_at = 0.0

    def set(self, surf) -> None:
        if surf is not self.cur:
            self.prev, self.cur, self.changed_at = self.cur, surf, now()


class Renderer:
    """Draws the face with pygame and hands each frame to ``frame_cb``."""

    def __init__(
        self,
        window_wh: Tuple[int, int],
        tuning: Tuning,
        state: Dict[str, Any],
        log: logging.Logger,
        eyes_dir: Path,
        mouths_dir: Path,
        frame_cb: Optional[Callable[["np.ndarray"], None]] = None,
        fps: int = 60,
    ):
        if pygame is None:
            raise RuntimeError("pygame not installed")
        if np is None:
            raise RuntimeError("numpy not installed (required by renderer)")

        self.log = log.getChild("Render")
        pygame.init()
        self.window_w, self.window_h = window_wh
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption("maskcast")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.state = state
        self.tuning = tuning
        self.frame_cb = frame_cb
        self.running = False

        self.eyes = self._scale_all(self._load_folder(eyes_dir), self._eyes_scaler())
        self.mouths = self._scale_all(self._load_folder(mouths_dir), self._mouth_scaler())

        self.base_center = (
            int(self.tuning.base_pos_x * self.window_w),
            int(self.tuning.base_pos_y * self.window_h),
        )
        self.mouth_center = (
            int(self.tuning.base_pos_x * self.window_w),
            int((self.tuning.base_pos_y + self.tuning.mouth_offset_y_scale) * self.window_h),
        )

        self._eyes_layer = _Layer()
        self._mouth_layer = _Layer()
        self._last_asset_name: Optional[str] = None

        self._gaze = SpringDamper2D.from_tuning(tuning)
        self._face_ema = Ema2D(tuning.face_pos_ema_alpha)

        self.log.info("Window: %dx%d | eyes centre: (%d,%d)", self.window_w, self.window_h, *self.base_center)

    # ------------------------------------------------------------------ assets
    def _load_folder(self, p: Path) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not p.exists():
            self.log.warning("Assets folder not found: %s", p)
            return out
        for f in sorted(p.glob("*.png")):
            try:
                surf = pygame.image.load(str(f)).convert_alpha()
                out[canonical_asset_name(f.stem)] = self._remove_black_and_crop(surf)
            except Exception as e:
                self.log.warning("Failed to load %s: %s", f, e)
        self.log.info("Loaded %d assets from %s", len(out), p)
        return out

    def _remove_black_and_crop(self, surf):
        """Treat near-black as transparent and crop to the visible art.

        This lets you drop in plain black-background PNGs (the usual format for
        LED-face art) without hand-editing alpha.
        """
        thresh = self.tuning.asset_black_threshold
        margin = self.tuning.asset_trim_margin_px
        rgb = pygame.surfarray.pixels3d(surf)
        alpha = pygame.surfarray.pixels_alpha(surf)
        alpha[(rgb <= thresh).all(axis=2)] = 0
        del rgb, alpha
        nz = np.argwhere(pygame.surfarray.array_alpha(surf) > 0)
        if nz.size == 0:
            return surf
        (x0, y0), (x1, y1) = nz.min(axis=0), nz.max(axis=0) + 1
        x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
        x1, y1 = min(surf.get_width(), x1 + margin), min(surf.get_height(), y1 + margin)
        return surf.subsurface(pygame.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))).copy()

    def _eyes_scaler(self):
        target_w = max(1, int(self.window_w * self.tuning.eyes_max_w_scale))

        def scale(surf):
            ow, oh = surf.get_size()
            return pygame.transform.smoothscale(surf, (target_w, max(1, round(oh * target_w / max(1, ow)))))

        return scale

    def _mouth_scaler(self):
        max_w = int(self.window_w * self.tuning.mouth_max_h_scale)
        max_h = int(self.window_h * self.tuning.mouth_max_y_scale)

        def scale(surf):
            ow, oh = surf.get_size()
            s = min(max_w / max(1, ow), max_h / max(1, oh))
            return pygame.transform.smoothscale(surf, (max(1, round(ow * s)), max(1, round(oh * s))))

        return scale

    @staticmethod
    def _scale_all(assets: Dict[str, Any], scale) -> Dict[str, Any]:
        return {name: scale(surf) for name, surf in assets.items()}

    def _pick(self, assets: Dict[str, Any], label: str):
        name, surf, fallback = select_asset(
            assets,
            label,
            float(self.state.get("valence", 0.0)),
            float(self.state.get("arousal", 0.0)),
            **self.tuning.va_thresholds(),
        )
        return name, surf, fallback

    # ------------------------------------------------------------------ frame
    def _current_label(self) -> str:
        label = canon(self.state.get("emotion_label", "neutral"))
        flash = self.state.get("flash_label")
        if flash:
            if now() <= float(self.state.get("flash_until", 0.0)):
                return flash
            self.state["flash_label"] = None
            self.state["flash_until"] = 0.0
        return label

    def _target_gaze(self, label: str):
        face = self._face_ema.update(self.state.get("face_norm", (0.0, 0.0)))
        energy = max(0.2, min(1.5, EMOTION_TO_ENERGY.get(label, 0.6)))
        return gaze_target(
            bool(self.state.get("has_face", False)),
            face,
            self.state.get("last_seen_face_norm", (0.0, 0.0)),
            now() - float(self.state.get("last_face_ts", 0.0)),
            self.tuning,
            energy,
        )

    def _draw_layer(self, layer: _Layer, center) -> bool:
        t = (now() - layer.changed_at) / max(1e-6, self.tuning.asset_transition_s)
        if layer.prev is not None and t < 1.0:
            for surf, a in ((layer.prev, 1.0 - t), (layer.cur, t)):
                if surf is not None:
                    surf.set_alpha(int(255 * a))
                    self.screen.blit(surf, surf.get_rect(center=center))
                    surf.set_alpha(255)
            return True
        layer.prev = None
        if layer.cur is not None:
            self.screen.blit(layer.cur, layer.cur.get_rect(center=center))
            return True
        return False

    def draw_frame(self, dt: float) -> None:
        self.screen.fill(BACKGROUND)
        label = self._current_label()
        gx, gy = self._gaze.step(self._target_gaze(label), dt)

        eyes_name, eyes, fb = self._pick(self.eyes, label)
        _, mouth, _ = self._pick(self.mouths, label)
        if eyes_name != self._last_asset_name:
            self.log.debug("Asset %s%s", eyes_name, " (fallback)" if fb else "")
            self._last_asset_name = eyes_name
        self._eyes_layer.set(eyes)
        self._mouth_layer.set(mouth)

        cx, cy = self.base_center[0] + round(gx), self.base_center[1] + round(gy)
        if not self._draw_layer(self._eyes_layer, (cx, cy)):
            for dx in (-30, 30):  # no art at all: draw placeholder dots
                pygame.draw.circle(self.screen, (220, 220, 220), (cx + dx, cy), 18)
        self._draw_layer(self._mouth_layer, self.mouth_center)

    def frame_array(self) -> "np.ndarray":
        """Current frame as an HxWx3 RGB array."""
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2)).copy()

    def stop(self) -> None:
        self.running = False

    def loop(self) -> None:
        self.running = True
        while self.running:
            dt = max(1e-4, self.clock.tick(self.fps) / 1000.0)
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                    self.running = False
            self.draw_frame(dt)
            pygame.display.flip()
            if self.frame_cb is not None:
                try:
                    self.frame_cb(self.frame_array())
                except Exception as e:
                    self.log.debug("frame callback failed: %s", e)
        pygame.quit()
        self.log.info("Renderer quit.")

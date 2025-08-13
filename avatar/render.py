from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Callable, Optional
from .optional_deps import pygame, np
from .tuning import Tuning, EMOTION_TO_ENERGY, canon
from .utils import now


class Renderer:
    def __init__(self, window_wh: Tuple[int, int], tuning: Tuning, state: Dict[str, Any], log: logging.Logger,
                 eyes_dir: Path, mouths_dir: Path, frame_cb: Optional[Callable[["np.ndarray"], None]] = None):
        if pygame is None:
            raise RuntimeError("pygame not installed")
        if np is None:
            raise RuntimeError("numpy not installed (required by renderer)")

        self.log = log.getChild("Render")
        pygame.init()
        self.window_w, self.window_h = window_wh
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption("Avatar")
        self.clock = pygame.time.Clock()
        self.state = state
        self.tuning = tuning
        self.frame_cb = frame_cb

        self._eyes_raw = self._load_folder(eyes_dir)
        self._mouths_raw = self._load_folder(mouths_dir)
        self.eyes = self._build_canonical_assets(self._eyes_raw)
        self.mouths = self._build_canonical_assets(self._mouths_raw)
        self._build_scaled_assets()

        self.base_center = (
            int(self.tuning.base_pos_x * self.window_w),
            int(self.tuning.base_pos_y * self.window_h),
        )

        # Asset transition state
        self.asset_transition_s = getattr(self.tuning, "asset_transition_s", 0.12)
        self._cur_eyes = None
        self._prev_eyes = None
        self._eyes_change_t = 0.0
        self._cur_mouth = None
        self._prev_mouth = None
        self._mouth_change_t = 0.0

        # Gaze state (position & velocity in pixels)
        self._gaze_px = np.array([0.0, 0.0], dtype=np.float32)
        self._gaze_vel = np.zeros(2, dtype=np.float32)

        # Smoothed face measurements (EMA)
        self._face_norm_ema = np.zeros(2, dtype=np.float32)
        self._face_vel_ema = np.zeros(2, dtype=np.float32)

        self.log.info(
            "Window: %dx%d | Base eyes center: (%d,%d)",
            self.window_w, self.window_h, *self.base_center
        )

    def _load_folder(self, p: Path):
        out = {}
        if not p.exists():
            self.log.warning("Assets folder not found: %s", p)
            return out
        for f in p.glob("*.png"):
            try:
                surf = pygame.image.load(str(f)).convert_alpha()  # type: ignore
                surf = self._remove_black_and_crop(surf)
                out[f.stem.lower()] = surf
            except Exception as e:
                self.log.warning("Failed to load %s: %s", f, e)
        self.log.info("Loaded %d assets from %s", len(out), p)
        return out

    def _build_canonical_assets(self, raw: Dict[str, "pygame.Surface"]) -> Dict[str, "pygame.Surface"]:
        out = dict(raw)
        for name, surf in list(raw.items()):
            c = canon(name)
            if c and c not in out:
                out[c] = surf
        return out

    def _scale_to_width(self, surf, target_w):
        ow, oh = surf.get_width(), surf.get_height()
        if ow == 0 or oh == 0:
            return surf
        target_h = max(1, int(round(oh * (target_w / ow))))
        return pygame.transform.smoothscale(surf, (int(target_w), int(target_h)))  # type: ignore

    def _fit_within(self, surf, max_w, max_h):
        ow, oh = surf.get_width(), surf.get_height()
        if ow == 0 or oh == 0:
            return surf
        scale = min(max_w / max(1, ow), max_h / max(1, oh))
        tw, th = max(1, int(round(ow * scale))), max(1, int(round(oh * scale)))
        return pygame.transform.smoothscale(surf, (tw, th))  # type: ignore

    def _remove_black_and_crop(self, surf):
        thresh = getattr(self.tuning, "asset_black_threshold", 10)
        margin = getattr(self.tuning, "asset_trim_margin_px", 0)
        rgb = pygame.surfarray.pixels3d(surf)
        alpha = pygame.surfarray.pixels_alpha(surf)
        mask = (
            (rgb[:, :, 0] <= thresh)
            & (rgb[:, :, 1] <= thresh)
            & (rgb[:, :, 2] <= thresh)
        )
        alpha[mask] = 0
        del rgb, alpha
        a = pygame.surfarray.array_alpha(surf)
        nz = np.argwhere(a > 0)
        if nz.size == 0:
            return surf
        x0, y0 = nz.min(axis=0)
        x1, y1 = nz.max(axis=0) + 1
        if margin > 0:
            x0 = max(0, x0 - margin)
            y0 = max(0, y0 - margin)
            x1 = min(a.shape[0], x1 + margin)
            y1 = min(a.shape[1], y1 + margin)
        rect = pygame.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        return surf.subsurface(rect).copy()

    def _build_scaled_assets(self):
        screen_w, screen_h = self.window_w, self.window_h
        eyes_target_w = max(1, int(screen_w * self.tuning.eyes_max_w_scale))
        self.eyes = {name: self._scale_to_width(surf, eyes_target_w) for name, surf in self.eyes.items()}
        max_mouth_w = int(screen_w * self.tuning.mouth_max_h_scale)
        max_mouth_h = int(screen_h * self.tuning.mouth_max_y_scale)
        self.mouths = {name: self._fit_within(surf, max_mouth_w, max_mouth_h) for name, surf in self.mouths.items()}
        self.log.info("Scaled assets: eyes_w=%d px | mouth_max=(%d×%d px)", eyes_target_w, max_mouth_w, max_mouth_h)

    def _energy_scalar(self, label: str) -> float:
        # Clamp overall energy to keep motion reasonable
        return max(0.2, min(1.5, EMOTION_TO_ENERGY.get(canon(label), 0.6)))

    def _get_asset(self, assets: Dict[str, "pygame.Surface"], label: str):
        """Return asset matching current valence/arousal if available."""
        v = float(self.state.get("valence", 0.0))
        a = float(self.state.get("arousal", 0.0))
        v_hi = getattr(self.tuning, "valence_high_threshold", 0.5)
        v_lo = getattr(self.tuning, "valence_low_threshold", -0.5)
        a_hi = getattr(self.tuning, "arousal_high_threshold", 0.5)
        a_lo = getattr(self.tuning, "arousal_low_threshold", -0.5)
        # Determine coarse classification for logging
        v_level = "high" if v >= v_hi else "low" if v <= v_lo else "medium"
        a_level = "high" if a >= a_hi else "low" if a <= a_lo else "medium"

        # Asset codes depend only on the sign of the value so that any positive
        # arousal/valence selects the corresponding "+" asset. This allows
        # medium values (e.g. arousal=0.26) to still choose assets like
        # ``happy_v+_a+``.
        v_code = "+" if v_level == "high" else "-" if v_level == "low" else "0"
        a_code = "+" if a_level == "high" else "-" if a_level == "low" else "0"
        name = f"{label}_v{v_code}_a{a_code}"

        asset = assets.get(name)
        asset_name = name
        fallback = False

        # Many asset sets omit v0 variants. If no exact match is found, fall back
        # to the closest available asset based on the sign of the valence so that
        # high arousal values still trigger an expression change.
        if asset is None and v_code == "0" and a_code != "0":
            fallback_v = "+" if v >= 0.0 else "-"
            alt_name = f"{label}"
            asset = assets.get(alt_name)
            if asset is not None:
                asset_name = alt_name
                fallback = True

        if asset is None:
            if label in assets:
                asset = assets[label]
                asset_name = label
                fallback = True
            elif "neutral" in assets:
                asset = assets["neutral"]
                asset_name = "neutral"
                fallback = True
            else:
                asset_name = "<none>"

        fb_msg = " (fallback)" if fallback else ""
        self.log.info(
            "Asset %s | valence=%s(%s) arousal=%s(%s)%s",
            asset_name,
            v_code,
            v_level,
            a_code,
            a_level,
            fb_msg,
        )
        return asset

    def _compute_target_gaze_px(self):
        # Raw measurements from tracker/state
        nx, ny = self.state.get("face_norm", (0.0, 0.0))
        vx, vy = self.state.get("face_norm_vel", (0.0, 0.0))
        last_face_ts = self.state.get("last_face_ts", 0.0)
        last_seen_norm = self.state.get("last_seen_face_norm", (0.0, 0.0))
        has_face = self.state.get("has_face", False)
        label = self.state.get("emotion_label", "neutral")
        energy = self._energy_scalar(label)

        lim_l = self.tuning.gaze_limit_px_left
        lim_r = self.tuning.gaze_limit_px_right
        lim_u = self.tuning.gaze_limit_px_up
        lim_d = self.tuning.gaze_limit_px_down

        # --- Smooth incoming face position & velocity with EMA ---
        alpha_p = getattr(self.tuning, "face_pos_ema_alpha", 0.12)
        alpha_v = getattr(self.tuning, "face_vel_ema_alpha", 0.25)
        mpos = np.array([nx, ny], dtype=np.float32)
        mvel = np.array([vx, vy], dtype=np.float32)
        self._face_norm_ema = (1.0 - alpha_p) * self._face_norm_ema + alpha_p * mpos
        self._face_vel_ema = (1.0 - alpha_v) * self._face_vel_ema + alpha_v * mvel
        nx, ny = map(float, self._face_norm_ema)
        # We keep smoothed velocity around if you want to use it later, but
        # we intentionally do NOT add feed-forward to the target anymore.

        def norm_to_px(nv):
            x, y = nv
            tx = max(-lim_l, min(lim_r, x * (lim_r if x >= 0 else lim_l) * energy))
            ty = max(-lim_u, min(lim_d, y * (lim_d if y >= 0 else lim_u) * energy))
            return np.array([tx, ty], dtype=np.float32)

        age = now() - last_face_ts
        if has_face:
            base = norm_to_px((nx, ny))
            return base  # No velocity feed-forward (prevents buzzing)
        if age <= self.tuning.face_hold_s:
            return norm_to_px(last_seen_norm)
        elif age <= self.tuning.face_hold_s + self.tuning.return_to_center_s:
            t = (age - self.tuning.face_hold_s) / max(1e-6, self.tuning.return_to_center_s)
            return (1.0 - t) * norm_to_px(last_seen_norm)
        else:
            return np.array([0.0, 0.0], dtype=np.float32)

    def _draw_centered(self, surf, center):
        rect = surf.get_rect(center=center)
        self.screen.blit(surf, rect)

    def _blend_assets(self, prev, cur, t: float, center):
        t = max(0.0, min(1.0, t))
        if prev:
            prev_alpha = prev.get_alpha()
            prev.set_alpha(int(255 * (1.0 - t)))
            self._draw_centered(prev, center)
            prev.set_alpha(prev_alpha if prev_alpha is not None else 255)
        if cur:
            cur_alpha = cur.get_alpha()
            cur.set_alpha(int(255 * t))
            self._draw_centered(cur, center)
            cur.set_alpha(cur_alpha if cur_alpha is not None else 255)

    def loop(self):
        running = True
        while running:
            # Frame timing first; dt drives physically-based integration
            dt = max(1e-4, self.clock.tick(60) / 1000.0)  # seconds

            for e in pygame.event.get():  # type: ignore
                if e.type == pygame.QUIT:
                    running = False
                if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                    running = False

            self.screen.fill((10, 10, 14))  # type: ignore

            target = self._compute_target_gaze_px()
            err = target - self._gaze_px

            # Deadzone to prevent micro-jitter when nearly settled
            deadzone = getattr(self.tuning, "gaze_deadzone_px", 0.75)
            if float(np.linalg.norm(err)) < deadzone:
                err[:] = 0.0

            # Spring-damper dynamics (natural frequency & damping ratio)
            omega = getattr(self.tuning, "gaze_omega", 8.0)   # rad/s
            zeta = getattr(self.tuning, "gaze_zeta", 0.95)    # >=1 critically damped
            acc = (omega * omega) * err - 2.0 * zeta * omega * self._gaze_vel

            # Optional clamps for stability & style
            amax = getattr(self.tuning, "gaze_max_acc_px", 2500.0)
            anorm = float(np.linalg.norm(acc))
            if anorm > amax:
                acc *= amax / max(1e-6, anorm)

            # Integrate velocity & position
            self._gaze_vel += acc * dt
            vmax = getattr(self.tuning, "gaze_max_vel_px", 500.0)
            vnorm = float(np.linalg.norm(self._gaze_vel))
            if vnorm > vmax:
                self._gaze_vel *= vmax / max(1e-6, vnorm)
            self._gaze_px += self._gaze_vel * dt

            # Draw eyes
            label = canon(self.state.get("emotion_label", "neutral"))
            flash = self.state.get("flash_label")
            flash_until = float(self.state.get("flash_until", 0.0))
            if flash and now() <= flash_until:
                label = flash
            elif flash and now() > flash_until:
                self.state["flash_label"] = None
                self.state["flash_until"] = 0.0
            new_eyes = self._get_asset(self.eyes, label)
            if new_eyes is not self._cur_eyes:
                self._prev_eyes = self._cur_eyes
                self._cur_eyes = new_eyes
                self._eyes_change_t = now()
            cx = self.base_center[0] + int(round(self._gaze_px[0]))
            cy = self.base_center[1] + int(round(self._gaze_px[1]))
            if self._prev_eyes is not None:
                t = (now() - self._eyes_change_t) / max(1e-6, self.asset_transition_s)
                if t < 1.0:
                    self._blend_assets(self._prev_eyes, self._cur_eyes, t, (cx, cy))
                else:
                    self._prev_eyes = None
                    if self._cur_eyes is not None:
                        self._draw_centered(self._cur_eyes, (cx, cy))
                    else:
                        pygame.draw.circle(self.screen, (220, 220, 220), (cx - 30, cy), 18)  # type: ignore
                        pygame.draw.circle(self.screen, (220, 220, 220), (cx + 30, cy), 18)  # type: ignore
            else:
                if self._cur_eyes is not None:
                    self._draw_centered(self._cur_eyes, (cx, cy))
                else:
                    pygame.draw.circle(self.screen, (220, 220, 220), (cx - 30, cy), 18)  # type: ignore
                    pygame.draw.circle(self.screen, (220, 220, 220), (cx + 30, cy), 18)  # type: ignore

            # Draw mouth
            new_mouth = self._get_asset(self.mouths, label)
            if new_mouth is not self._cur_mouth:
                self._prev_mouth = self._cur_mouth
                self._cur_mouth = new_mouth
                self._mouth_change_t = now()
            mouth_cx = int(self.tuning.base_pos_x * self.window_w)
            mouth_cy = int((self.tuning.base_pos_y + self.tuning.mouth_offset_y_scale) * self.window_h)
            if self._prev_mouth is not None:
                t = (now() - self._mouth_change_t) / max(1e-6, self.asset_transition_s)
                if t < 1.0:
                    self._blend_assets(self._prev_mouth, self._cur_mouth, t, (mouth_cx, mouth_cy))
                else:
                    self._prev_mouth = None
                    if self._cur_mouth is not None:
                        self._draw_centered(self._cur_mouth, (mouth_cx, mouth_cy))
            else:
                if self._cur_mouth is not None:
                    self._draw_centered(self._cur_mouth, (mouth_cx, mouth_cy))

            pygame.display.flip()  # type: ignore

            if self.frame_cb is not None:
                try:
                    arr = pygame.surfarray.array3d(self.screen)
                    frame = np.transpose(arr, (1, 0, 2)).copy()
                    self.frame_cb(frame)
                except Exception:
                    pass

        pygame.quit()             # type: ignore
        self.log.info("Renderer quit.")

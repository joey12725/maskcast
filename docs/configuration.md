# Configuration

maskcast reads `configs/default.json` (or the file passed with `-c`). Override any key from the command line with a dotted path:

```bash
maskcast -o window.mirror=false -o asr.whisper_model=medium.en -o tuning.gaze_omega=14
```

Values are parsed as booleans, numbers or `null` when they look like one, and as strings otherwise.

## Top level

| Key | Default | Meaning |
|---|---|---|
| `camera` | `0` | OpenCV camera index |
| `assets.eyes` / `assets.mouths` | `assets/eyes`, `assets/mouths` | art folders |
| `window.size` | `"auto"` | `"auto"` follows the camera aspect; or `"WxH"` / `[w, h]` (capped to 720x1080) |
| `window.mirror` | `true` | mirror horizontal movement, like a selfie camera |
| `backends.detector` | `"insightface"` | `insightface` or `haar` |
| `backends.emotion` | `"emotieff"` | `emotieff`, or anything else to disable (neutral face) |
| `asr.enabled` | `true` | speech recognition on/off |
| `asr.whisper_model` | `"small.en"` | any faster-whisper model name (`tiny.en` to `large-v3`) |
| `asr.device` | `"auto"` | `auto` or `cpu` |
| `plots.enabled` | `false` | telemetry dashboard |
| `plots.hz`, `plots.seconds` | `10`, `180` | dashboard update rate and history length |
| `remote.enabled` | `false` | WebRTC server; browser camera/mic replace the local ones |
| `remote.host`, `remote.port` | `127.0.0.1`, `8080` | bind address; use `0.0.0.0` to accept LAN clients |
| `mask.enabled` | `false` | stream frames to a BLE LED mask |
| `mask.width`, `mask.height` | `45`, `60` | LED panel resolution |
| `mask.name_prefix` | `"Mask"` | BLE advertised-name prefix to connect to |
| `logging.debug` / `logging.trace` | `false` | INFO / DEBUG logging, including a per-tick status line |

## `tuning`

Every field of `maskcast.tuning.Tuning` can be set here. Unknown keys are ignored with a warning.

### Layout

| Key | Default | Meaning |
|---|---|---|
| `eyes_max_w_scale` | `0.55` | eye art width as a fraction of window width |
| `mouth_max_h_scale` | `0.30` | max mouth width as a fraction of window width |
| `mouth_max_y_scale` | `0.14` | max mouth height as a fraction of window height |
| `base_pos_x`, `base_pos_y` | `0.5`, `0.42` | resting eye centre (fractions of the window) |
| `mouth_offset_y_scale` | `0.27` | mouth centre, below the eyes |
| `asset_black_threshold` | `10` | RGB values at or below this become transparent |
| `asset_trim_margin_px` | `0` | padding kept around cropped art |
| `asset_transition_s` | `0.12` | cross-fade duration |

### Gaze

| Key | Default | Meaning |
|---|---|---|
| `gaze_limit_px_left/right/up/down` | `40/40/28/32` | max travel from rest, per direction |
| `gaze_omega` | `10` | spring natural frequency (rad/s); higher is snappier |
| `gaze_zeta` | `0.75` | damping ratio; `1.0` is critically damped, lower overshoots |
| `gaze_max_vel_px`, `gaze_max_acc_px` | `500`, `2500` | motion clamps |
| `gaze_deadzone_px` | `0.75` | ignore errors smaller than this |
| `face_pos_ema_alpha` | `0.12` | smoothing on the tracked face position |
| `face_hold_s` | `1.0` | keep looking at the last position after the face is lost |
| `return_to_center_s` | `0.6` | then ease back to centre over this long |

### Emotion

| Key | Default | Meaning |
|---|---|---|
| `min_face_conf` | `0.6` | detector score threshold |
| `emotion_hz` | `2.0` | emotion model rate |
| `emotion_smooth_alpha` | `0.35` | EMA on class probabilities |
| `valence_high_threshold` / `valence_low_threshold` | `0.5` / `-0.5` | bucket edges for `_v+` / `_v-` |
| `arousal_high_threshold` / `arousal_low_threshold` | `0.5` / `-0.5` | bucket edges for `_a+` / `_a-` |
| `keyword_flash_s` | `1.0` | how long a spoken keyword holds its expression |

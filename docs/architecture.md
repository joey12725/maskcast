# Architecture

maskcast is a set of producer threads writing to one shared dictionary (the *state*) and a render loop reading from it. Nothing waits on anything else: if the emotion model takes 80 ms, the eyes keep moving at 60 fps on the last known values.

```
              ┌─────────────┐   face pos, bbox        ┌─────────────┐
 camera ────▶ │ VideoWorker │ ──────────────────────▶ │             │
 (or WebRTC)  │  SCRFD/Haar │   emotion, V/A (2 Hz)   │             │
              │  EmotiEff   │ ──────────────────────▶ │    state    │ ──▶ Renderer (main thread, 60 fps)
              └─────────────┘                         │   (dict)    │        │
              ┌─────────────┐   transcript window     │             │        ├──▶ window
 mic ───────▶ │ WhisperASR  │ ──────────────────────▶ │             │        ├──▶ WebRTC outgoing track
 (or WebRTC)  └─────────────┘                         │             │        └──▶ BLE LED mask
              ┌─────────────┐   keyword flash         │             │
              │  Reporter   │ ◀────── reads ───────── │             │ ──▶ plot process (mp.Queue)
              └─────────────┘ ─────── writes ───────▶ └─────────────┘
```

## Perception

### Face detection (`detectors.py`)

InsightFace's `buffalo_l` pack provides SCRFD, a fast single-stage face detector. It runs through ONNX Runtime, and `hw.onnxruntime_providers()` builds the provider list in priority order for the host OS: CoreML on macOS, DirectML then CUDA on Windows, CUDA then ROCm on Linux, CPU last. If InsightFace or ONNX Runtime is missing, the detector falls back to OpenCV's frontal-face Haar cascade so the app still runs on a bare install.

Detections below `tuning.min_face_conf` are dropped and the largest remaining face wins. Its centre is normalised to `[-1, 1]` (mirrored by default, so moving left moves the eyes left) and a velocity estimate is kept alongside.

### Emotion and affect (`emotions.py`, `video.py`)

EmotiEffLib's `enet_b0_8_va_mtl` is an EfficientNet-B0 trained multi-task on AffectNet. One forward pass gives:

- probabilities over 8 classes (anger, contempt, disgust, fear, happiness, neutral, sadness, surprise)
- **valence** (unpleasant to pleasant) and **arousal** (calm to excited) as continuous values

The model runs at `tuning.emotion_hz` (2 Hz by default) on the face crop. Class probabilities are smoothed with an exponential moving average (`emotion_smooth_alpha`) before taking the top label, which stops the face from flickering between neighbouring classes.

If the recogniser reports no valence/arousal (a different model, or a failed head), `expressions.derive_valence_arousal()` estimates them from the class distribution: valence is P(happy) - P(sad), and arousal is the margin between the top two classes. The plotter flags when this fallback is active.

The Torch device is chosen by `hw.pick_torch_device()`: CUDA, then Apple MPS, then CPU. A small shim handles PyTorch 2.6's `weights_only` default for the pickled EmotiEffLib checkpoint.

### Speech (`asr.py`)

faster-whisper (Whisper on CTranslate2) transcribes a rolling 6-second, 16 kHz mono window with voice-activity detection. When the window fills and the input is louder than -55 dBFS, it transcribes, then slides forward by a third, so each phrase gets several chances to be heard in context. Compute type is float16 on CUDA and int8 elsewhere.

Audio comes from `sounddevice` locally, or from the WebRTC audio track (resampled to 16 kHz s16 mono with PyAV) in remote mode.

## Expression mapping (`expressions.py`)

The renderer resolves art in this order:

1. A **keyword flash**, if one is active. The Reporter thread checks the transcript against `KEYWORD_ALIASES` and arms a flash for `keyword_flash_s` seconds. Keywords are only active when an eyes image with that name exists.
2. A **valence/arousal variant**: valence and arousal are bucketed into `+`, `0` or `-` with the thresholds in `tuning`, giving names like `happy_v+_a+`.
3. The **base emotion** (`happy`).
4. `neutral`.

All of this is pure code with no I/O, and it is what most of the test suite covers.

## Rendering (`render.py`, `motion.py`)

Assets are loaded once. Near-black pixels are keyed to transparent and each image is cropped to its visible bounds, then scaled to the window: eyes to a fixed fraction of the width, mouths to fit a box. Changing expression cross-fades the old and new art over `asset_transition_s`.

Gaze is a two-stage filter:

1. The face position gets an EMA (`face_pos_ema_alpha`) and is mapped to a pixel offset with separate left/right/up/down travel limits, scaled by an emotion-dependent *energy* (surprised eyes travel further than sad ones). If the face disappears, the target holds for `face_hold_s`, then fades linearly to centre over `return_to_center_s`.
2. A second-order **spring-damper** chases that target:

   ```
   a = ω²·(target − pos) − 2ζω·vel
   ```

   with `ω = gaze_omega` and `ζ = gaze_zeta`. Slightly underdamped (`ζ < 1`) gives a small, lifelike overshoot. Acceleration and velocity are clamped, and errors under `gaze_deadzone_px` are ignored so the eyes rest instead of buzzing.

Each finished frame is copied to an RGB array and handed to every enabled output.

## Outputs

### WebRTC (`remote.py`, `web/remote.html`)

An aiohttp server serves the client page at `/` and does SDP offer/answer at `/offer`. The browser sends its camera and microphone; the server pushes incoming video frames into the same queue the VideoWorker would read from a webcam, and resampled audio into the Whisper queue. The rendered face goes back as an `aiortc` video track. A data channel carries ping/pong messages, and half the round trip is shown as `rtc` latency in the logs and plots.

All queues hold the newest item only (size 1 for video), so a slow network drops frames instead of building delay.

### BLE LED mask (`mask.py`)

Some consumer LED face masks accept images over Bluetooth LE. Commands are AES-128-ECB encrypted with the masks' fixed protocol key; image data is sent in plaintext chunks. An upload is:

```
DATS <len> <len>   → wait DATSOK
DATA [n+1, idx, payload…] × ⌈len/90⌉   → wait REOK after each
DATCP              → wait DATCPOK
PLAY 1             → wait PLAYOK
```

`MaskOutput` runs the async `bleak` client on its own event loop thread. Like the WebRTC path, it keeps one pending frame, so the mask always shows the latest expression rather than a backlog.

### Telemetry (`plotter.py`)

When `plots.enabled` is on, a separate process (spawned, so Matplotlib's GUI loop never touches the render thread) plots valence, arousal, per-stage latency, the eight class probabilities, face x/y and the transcript history.

## Threads at a glance

| Thread | Work | Rate |
|---|---|---|
| main | pygame render loop | 60 fps |
| Video | capture, detect, classify | camera rate; emotion at 2 Hz |
| ASR-Whisper | transcribe rolling window | per 2 s of new audio |
| Reporter | keyword matching, status log, plot feed | `plots.hz` |
| WebRTC | aiohttp + aiortc event loop | event driven |
| LEDMask | bleak event loop | as fast as BLE allows |
| Demo | scripted state (with `--demo`, replaces Video and ASR) | 30 Hz |

On Linux, the video and ASR threads pin themselves to separate cores.

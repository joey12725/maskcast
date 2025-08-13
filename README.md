# maskcast

A lightweight desktop avatar that tracks your webcam, classifies your expression and speech, and renders anime-style eyes and mouths in real time. The project can also stream the avatar over WebRTC so it can be used remotely.

## Features

- Face detection via InsightFace with OpenCV Haar cascade fallback.
- Emotion recognition through [EmotiEffLib](https://github.com/EmotionEvaluation/EmotiEff).
- Whisper-based automatic speech recognition (ASR).
- Renders PNG assets from `eyes/` and `mouths/` to display different expressions.
- Optional WebRTC server that can stream the avatar to a browser client.

## Requirements

- Python 3.10+
- Recommended Python packages:
  ```bash
  pip install aiohttp aiortc av numpy opencv-python pygame sounddevice faster-whisper torch emotiefflib onnxruntime insightface
  ```
  GPU acceleration is used automatically if CUDA is available.

## Running the server

1. Inspect the default configuration in `configs/default.json`. It controls camera index, asset paths, window options, backends, ASR and the remote server.
2. Start the avatar using:
   ```bash
   python -m avatar -c configs/default.json
   ```
   Configuration values can be overridden from the command line using `-o key=value` pairs, e.g. `-o window.mirror=false`.
3. When the `remote.enabled` flag is true the app launches a WebRTC server and begins forwarding frames and audio.

## Remote client

The repo includes a minimal browser client in `remote.html`. When the remote server is enabled, browse to `http://<server>:8080/` or open `remote.html` directly. When opened from the filesystem it defaults to connecting to `http://localhost:8080` or you can specify another server via a query parameter, e.g. `remote.html?server=http://<server>:8080`. The page captures your webcam and microphone, sends them to the server and displays the rendered avatar stream alongside the local preview.

Enable the remote server by setting the `remote` block in the config:
```json
"remote": {
  "enabled": true,
  "host": "0.0.0.0",
  "port": 8080
}
```
With this enabled you can open the client from another machine on the network.

## Project structure

```
avatar/        Core application modules and WebRTC server
configs/       JSON configuration files
eyes/          Eye asset images
mouths/        Mouth asset images
remote.html    Browser client for remote streaming
```

## Notes

- Custom eye and mouth images can be added to the respective folders to extend available expressions.
- ASR, plotting and remote streaming can be toggled on/off in the configuration.
- The application shuts down cleanly with `Ctrl+C`.


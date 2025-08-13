from __future__ import annotations
import asyncio, logging, threading, time
from pathlib import Path
from typing import Optional, Dict, Any
from queue import Queue, Full

try:
    from aiohttp import web
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaRelay
    import av
except Exception as e:  # pragma: no cover - import guard
    raise ImportError(
        "Remote control requires optional dependencies 'aiohttp', 'aiortc' and 'av'."
    ) from e


class OutgoingVideoTrack(VideoStreamTrack):
    kind = "video"
    def __init__(self, q: Queue):
        super().__init__()
        self.q = q

    async def recv(self):
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, self.q.get)
        vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
        pts, time_base = await self.next_timestamp()
        vf.pts = pts
        vf.time_base = time_base
        return vf


class WebRTCServer(threading.Thread):
    def __init__(self, video_queue: Queue, audio_queue: Queue, outgoing_queue: Queue, state: Optional[Dict[str, Any]] = None, host: str = "0.0.0.0", port: int = 8080):
        super().__init__(daemon=True, name="WebRTC")
        self.video_queue = video_queue
        self.audio_queue = audio_queue
        self.outgoing_queue = outgoing_queue
        self.state = state
        self.host = host
        self.port = port
        self.log = logging.getLogger("WebRTC")
        self._stop_evt = threading.Event()
        self.pcs = set()

    def send_frame(self, frame):
        try:
            self.outgoing_queue.put_nowait(frame)
        except Full:
            try:
                _ = self.outgoing_queue.get_nowait()
            except Exception:
                pass
            try:
                self.outgoing_queue.put_nowait(frame)
            except Exception:
                pass

    def stop(self):
        self._stop_evt.set()

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_post("/offer", self.offer)
        app.router.add_options("/offer", self.preflight)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self.log.info("WebRTC server listening on %s:%d", self.host, self.port)
        while not self._stop_evt.is_set():
            await asyncio.sleep(0.1)
        for pc in list(self.pcs):
            await pc.close()
        await runner.cleanup()

    async def index(self, request):
        html = (Path(__file__).resolve().parent.parent / "remote.html").read_text()
        return web.Response(
            content_type="text/html", text=html, headers={"Access-Control-Allow-Origin": "*"}
        )

    async def offer(self, request):
        sdp = await request.text()
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        pc = RTCPeerConnection()
        self.pcs.add(pc)
        relay = MediaRelay()

        channel = pc.createDataChannel("latency")

        async def pinger():
            while True:
                await asyncio.sleep(1.0)
                if channel.readyState != "open":
                    continue
                try:
                    channel.send(f"ping:{time.time()}")
                except Exception:
                    pass

        asyncio.create_task(pinger())

        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("pong:"):
                try:
                    ts = float(message.split(":", 1)[1])
                    rtt_ms = (time.time() - ts) * 1000.0
                    if self.state is not None:
                        self.state["remote_latency_ms"] = rtt_ms / 2.0
                except Exception:
                    pass

        @pc.on("track")
        async def on_track(track):
            if track.kind == "video":
                asyncio.create_task(self._handle_video(relay.subscribe(track)))
            elif track.kind == "audio":
                asyncio.create_task(self._handle_audio(relay.subscribe(track)))

        pc.addTrack(OutgoingVideoTrack(self.outgoing_queue))
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.Response(
            content_type="application/sdp",
            text=pc.localDescription.sdp,
            headers={"Access-Control-Allow-Origin": "*"},
        )

    async def preflight(self, request):
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        }
        return web.Response(status=204, headers=headers)

    async def _handle_video(self, track):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="rgb24")
            try:
                self.video_queue.put_nowait(img)
            except Full:
                try:
                    _ = self.video_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self.video_queue.put_nowait(img)
                except Exception:
                    pass

    async def _handle_audio(self, track):
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        while True:
            frame = await track.recv()
            for out in resampler.resample(frame):
                pcm = out.to_ndarray().flatten()
                try:
                    self.audio_queue.put_nowait(pcm)
                except Full:
                    try:
                        _ = self.audio_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self.audio_queue.put_nowait(pcm)
                    except Exception:
                        pass

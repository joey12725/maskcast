"""Bluetooth LE output to a "Shining Mask" style LED face mask.

The packet framing below is pure Python and unit-tested. The BLE transport
(``bleak``) and AES (``pycryptodome``) are imported lazily, so the rest of
maskcast runs without them. Install with ``pip install -e '.[mask]'``.

The AES key is the fixed, publicly documented key these masks use for their
command channel. It is a protocol constant, not a credential.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Any, Iterator, Optional

from .optional_deps import cv2, np

CMD_UUID = "d44bc439-abfd-45a2-b575-925416129600"
NOTIFY_UUID = "d44bc439-abfd-45a2-b575-925416129601"
DATA_UUID = "d44bc439-abfd-45a2-b575-92541612960a"

PROTOCOL_KEY = bytes.fromhex("32672f7974ad43451d9c6c894a0e8764")

CHUNK_SIZE = 90  # payload bytes per DATA packet


# --------------------------------------------------------------- framing
def pad16(payload: bytes) -> bytes:
    """Zero-pad a command to one AES block."""
    if len(payload) > 16:
        raise ValueError("command longer than one AES block")
    return payload + b"\x00" * (16 - len(payload))


def dats_command(total_len: int) -> bytes:
    """``DATS``: announce an upload of ``total_len`` bytes into slot 0."""
    n = total_len.to_bytes(2, "big")
    return bytes([9]) + b"DATS" + n + n + b"\x00"


def datcp_command() -> bytes:
    """``DATCP``: upload complete."""
    return b"\x05DATCP"


def play_command(slot: int = 1) -> bytes:
    """``PLAY``: show the uploaded image."""
    return bytes([6]) + b"PLAY" + bytes([slot, 0])


def data_packets(data: bytes, chunk_size: int = CHUNK_SIZE) -> Iterator[bytes]:
    """Split image bytes into ``[len+1, index, ...payload]`` DATA packets."""
    for idx, off in enumerate(range(0, len(data), chunk_size)):
        chunk = data[off : off + chunk_size]
        yield bytes([len(chunk) + 1, idx & 0xFF]) + chunk


def parse_notify(decrypted: bytes) -> str:
    """Notifications are ``[len, ascii...]`` after decryption."""
    n = decrypted[0]
    return decrypted[1 : 1 + n].decode("ascii", errors="ignore")


def resize_nearest(frame: Any, width: int, height: int) -> Any:
    """Downscale an HxWx3 frame to the mask's resolution."""
    if cv2 is not None:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8)
    y_idx = np.linspace(0, frame.shape[0] - 1, height).astype(np.intp)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).astype(np.intp)
    return frame[y_idx[:, None], x_idx[None, :]].astype(np.uint8)


# ------------------------------------------------------------- transport
class MaskClient:
    """Async BLE client. Keeps at most one pending frame, so latency never builds up."""

    def __init__(self, width: int = 45, height: int = 60, name_prefix: str = "Mask") -> None:
        if np is None:
            raise RuntimeError("numpy is required for MaskClient")
        from Crypto.Cipher import AES  # lazy: optional dependency

        self.width = int(width)
        self.height = int(height)
        self.name_prefix = name_prefix
        self.client: Any = None
        self._cipher = AES.new(PROTOCOL_KEY, AES.MODE_ECB)
        self._queue: Optional[asyncio.Queue] = None
        self._notify_q: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _encrypt(self, payload: bytes) -> bytes:
        return self._cipher.encrypt(pad16(payload))

    def _handle_notify(self, _: Any, data: bytearray) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self._notify_q.put_nowait(parse_notify(self._cipher.decrypt(bytes(data))))

    async def _wait_for(self, keyword: str, timeout: float = 5.0) -> None:
        while True:
            msg = await asyncio.wait_for(self._notify_q.get(), timeout)
            if keyword in msg:
                return

    async def start(self) -> None:
        from bleak import BleakClient, BleakScanner  # lazy: optional dependency

        self._queue = asyncio.Queue(maxsize=1)
        self._notify_q = asyncio.Queue()
        device = await BleakScanner.find_device_by_filter(
            lambda d, adv: bool(d.name) and d.name.startswith(self.name_prefix)
        )
        if device is None:
            raise RuntimeError(f"No BLE device named {self.name_prefix}* found")
        self.client = BleakClient(device)
        await self.client.connect()
        await self.client.start_notify(NOTIFY_UUID, self._handle_notify)
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._sender())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task
        if self.client and self.client.is_connected:
            await self.client.stop_notify(NOTIFY_UUID)
            await self.client.disconnect()

    def send(self, frame: Any) -> None:
        """Queue a frame from any thread. The newest frame always wins."""
        if self._loop is None:
            return

        def put() -> None:
            if self._queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
            self._queue.put_nowait(frame)

        self._loop.call_soon_threadsafe(put)

    async def _sender(self) -> None:
        while True:
            frame = await self._queue.get()
            try:
                await self._upload_frame(frame)
            except Exception:
                pass  # transient BLE errors: keep streaming

    async def _upload_frame(self, frame: Any) -> None:
        data = resize_nearest(frame, self.width, self.height).tobytes()
        write = self.client.write_gatt_char
        await write(CMD_UUID, self._encrypt(dats_command(len(data))), response=False)
        await self._wait_for("DATSOK")
        for pkt in data_packets(data):
            await write(DATA_UUID, pkt, response=False)
            await self._wait_for("REOK")
        await write(CMD_UUID, self._encrypt(datcp_command()), response=False)
        await self._wait_for("DATCPOK")
        await write(CMD_UUID, self._encrypt(play_command()), response=False)
        await self._wait_for("PLAYOK")


class MaskOutput(threading.Thread):
    """Runs a MaskClient on its own event loop so the renderer can call ``send``."""

    def __init__(self, width: int, height: int, name_prefix: str = "Mask") -> None:
        super().__init__(daemon=True, name="LEDMask")
        self.log = logging.getLogger("LEDMask")
        self.client = MaskClient(width, height, name_prefix)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self.client.start())
            self.log.info("LED mask connected (%dx%d)", self.client.width, self.client.height)
            self._loop.run_forever()
        except Exception as e:
            self.log.error("LED mask output disabled: %s", e)
        finally:
            with contextlib.suppress(Exception):
                self._loop.run_until_complete(self.client.stop())
            self._loop.close()

    def send(self, frame: Any) -> None:
        self.client.send(frame)

    def stop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

from __future__ import annotations
import asyncio
import contextlib
from typing import Optional

from bleak import BleakClient, BleakScanner
from Crypto.Cipher import AES

from .optional_deps import np, cv2

CMD_UUID = "d44bc439-abfd-45a2-b575-925416129600"
NOTIFY_UUID = "d44bc439-abfd-45a2-b575-925416129601"
DATA_UUID = "d44bc439-abfd-45a2-b575-92541612960a"

AES_KEY = bytes([
    0x32, 0x67, 0x2F, 0x79, 0x74, 0xAD, 0x43, 0x45,
    0x1D, 0x9C, 0x6C, 0x89, 0x4A, 0x0E, 0x87, 0x64,
])


class MaskClient:
    """Minimal Shining Mask client.

    Frames are queued and uploaded sequentially. The queue holds at most one
    frame so new frames replace older ones, preventing latency build up.
    """

    def __init__(self, width: int = 45, height: int = 60) -> None:
        if np is None:
            raise RuntimeError("numpy is required for MaskClient")
        self.width = int(width)
        self.height = int(height)
        self.client: Optional[BleakClient] = None
        self._cipher = AES.new(AES_KEY, AES.MODE_ECB)
        self._queue: asyncio.Queue["np.ndarray"] = asyncio.Queue(maxsize=1)
        self._notify_q: asyncio.Queue[str] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------ utils
    def _encrypt(self, payload: bytes) -> bytes:
        if len(payload) < 16:
            payload = payload + b"\x00" * (16 - len(payload))
        return self._cipher.encrypt(payload)

    def _handle_notify(self, _: str, data: bytearray) -> None:
        decrypted = self._cipher.decrypt(bytes(data))
        n = decrypted[0]
        msg = decrypted[1:1 + n].decode("ascii", errors="ignore")
        try:
            self._notify_q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    async def _wait_for(self, keyword: str, timeout: float = 5.0) -> None:
        while True:
            msg = await asyncio.wait_for(self._notify_q.get(), timeout)
            if keyword in msg:
                return

    def _resize(self, frame: "np.ndarray") -> "np.ndarray":
        if cv2 is not None:
            resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            y_idx = (np.linspace(0, frame.shape[0] - 1, self.height)).astype(np.intp)
            x_idx = (np.linspace(0, frame.shape[1] - 1, self.width)).astype(np.intp)
            resized = frame[y_idx[:, None], x_idx[None, :]]
        return resized.astype(np.uint8)

    # ----------------------------------------------------------------- public
    async def start(self) -> None:
        device = await BleakScanner.find_device_by_filter(
            lambda d, adv: d.name and d.name.startswith("Mask")
        )
        if device is None:
            raise RuntimeError("Mask device not found")
        self.client = BleakClient(device)
        await self.client.connect()
        await self.client.start_notify(NOTIFY_UUID, self._handle_notify)
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._sender())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(Exception):
                await self._task
        if self.client and self.client.is_connected:
            await self.client.stop_notify(NOTIFY_UUID)
            await self.client.disconnect()

    def send(self, frame: np.ndarray) -> None:
        """Queue a frame for display on the mask.

        Can be called from any thread. The newest frame always wins; older
        frames are dropped if necessary.
        """
        if self._loop is None:
            raise RuntimeError("MaskClient not started")

        def put() -> None:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(frame)

        self._loop.call_soon_threadsafe(put)

    # --------------------------------------------------------------- internals
    async def _sender(self) -> None:
        assert self.client is not None
        while True:
            frame = await self._queue.get()
            try:
                await self._upload_frame(frame)
            except Exception:
                # Ignore transient errors to keep loop running
                pass

    async def _upload_frame(self, frame: "np.ndarray") -> None:
        assert self.client is not None
        data = self._resize(frame).tobytes()
        total_len = len(data)

        # Initiate upload
        dats = bytearray([9]) + b"DATS" + total_len.to_bytes(2, "big") + total_len.to_bytes(2, "big") + b"\x00"
        await self.client.write_gatt_char(CMD_UUID, self._encrypt(bytes(dats)), response=False)
        await self._wait_for("DATSOK")

        # Data chunks
        chunk_sz = 90  # bytes per chunk payload
        idx = 0
        for off in range(0, total_len, chunk_sz):
            chunk = data[off:off + chunk_sz]
            pkt = bytes([len(chunk) + 1, idx]) + chunk
            await self.client.write_gatt_char(DATA_UUID, pkt, response=False)
            idx += 1
            await self._wait_for("REOK")

        # Complete upload
        await self.client.write_gatt_char(CMD_UUID, self._encrypt(b"\x05DATCP"), response=False)
        await self._wait_for("DATCPOK")

        # Show image from slot 0
        play = bytearray([6]) + b"PLAY" + b"\x01\x00"
        await self.client.write_gatt_char(CMD_UUID, self._encrypt(bytes(play)), response=False)
        await self._wait_for("PLAYOK")

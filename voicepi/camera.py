from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class CameraStatus:
    enabled: bool
    running: bool
    ok: bool
    error: str | None = None
    frame_id: int = 0
    last_frame_age_s: float | None = None
    size: tuple[int, int] = (640, 480)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self.running,
            "ok": self.ok,
            "error": self.error,
            "frame_id": self.frame_id,
            "last_frame_age_s": None if self.last_frame_age_s is None else round(self.last_frame_age_s, 3),
            "size": list(self.size),
        }


class StreamingOutput(io.BufferedIOBase):
    """Picamera2 FileOutput target that stores the latest JPEG frame."""

    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.frame_id = 0
        self.last_frame_at = 0.0
        self.condition = threading.Condition()

    def write(self, buf: bytes | bytearray | memoryview) -> int:
        data = bytes(buf)
        with self.condition:
            self.frame = data
            self.frame_id += 1
            self.last_frame_at = time.monotonic()
            self.condition.notify_all()
        return len(data)

    def latest(self) -> tuple[int, bytes | None, float | None]:
        with self.condition:
            age = None if self.last_frame_at <= 0 else time.monotonic() - self.last_frame_at
            return self.frame_id, self.frame, age

    def wait_next(self, after_frame_id: int = 0, timeout_s: float = 5.0) -> tuple[int, bytes | None, float | None]:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.frame_id <= after_frame_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=remaining)
            age = None if self.last_frame_at <= 0 else time.monotonic() - self.last_frame_at
            return self.frame_id, self.frame, age


class CameraManager:
    """Small Picamera2 wrapper used by Flask routes, UI MJPEG stream and vision snapshots."""

    def __init__(self, cfg, logger=None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.enabled = bool(cfg.get("camera.enabled", False))
        self.size = (
            int(cfg.get("camera.width", 640)),
            int(cfg.get("camera.height", 480)),
        )
        self.output = StreamingOutput()
        self.picam2 = None
        self.running = False
        self.error: str | None = None
        self.lock = threading.RLock()

    def start(self) -> None:
        if not self.enabled:
            return
        with self.lock:
            if self.running:
                return
            try:
                from picamera2 import Picamera2  # type: ignore
                from picamera2.encoders import JpegEncoder  # type: ignore
                from picamera2.outputs import FileOutput  # type: ignore

                picam2 = Picamera2()
                config = picam2.create_video_configuration(main={"size": self.size})
                picam2.configure(config)
                picam2.start_recording(JpegEncoder(), FileOutput(self.output))
                self.picam2 = picam2
                self.running = True
                self.error = None
                self._log("camera", "info", "camera started", size=self.size)
            except Exception as exc:
                self.running = False
                self.error = str(exc)
                self._log("camera", "error", f"camera failed to start: {exc}")

    def stop(self) -> None:
        with self.lock:
            if self.picam2 is not None:
                try:
                    self.picam2.stop_recording()
                except Exception as exc:
                    self._log("camera", "warning", f"camera stop failed: {exc}")
            self.picam2 = None
            self.running = False

    def status(self) -> CameraStatus:
        frame_id, frame, age = self.output.latest()
        ok = self.enabled and self.running and frame is not None and self.error is None
        return CameraStatus(
            enabled=self.enabled,
            running=self.running,
            ok=ok,
            error=self.error,
            frame_id=frame_id,
            last_frame_age_s=age,
            size=self.size,
        )

    def latest_jpeg(self) -> bytes | None:
        _, frame, _ = self.output.latest()
        return frame

    def mjpeg_stream(self) -> Iterator[bytes]:
        last_frame_id = 0
        while True:
            frame_id, frame, _ = self.output.wait_next(after_frame_id=last_frame_id, timeout_s=5.0)
            if frame is None:
                continue
            last_frame_id = frame_id
            yield b"--FRAME\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(frame)}\r\n".encode("ascii")
            yield b"\r\n"
            yield frame
            yield b"\r\n"

    def _log(self, source: str, level: str, message: str, **fields: Any) -> None:
        if self.logger is not None:
            try:
                self.logger.event(source, level, message, **fields)
            except Exception:
                pass

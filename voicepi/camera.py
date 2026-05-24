"""Camera manager with Picamera2 (CSI) primary and cv2.VideoCapture (USB) fallback."""
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
    backend: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self.running,
            "ok": self.ok,
            "error": self.error,
            "frame_id": self.frame_id,
            "last_frame_age_s": None if self.last_frame_age_s is None else round(self.last_frame_age_s, 3),
            "size": list(self.size),
            "backend": self.backend,
        }


class StreamingOutput(io.BufferedIOBase):
    """Shared frame buffer written by camera backend, read by HTTP routes."""

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
    """Camera wrapper: tries Picamera2 (CSI ribbon cable) first, falls back to cv2 (USB).

    On Raspberry Pi OS you must enable the camera in raspi-config or /boot/config.txt.
    For USB cameras the fallback activates automatically.
    """

    def __init__(self, cfg, logger=None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.enabled = bool(cfg.get("camera.enabled", False))
        self.size = (
            int(cfg.get("camera.width", 640)),
            int(cfg.get("camera.height", 480)),
        )
        # Which device index to try for cv2 fallback (0 = /dev/video0).
        self._cv2_index = int(cfg.get("camera.cv2_device_index", 0))

        self.output = StreamingOutput()
        self._picam2 = None
        self._cv2_thread: threading.Thread | None = None
        self._cv2_stop = threading.Event()
        self.running = False
        self.error: str | None = None
        self._backend = "none"
        self.lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            return
        with self.lock:
            if self.running:
                return
            # Try Picamera2 first (CSI camera), then fall back to cv2 (USB / V4L2).
            if self._try_picamera2():
                return
            self._log("camera", "info", "Picamera2 unavailable -- trying cv2.VideoCapture fallback")
            self._try_cv2()

    def stop(self) -> None:
        with self.lock:
            # Stop Picamera2
            if self._picam2 is not None:
                try:
                    self._picam2.stop_recording()
                except Exception as exc:
                    self._log("camera", "warning", f"picamera2 stop failed: {exc}")
                self._picam2 = None

            # Stop cv2 thread
            self._cv2_stop.set()
            if self._cv2_thread and self._cv2_thread.is_alive():
                self._cv2_thread.join(timeout=3.0)
            self._cv2_thread = None
            self._cv2_stop.clear()

            self.running = False
            self._backend = "none"

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
            backend=self._backend,
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

    # ------------------------------------------------------------------
    # Private -- Picamera2 backend
    # ------------------------------------------------------------------

    def _try_picamera2(self) -> bool:
        """Returns True if Picamera2 started successfully."""
        try:
            from picamera2 import Picamera2  # type: ignore
            from picamera2.encoders import JpegEncoder  # type: ignore
            from picamera2.outputs import FileOutput  # type: ignore

            picam2 = Picamera2()
            # Prefer a format that JpegEncoder can handle on all Pi generations.
            video_config = picam2.create_video_configuration(
                main={"size": self.size, "format": "RGB888"},
                controls={"FrameRate": 10},
            )
            picam2.configure(video_config)
            picam2.start_recording(JpegEncoder(), FileOutput(self.output))
            self._picam2 = picam2
            self.running = True
            self.error = None
            self._backend = "picamera2"
            self._log("camera", "info", "camera started (picamera2)", size=self.size)
            return True

        except ImportError:
            self._log("camera", "info", "picamera2 not installed -- will try cv2")
        except Exception as exc:
            self._log("camera", "warning", f"picamera2 failed: {exc} -- will try cv2")
        return False

    # ------------------------------------------------------------------
    # Private -- cv2 / V4L2 fallback backend
    # ------------------------------------------------------------------

    def _try_cv2(self) -> None:
        try:
            import cv2  # type: ignore  # noqa: F401
        except ImportError:
            self.error = "Neither picamera2 nor opencv-python is available"
            self._log("camera", "error", self.error)
            return

        import cv2  # type: ignore

        cap = cv2.VideoCapture(self._cv2_index)
        if not cap.isOpened():
            self.error = f"cv2.VideoCapture({self._cv2_index}) could not open -- no camera found"
            self._log("camera", "error", self.error)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, 10)
        cap.release()

        self.running = True
        self.error = None
        self._backend = f"cv2-v4l2-/dev/video{self._cv2_index}"
        self._log("camera", "info", f"camera started (cv2 fallback, /dev/video{self._cv2_index})", size=self.size)

        self._cv2_stop.clear()
        self._cv2_thread = threading.Thread(
            target=self._cv2_capture_loop,
            args=(self._cv2_index,),
            name="voicepi-camera-cv2",
            daemon=True,
        )
        self._cv2_thread.start()

    def _cv2_capture_loop(self, device_index: int) -> None:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            with self.lock:
                self.error = f"cv2 capture failed to (re)open /dev/video{device_index}"
                self.running = False
            self._log("camera", "error", self.error)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, 10)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        consecutive_failures = 0

        try:
            while not self._cv2_stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        with self.lock:
                            self.error = "cv2 camera stopped delivering frames"
                            self.running = False
                        self._log("camera", "error", self.error)
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                _, buf = cv2.imencode(".jpg", frame, encode_params)
                self.output.write(buf.tobytes())
                # ~10 fps target
                time.sleep(0.08)
        finally:
            cap.release()

    # ------------------------------------------------------------------

    def _log(self, source: str, level: str, message: str, **fields: Any) -> None:
        if self.logger is not None:
            try:
                self.logger.event(source, level, message, **fields)
            except Exception:
                pass

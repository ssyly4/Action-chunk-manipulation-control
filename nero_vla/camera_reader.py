"""Timestamped latest-frame camera readers for NERO data collection."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    image_rgb: np.ndarray
    monotonic_ns: int
    unix_ns: int
    sequence: int


class V4L2CameraReader:
    def __init__(
        self,
        device: str,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        name: str = "camera",
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self._condition = threading.Condition()
        self._latest: CameraFrame | None = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._received = 0
        self._started_ns: int | None = None

    def start(self) -> "V4L2CameraReader":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._started_ns = time.monotonic_ns()
        self._thread = threading.Thread(target=self._run, name=f"nero-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2)
            if thread.is_alive():
                # Normal V4L2 reads return every frame. Release externally only
                # as a fallback for a driver-stalled read; concurrent release
                # during capture.read() can segfault inside OpenCV.
                capture = self._capture
                if capture is not None:
                    capture.release()
                thread.join(timeout=1)
        self._thread = None

    def __enter__(self) -> "V4L2CameraReader":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def wait_ready(self, timeout_sec: float = 5.0) -> CameraFrame:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while self._latest is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    suffix = f": {self._error}" if self._error else ""
                    raise TimeoutError(f"No frame from {self.name} ({self.device}){suffix}")
                self._condition.wait(remaining)
            return self._latest

    def latest(self, max_age_sec: float = 0.2, copy: bool = True) -> CameraFrame:
        with self._condition:
            frame = self._latest
        if frame is None:
            raise RuntimeError(f"No frame from {self.name}")
        age = (time.monotonic_ns() - frame.monotonic_ns) / 1e9
        if age > max_age_sec:
            raise RuntimeError(f"{self.name} frame is stale: {age:.3f}s")
        image = frame.image_rgb.copy() if copy else frame.image_rgb
        return CameraFrame(image, frame.monotonic_ns, frame.unix_ns, frame.sequence)

    @property
    def measured_hz(self) -> float:
        if self._started_ns is None:
            return 0.0
        elapsed = (time.monotonic_ns() - self._started_ns) / 1e9
        return self._received / elapsed if elapsed > 0 else 0.0

    def _run(self) -> None:
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        self._capture = capture
        try:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            # The uvcvideo driver needs multiple mmap buffers to sustain 30 FPS.
            # This thread drains them continuously, while callers only see _latest.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)
            if not capture.isOpened():
                raise RuntimeError("OpenCV could not open the V4L2 device")
            sequence = 0
            while not self._stop.is_set():
                ok, image_bgr = capture.read()
                received_mono_ns = time.monotonic_ns()
                received_unix_ns = time.time_ns()
                if not ok or image_bgr is None:
                    raise RuntimeError("V4L2 frame read failed")
                if image_bgr.shape != (self.height, self.width, 3):
                    raise RuntimeError(
                        f"Expected {(self.height, self.width, 3)}, got {image_bgr.shape}"
                    )
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb.setflags(write=False)
                frame = CameraFrame(image_rgb, received_mono_ns, received_unix_ns, sequence)
                with self._condition:
                    self._latest = frame
                    self._received += 1
                    self._condition.notify_all()
                sequence += 1
        except Exception as exc:
            with self._condition:
                self._error = f"{type(exc).__name__}: {exc}"
                self._condition.notify_all()
        finally:
            capture.release()
            self._capture = None

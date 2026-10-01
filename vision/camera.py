"""Camera capture abstraction.

A ``CameraSource`` yields BGR frames (``numpy`` uint8 ``H x W x 3``, OpenCV's
native layout) one at a time. The abstraction mirrors ``transport.base``: a small
ABC with an ``open`` / ``read`` / ``close`` lifecycle, a real hardware
implementation (``OpenCVCameraSource``), and a hardware-free ``MockCameraSource``
so tests and the ``--backend mock`` flow never need a physical webcam.

``cv2`` is imported lazily inside ``OpenCVCameraSource.open`` so this module
imports cleanly on machines without ``opencv-python`` — the same guard style used
for ``sounddevice`` in ``conversation.audio``.
"""

from __future__ import annotations

import abc
import logging
import math
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# A frame is a BGR uint8 numpy array of shape (height, width, 3).
Frame = np.ndarray


class CameraSource(abc.ABC):
    """Pluggable source of camera frames.

    ``read`` returns the most recent frame or ``None`` if one is not available
    (camera warming up, transient grab failure). Callers must treat ``None`` as
    "no new data this tick", not as a fatal error.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def open(self) -> None:
        """Acquire the device. Must be idempotent."""

    @abc.abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the latest BGR frame, or ``None`` if unavailable."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the device. Must be idempotent."""

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class OpenCVCameraSource(CameraSource):
    """Webcam capture via ``cv2.VideoCapture``.

    Grabs frames on demand (one ``read`` per caller tick) rather than running its
    own buffering thread — the face-tracking loop already ticks at a fixed rate,
    so pulling the freshest frame each tick keeps latency low without a second
    thread. A stale ``VideoCapture`` buffer isn't a concern at our ~12 Hz poll.
    """

    name = "opencv"

    def __init__(self, index: int = 0, *, width: int = 640, height: int = 480) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap = None

    def open(self) -> None:
        if self._cap is not None:
            return
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "opencv-python is required for webcam capture. "
                "Install it via `pip install opencv-python`."
            ) from exc
        cap = cv2.VideoCapture(self.index)
        # Request a modest resolution: face detection doesn't need 1080p and
        # smaller frames keep per-tick detection cost low.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"could not open camera index {self.index}. "
                "Check it's connected and not in use by another app, or set "
                "vision.camera_index in config.yaml."
            )
        self._cap = cap
        log.info("camera %d opened (%dx%d)", self.index, self.width, self.height)

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                log.exception("error releasing camera")
            self._cap = None


class MockCameraSource(CameraSource):
    """Hardware-free camera that renders a moving bright dot on a dark frame.

    Used by tests and by ``--backend mock`` runs with no webcam. The dot traces a
    slow Lissajous path so a real face detector (or, in tests, a stub) has a
    predictable moving target. ``read`` never returns ``None`` once opened.
    """

    name = "mock"

    def __init__(self, *, width: int = 640, height: int = 480, dot_radius: int = 40) -> None:
        self.width = width
        self.height = height
        self.dot_radius = dot_radius
        self._t = 0.0
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def read(self) -> Optional[Frame]:
        if not self._opened:
            return None
        self._t += 0.05
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cx = int((0.5 + 0.35 * math.sin(self._t)) * self.width)
        cy = int((0.5 + 0.25 * math.sin(self._t * 1.7)) * self.height)
        r = self.dot_radius
        y0, y1 = max(0, cy - r), min(self.height, cy + r)
        x0, x1 = max(0, cx - r), min(self.width, cx + r)
        frame[y0:y1, x0:x1] = (200, 200, 200)
        return frame

    def close(self) -> None:
        self._opened = False


def build_camera_source(kind: str, *, index: int = 0) -> CameraSource:
    """Factory mirroring ``build_llm_provider`` / ``build_x`` elsewhere."""
    if kind == "mock":
        return MockCameraSource()
    return OpenCVCameraSource(index=index)

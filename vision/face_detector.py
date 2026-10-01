"""Face detection.

Turns a BGR frame into a list of :class:`FaceBox` records with normalized
coordinates in ``[0, 1]``, so everything downstream (primary-face selection,
gaze mapping) is resolution-independent.

Two backends, chosen by ``build_face_detector``:

* **MediaPipe Face Detection** (``mediapipe``) — accurate, and the same dependency
  opens the door to Hands/Pose gestures later. It relies on the legacy
  ``mp.solutions`` API, which is **not available on newer MediaPipe wheels**
  (e.g. the Tasks-only 0.10.x builds shipped for Python 3.13), so it can't be the
  hard default.
* **OpenCV Haar cascade** (``opencv``) — bundled with ``opencv-python`` (no model
  file, no download), works on every Python version. Frontal-face only and a bit
  less robust, but plenty for head tracking.

The default ``auto`` picks MediaPipe when its ``solutions`` API is importable and
otherwise falls back to OpenCV, so face tracking works out of the box regardless
of the MediaPipe build. Heavy imports are lazy; tests inject a stub detector.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceBox:
    """A detected face in normalized frame coordinates.

    ``cx`` / ``cy`` are the box center in ``[0, 1]`` (origin top-left, x rightward,
    y downward — the image convention). ``area`` is the normalized box area, used
    as a cheap "nearness" proxy: a bigger box means a closer face. ``score`` is the
    detector's confidence in ``[0, 1]``.
    """

    cx: float
    cy: float
    width: float
    height: float
    score: float

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


class FaceDetector(abc.ABC):
    """Pluggable face detector: frame in, normalized boxes out."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """Return detected faces (possibly empty) for a BGR frame."""

    def close(self) -> None:
        """Release any held resources. Default is a no-op."""


class MediaPipeFaceDetector(FaceDetector):
    """MediaPipe Face Detection wrapper.

    ``model_selection=1`` is the full-range model (good to a few metres, right for
    a booth); ``0`` is the short-range face-close-to-camera model. We convert
    MediaPipe's relative bounding box straight into a :class:`FaceBox`.
    """

    def __init__(self, *, min_confidence: float = 0.5, model_selection: int = 1) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "mediapipe is required for the mediapipe face detector. "
                "Install it via `pip install mediapipe`."
            ) from exc
        if not hasattr(mp, "solutions"):
            # Newer (Tasks-only) MediaPipe wheels, e.g. on Python 3.13, drop the
            # legacy solutions API entirely. Fail clearly so `auto` can fall back.
            raise RuntimeError(
                "this mediapipe build has no `solutions` API "
                f"(version {getattr(mp, '__version__', '?')}); "
                "use the opencv detector instead."
            )
        self._mp = mp
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        # MediaPipe expects RGB; OpenCV frames are BGR.
        rgb = frame[:, :, ::-1]
        results = self._detector.process(rgb)
        boxes: List[FaceBox] = []
        detections = getattr(results, "detections", None) or []
        for det in detections:
            rel = det.location_data.relative_bounding_box
            w = float(rel.width)
            h = float(rel.height)
            cx = float(rel.xmin) + w / 2.0
            cy = float(rel.ymin) + h / 2.0
            score = float(det.score[0]) if det.score else 0.0
            boxes.append(
                FaceBox(
                    cx=_clamp01(cx),
                    cy=_clamp01(cy),
                    width=w,
                    height=h,
                    score=score,
                )
            )
        return boxes

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:  # noqa: BLE001
            log.exception("error closing mediapipe detector")


class OpenCVFaceDetector(FaceDetector):
    """Frontal-face detection via OpenCV's bundled Haar cascade.

    Uses the ``haarcascade_frontalface_default.xml`` that ships inside
    ``opencv-python`` (``cv2.data.haarcascades``) — no model download, works on any
    Python. Haar gives no confidence score, so :attr:`FaceBox.score` is set to 1.0.
    """

    def __init__(self, *, min_size: int = 60, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "opencv-python is required for the opencv face detector."
            ) from exc
        self._cv2 = cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade from {cascade_path}")
        self._min_size = min_size
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        cv2 = self._cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(self._min_size, self._min_size),
        )
        boxes: List[FaceBox] = []
        for (x, y, bw, bh) in rects:
            boxes.append(
                FaceBox(
                    cx=_clamp01((x + bw / 2.0) / w),
                    cy=_clamp01((y + bh / 2.0) / h),
                    width=bw / w,
                    height=bh / h,
                    score=1.0,
                )
            )
        return boxes


def build_face_detector(kind: str = "auto", **kwargs) -> FaceDetector:
    """Factory mirroring the ``build_x`` pattern used across the codebase.

    ``auto`` (default) prefers MediaPipe when its ``solutions`` API is available
    and otherwise falls back to the always-available OpenCV Haar detector.
    """
    kind = (kind or "auto").lower()
    if kind == "mediapipe":
        return MediaPipeFaceDetector(**kwargs)
    if kind == "opencv":
        return OpenCVFaceDetector(**kwargs)
    if kind == "auto":
        try:
            detector = MediaPipeFaceDetector()
            log.info("face detector: using mediapipe")
            return detector
        except Exception as exc:  # noqa: BLE001
            log.info("face detector: falling back to opencv (%s)", exc)
            return OpenCVFaceDetector()
    raise ValueError(f"unknown face detector: {kind!r}")


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)

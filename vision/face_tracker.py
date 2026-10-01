"""Face-tracking loop: webcam frames in, head-aim target out.

``FaceTracker`` is the vision analog of :class:`motion.scheduler.MotionScheduler`
— a single ``asyncio.Task`` with a ``start`` / ``stop`` lifecycle that ticks at a
fixed rate. Each tick it grabs a frame, detects faces, picks the primary one, maps
it to a normalized head target, and writes that into a shared
:class:`motion.models.GazeContext`. The behavior engine reads that context once per
motion tick and blends it into Maxwell's head motion.

The blocking work (camera grab + MediaPipe inference) runs in a thread executor so
it never stalls the audio / motion event loop.

The pure functions ``select_primary`` and ``map_face_to_gaze`` hold all the logic
worth unit-testing; the task itself is just plumbing around them.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from motion.models import GazeContext

from .camera import CameraSource
from .face_detector import FaceBox, FaceDetector

log = logging.getLogger(__name__)


def select_primary(
    faces: List[FaceBox],
    prev_center: Optional[Tuple[float, float]],
    hysteresis: float,
) -> Optional[FaceBox]:
    """Pick the face Maxwell should look at, with switch hysteresis.

    Prefers the largest (nearest) face, but once locked onto someone it stays with
    the face closest to where the previous primary was unless a challenger is
    ``hysteresis`` fraction *bigger* — so two people at similar distance don't make
    the head flick back and forth every frame.
    """
    if not faces:
        return None
    largest = max(faces, key=lambda f: f.area)
    if prev_center is None:
        return largest
    nearest = min(
        faces,
        key=lambda f: (f.cx - prev_center[0]) ** 2 + (f.cy - prev_center[1]) ** 2,
    )
    if nearest is largest:
        return largest
    # A different, bigger face only wins if it clearly beats the one we were
    # already tracking.
    if largest.area > nearest.area * (1.0 + max(0.0, hysteresis)):
        return largest
    return nearest


def map_face_to_gaze(
    face: FaceBox,
    *,
    gain_lr: float,
    gain_ud: float,
    invert_lr: bool,
    invert_ud: bool,
    deadzone: float,
) -> Tuple[float, float]:
    """Map a face's normalized center to a normalized head target in ``[0, 1]``.

    A face centered in frame yields ``(0.5, 0.5)`` (look straight ahead). Offsets
    are scaled by ``gain`` (how far the head swings for a given face offset) and
    can be flipped per axis with ``invert`` — needed because whether a mirrored
    webcam / a given servo mounting turns "the right way" isn't knowable ahead of
    time. ``deadzone`` zeroes out tiny offsets so a roughly-centered face doesn't
    cause the head to hunt.
    """
    ex = face.cx - 0.5
    ey = face.cy - 0.5
    if abs(ex) < deadzone:
        ex = 0.0
    if abs(ey) < deadzone:
        ey = 0.0
    sx = -1.0 if invert_lr else 1.0
    sy = -1.0 if invert_ud else 1.0
    lr = 0.5 + sx * gain_lr * ex
    ud = 0.5 + sy * gain_ud * ey
    return _clamp01(lr), _clamp01(ud)


@dataclass
class FaceTracker:
    """Drives face detection at a fixed rate and updates a shared GazeContext."""

    camera: CameraSource
    detector: FaceDetector
    gaze: GazeContext
    fps: float = 12.0
    gain_lr: float = 1.4
    gain_ud: float = 1.2
    invert_lr: bool = False
    invert_ud: bool = False
    deadzone: float = 0.05
    lost_face_timeout_s: float = 1.5
    # Per-tick smoothing on the target + confidence so the head glides rather than
    # snapping between detections.
    target_smoothing: float = 0.35
    confidence_attack: float = 0.4

    _task: Optional[asyncio.Task] = field(default=None, init=False)
    _stopped: bool = field(default=False, init=False)
    _prev_center: Optional[Tuple[float, float]] = field(default=None, init=False)
    _last_frame: Optional[np.ndarray] = field(default=None, init=False)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopped = False
        # Open the camera off the event loop — construction can block briefly.
        await asyncio.get_event_loop().run_in_executor(None, self.camera.open)
        self._task = asyncio.create_task(self._run(), name="face-tracker")
        log.info("face tracker started (%.0f Hz)", self.fps)

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._safe_close)
        # Drop confidence so the head returns to procedural motion immediately.
        self.gaze.confidence = 0.0
        log.info("face tracker stopped")

    def _safe_close(self) -> None:
        try:
            self.camera.close()
        except Exception:  # noqa: BLE001
            log.exception("error closing camera")
        try:
            self.detector.close()
        except Exception:  # noqa: BLE001
            log.exception("error closing detector")

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def latest_frame(self) -> Optional[np.ndarray]:
        """Most recent frame the tracker grabbed, for on-demand scene capture.

        Reusing the tracker's frame means scene understanding never opens a second
        reader on the same ``VideoCapture`` (which would race for frames).
        """
        return self._last_frame

    def _grab_and_detect(self) -> Optional[FaceBox]:
        """Blocking: grab a frame, detect, choose the primary face. Runs in executor."""
        frame = self.camera.read()
        if frame is None:
            return None
        self._last_frame = frame
        faces = self.detector.detect(frame)
        primary = select_primary(faces, self._prev_center, hysteresis=0.25)
        if primary is not None:
            self._prev_center = (primary.cx, primary.cy)
        return primary

    async def _run(self) -> None:
        period = 1.0 / max(1.0, self.fps)
        loop = asyncio.get_event_loop()
        next_tick = time.monotonic()
        try:
            while not self._stopped:
                now = time.monotonic()
                try:
                    primary = await loop.run_in_executor(None, self._grab_and_detect)
                except Exception:  # noqa: BLE001 - a bad frame must not kill the loop
                    log.exception("face detect tick failed")
                    primary = None

                if primary is not None:
                    lr, ud = map_face_to_gaze(
                        primary,
                        gain_lr=self.gain_lr,
                        gain_ud=self.gain_ud,
                        invert_lr=self.invert_lr,
                        invert_ud=self.invert_ud,
                        deadzone=self.deadzone,
                    )
                    s = _clamp01(self.target_smoothing)
                    self.gaze.target_lr += (lr - self.gaze.target_lr) * s
                    self.gaze.target_ud += (ud - self.gaze.target_ud) * s
                    a = _clamp01(self.confidence_attack)
                    self.gaze.confidence += (1.0 - self.gaze.confidence) * a
                    self.gaze.last_seen = now
                else:
                    # Decay confidence toward zero over the lost-face timeout; the
                    # behavior engine eases the head back to center as it drops.
                    if self.lost_face_timeout_s > 0:
                        decay = math.exp(-period / self.lost_face_timeout_s)
                        self.gaze.confidence *= decay
                    else:
                        self.gaze.confidence = 0.0
                    if self.gaze.confidence < 1e-3:
                        self.gaze.confidence = 0.0
                        self._prev_center = None

                next_tick += period
                sleep_for = next_tick - time.monotonic()
                if sleep_for < 0:
                    next_tick = time.monotonic()
                    sleep_for = 0
                await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            return


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)

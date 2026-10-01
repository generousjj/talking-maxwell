"""Face-tracking logic: gaze mapping, primary selection, and the async loop."""

from __future__ import annotations

import asyncio
from typing import List

from motion.models import GazeContext
from vision.camera import MockCameraSource
from vision.face_detector import FaceBox, FaceDetector
from vision.face_tracker import FaceTracker, map_face_to_gaze, select_primary


def test_map_centered_face_looks_straight_ahead():
    face = FaceBox(cx=0.5, cy=0.5, width=0.2, height=0.2, score=1.0)
    lr, ud = map_face_to_gaze(
        face, gain_lr=1.4, gain_ud=1.2, invert_lr=False, invert_ud=False, deadzone=0.05
    )
    assert lr == 0.5
    assert ud == 0.5


def test_map_offset_face_swings_head_and_invert_flips_it():
    face = FaceBox(cx=0.8, cy=0.2, width=0.2, height=0.2, score=1.0)
    lr, ud = map_face_to_gaze(
        face, gain_lr=1.4, gain_ud=1.2, invert_lr=False, invert_ud=False, deadzone=0.05
    )
    # Face right + high => head_lr right of center, head_ud below center.
    assert lr > 0.5
    assert ud < 0.5

    lr_i, ud_i = map_face_to_gaze(
        face, gain_lr=1.4, gain_ud=1.2, invert_lr=True, invert_ud=True, deadzone=0.05
    )
    # Inverting mirrors both axes across center.
    assert abs((lr_i - 0.5) + (lr - 0.5)) < 1e-9
    assert abs((ud_i - 0.5) + (ud - 0.5)) < 1e-9


def test_deadzone_treats_near_center_as_centered():
    face = FaceBox(cx=0.52, cy=0.48, width=0.2, height=0.2, score=1.0)
    lr, ud = map_face_to_gaze(
        face, gain_lr=1.4, gain_ud=1.2, invert_lr=False, invert_ud=False, deadzone=0.05
    )
    assert lr == 0.5
    assert ud == 0.5


def test_map_output_is_clamped():
    face = FaceBox(cx=1.0, cy=0.0, width=0.2, height=0.2, score=1.0)
    lr, ud = map_face_to_gaze(
        face, gain_lr=5.0, gain_ud=5.0, invert_lr=False, invert_ud=False, deadzone=0.0
    )
    assert 0.0 <= lr <= 1.0
    assert 0.0 <= ud <= 1.0


def test_select_primary_picks_largest_when_no_history():
    faces = [
        FaceBox(cx=0.2, cy=0.5, width=0.1, height=0.1, score=1.0),
        FaceBox(cx=0.8, cy=0.5, width=0.3, height=0.3, score=1.0),
    ]
    primary = select_primary(faces, prev_center=None, hysteresis=0.25)
    assert primary is faces[1]


def test_select_primary_has_hysteresis():
    small = FaceBox(cx=0.3, cy=0.5, width=0.2, height=0.2, score=1.0)
    slightly_bigger = FaceBox(cx=0.7, cy=0.5, width=0.21, height=0.21, score=1.0)
    faces = [small, slightly_bigger]
    # We were tracking the face near (0.3, 0.5); a marginally bigger challenger
    # should NOT steal focus (avoids twitching between two similar faces).
    primary = select_primary(faces, prev_center=(0.3, 0.5), hysteresis=0.25)
    assert primary is small

    # A clearly bigger challenger does win.
    much_bigger = FaceBox(cx=0.7, cy=0.5, width=0.4, height=0.4, score=1.0)
    primary2 = select_primary([small, much_bigger], prev_center=(0.3, 0.5), hysteresis=0.25)
    assert primary2 is much_bigger


def test_select_primary_empty_returns_none():
    assert select_primary([], prev_center=None, hysteresis=0.25) is None


class _StubDetector(FaceDetector):
    """Returns a fixed face regardless of frame; None-position means no face."""

    def __init__(self, cx: float | None, cy: float = 0.5) -> None:
        self.cx = cx
        self.cy = cy

    def detect(self, frame) -> List[FaceBox]:
        if self.cx is None:
            return []
        return [FaceBox(cx=self.cx, cy=self.cy, width=0.3, height=0.3, score=0.99)]


def test_tracker_moves_gaze_toward_face_then_decays_when_lost():
    async def run() -> GazeContext:
        gaze = GazeContext()
        detector = _StubDetector(cx=0.85, cy=0.5)
        tracker = FaceTracker(
            camera=MockCameraSource(),
            detector=detector,
            gaze=gaze,
            fps=60.0,
            gain_lr=1.4,
            gain_ud=1.2,
            lost_face_timeout_s=0.2,
        )
        await tracker.start()
        # Let the loop run enough ticks to lock on.
        await asyncio.sleep(0.4)
        assert gaze.confidence > 0.5, "should be confident with a steady face"
        assert gaze.target_lr > 0.6, "head should swing right toward a right-side face"
        locked_conf = gaze.confidence

        # Face disappears -> confidence should decay.
        detector.cx = None
        await asyncio.sleep(0.4)
        assert gaze.confidence < locked_conf
        await tracker.stop()
        return gaze

    gaze = asyncio.run(run())
    # After stop, confidence is forced to zero so the head returns to procedural motion.
    assert gaze.confidence == 0.0

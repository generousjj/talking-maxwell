"""Motion mapping sanity tests."""

from __future__ import annotations

import numpy as np

from motion.behavior_engine import BehaviorEngine
from motion.envelope import EnvelopeFollower
from motion.models import (
    BehaviorGains,
    ConversationState,
    JawCalibration,
    MotionFrame,
    SpeakingContext,
)


def test_motion_frame_clamp_keeps_range():
    frame = MotionFrame(jaw_open=1.5, head_lr=-0.2, head_ud=0.4, wing=2.0, timestamp=1.0)
    clamped = frame.clamped()
    assert 0.0 <= clamped.jaw_open <= 1.0
    assert 0.0 <= clamped.head_lr <= 1.0
    assert 0.0 <= clamped.head_ud <= 1.0
    assert 0.0 <= clamped.wing <= 1.0


def test_behavior_outputs_stay_in_unit_interval_under_stress():
    engine = BehaviorEngine(gains=BehaviorGains(seed=7))
    for i in range(600):
        ctx = SpeakingContext(
            envelope=(-0.4 if i % 2 == 0 else 1.6),
            text="Excited question?",
            phrase_boundary=(i % 20 == 0),
            emphasis=(2.0 if i % 17 == 0 else 0.0),
            question_like=True,
            excited=(i % 23 == 0),
        )
        state = [
            ConversationState.IDLE,
            ConversationState.LISTENING,
            ConversationState.THINKING,
            ConversationState.SPEAKING,
        ][i % 4]
        frame = engine.tick(state, now=i / 30.0, speaking=ctx).to_frame(timestamp=i / 30.0)
        for v in (frame.jaw_open, frame.head_lr, frame.head_ud, frame.wing):
            assert 0.0 <= v <= 1.0


def test_envelope_mapping_stays_within_calibration_bounds():
    cal = JawCalibration(floor=0.2, ceiling=0.7)
    follower = EnvelopeFollower(sample_rate=16000, calibration=cal, frame_ms=20.0)
    for _ in range(50):
        value = follower.process(np.random.uniform(-0.5, 0.5, 320).astype(np.float32))
        assert cal.floor <= value <= cal.ceiling

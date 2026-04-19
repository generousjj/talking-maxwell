"""EnvelopeFollower behavior tests."""

from __future__ import annotations

import numpy as np

from motion.envelope import EnvelopeFollower
from motion.models import JawCalibration


def _make(**overrides) -> EnvelopeFollower:
    cal = JawCalibration(**overrides)
    return EnvelopeFollower(sample_rate=16000, calibration=cal, frame_ms=20.0)


def test_silence_maps_to_floor():
    follower = _make()
    value = follower.process(np.zeros(320, dtype=np.float32))
    assert abs(value - follower.calibration.floor) < 1e-6


def test_noise_floor_zeros_small_inputs():
    follower = _make(noise_floor=0.05, gain=1.0)
    samples = np.full(320, 0.01, dtype=np.float32)
    value = follower.process(samples)
    assert abs(value - follower.calibration.floor) < 1e-6


def test_output_is_bounded_to_floor_and_ceiling():
    follower = _make(floor=0.1, ceiling=0.9, gain=100.0)
    loud = np.ones(320, dtype=np.float32)
    for _ in range(20):
        value = follower.process(loud)
    assert 0.1 <= value <= 0.9


def test_attack_is_faster_than_release():
    fast = _make(attack=0.9, release=0.05, gain=1.0, floor=0.0, ceiling=1.0, peak_hold_ms=0)
    loud = np.full(320, 0.5, dtype=np.float32)
    silence = np.zeros(320, dtype=np.float32)

    rising = fast.process(loud)
    assert rising > 0.3

    # Now silence should decay slowly (release), so still above noise.
    decayed = fast.process(silence)
    assert decayed > 0.1


def test_process_rms_accepts_precomputed_values():
    follower = _make(floor=0.0, ceiling=1.0, gain=1.0, peak_hold_ms=0, noise_floor=0)
    for _ in range(5):
        out = follower.process_rms(0.4)
    assert 0.0 <= out <= 1.0
    out = follower.process_rms(0.0)
    assert out >= 0.0

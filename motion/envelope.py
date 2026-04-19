"""Audio envelope follower used to drive the jaw in real time.

The follower consumes mono PCM audio frames (floats or int16) and emits a
smoothed normalized loudness signal in [0, 1]. It applies separate attack and
release smoothing, a configurable noise floor (deadband), a short peak-hold,
and finally maps into a [floor, ceiling] range so the jaw stays expressive
without slamming shut.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .models import JawCalibration


@dataclass
class EnvelopeFollower:
    """Streaming RMS envelope follower with attack/release smoothing.

    Usage::

        follower = EnvelopeFollower(sample_rate=24000, calibration=JawCalibration())
        for frame in pcm_frames:   # frame: np.ndarray, mono float32 in [-1, 1]
            jaw = follower.process(frame)
    """

    sample_rate: int
    calibration: JawCalibration
    frame_ms: float = 20.0

    _smoothed: float = 0.0
    _peak: float = 0.0
    _peak_hold_until: float = 0.0

    @property
    def frame_samples(self) -> int:
        return max(1, int(self.sample_rate * self.frame_ms / 1000.0))

    def reset(self) -> None:
        self._smoothed = 0.0
        self._peak = 0.0
        self._peak_hold_until = 0.0

    def process(self, samples: Sequence[float] | np.ndarray) -> float:
        """Feed a chunk of mono samples and return the current jaw envelope."""
        arr = _to_float_mono(samples)
        if arr.size == 0:
            return self.process_rms(0.0)

        rms = float(np.sqrt(np.mean(np.square(arr))))
        return self.process_rms(rms)

    def process_rms(self, rms: float) -> float:
        """Feed a pre-computed RMS value (already in [0, 1]) for this frame.

        Useful when the audio plumbing already computed RMS to avoid redoing it.
        """
        cal = self.calibration
        rms = max(0.0, float(rms))
        if rms < cal.noise_floor:
            rms = 0.0

        target = min(1.0, rms * cal.gain)

        if target > self._smoothed:
            coeff = cal.attack
        else:
            coeff = cal.release
        coeff = max(0.0, min(1.0, coeff))
        self._smoothed = self._smoothed + coeff * (target - self._smoothed)

        now = time.monotonic()
        if self._smoothed >= self._peak or now >= self._peak_hold_until:
            self._peak = self._smoothed
            if cal.peak_hold_ms > 0 and self._smoothed >= self._peak:
                self._peak_hold_until = now + cal.peak_hold_ms / 1000.0

        value = max(self._smoothed, self._peak)
        return self._map_to_jaw(value)

    def process_chunks(self, samples: Sequence[float] | np.ndarray) -> list[float]:
        """Process a longer buffer in `frame_samples`-sized windows."""
        arr = _to_float_mono(samples)
        step = self.frame_samples
        out: list[float] = []
        for start in range(0, arr.size, step):
            out.append(self.process(arr[start : start + step]))
        return out

    def _map_to_jaw(self, smoothed: float) -> float:
        cal = self.calibration
        if smoothed <= 0.0:
            return cal.floor
        jaw = cal.floor + smoothed * (cal.ceiling - cal.floor)
        if jaw < cal.floor:
            return cal.floor
        if jaw > cal.ceiling:
            return cal.ceiling
        return jaw


def _to_float_mono(samples: Sequence[float] | np.ndarray) -> np.ndarray:
    if isinstance(samples, np.ndarray):
        arr = samples
    else:
        arr = np.asarray(list(samples))

    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        arr = arr.astype(np.float32) / float(2**31)
    else:
        arr = arr.astype(np.float32, copy=False)

    if arr.ndim == 2:
        # Downmix to mono by averaging channels.
        arr = arr.mean(axis=1)
    return arr


def compute_offline_envelope(
    pcm: np.ndarray,
    sample_rate: int,
    calibration: JawCalibration,
    frame_ms: float = 20.0,
) -> tuple[np.ndarray, float]:
    """Compute envelope values for an entire PCM buffer.

    Returns an array of jaw values plus the hop in seconds between samples.
    """
    follower = EnvelopeFollower(
        sample_rate=sample_rate, calibration=calibration, frame_ms=frame_ms
    )
    values = follower.process_chunks(pcm)
    hop_s = frame_ms / 1000.0
    return np.asarray(values, dtype=np.float32), hop_s

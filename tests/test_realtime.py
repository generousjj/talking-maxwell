"""Tests for the Realtime event helpers.

We don't try to exercise the live SDK here — that would need a real
WebSocket and an OpenAI key. The thing worth testing is the dispatch
classifier and base64 audio decoder, since both run on every event.
"""

from __future__ import annotations

import base64

import numpy as np

from conversation.realtime import (
    EVENT_AUDIO_DELTA,
    EVENT_RESPONSE_DONE,
    EVENT_SPEECH_STARTED,
    classify_event,
    decode_audio_delta,
)


def test_classify_dict_event() -> None:
    assert classify_event({"type": EVENT_AUDIO_DELTA}) == EVENT_AUDIO_DELTA
    assert classify_event({"type": EVENT_RESPONSE_DONE}) == EVENT_RESPONSE_DONE
    assert classify_event({}) is None


def test_classify_object_event() -> None:
    class FakeEvent:
        type = EVENT_SPEECH_STARTED

    assert classify_event(FakeEvent()) == EVENT_SPEECH_STARTED


def test_decode_audio_delta_round_trip() -> None:
    """A delta event with PCM16 base64 should decode to float32 in [-1, 1]."""
    pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    b64 = base64.b64encode(pcm.tobytes()).decode("ascii")
    samples = decode_audio_delta({"type": EVENT_AUDIO_DELTA, "delta": b64})
    assert samples is not None
    assert samples.dtype == np.float32
    assert samples.shape == (5,)
    # Within float32 quantization tolerance for the int16 → float32 conversion.
    assert abs(samples[0]) < 1e-6
    assert 0.49 < samples[1] < 0.51
    assert -0.51 < samples[2] < -0.49
    assert samples.max() <= 1.0 and samples.min() >= -1.0001


def test_decode_audio_delta_empty() -> None:
    assert decode_audio_delta({"type": EVENT_AUDIO_DELTA}) is None
    assert decode_audio_delta({"type": EVENT_AUDIO_DELTA, "delta": ""}) is None

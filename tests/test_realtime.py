"""Tests for the Realtime event helpers.

We don't try to exercise the live SDK here — that would need a real
WebSocket and an OpenAI key. The thing worth testing is the dispatch
classifier and base64 audio decoder, since both run on every event.
"""

from __future__ import annotations

import asyncio
import base64

import numpy as np

from conversation.realtime import (
    EVENT_AUDIO_DELTA,
    EVENT_FUNCTION_CALL_ARGS_DONE,
    EVENT_RESPONSE_DONE,
    EVENT_SPEECH_STARTED,
    RealtimeSession,
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


# ---- generic function-tool dispatch ----


def test_classify_function_call_event() -> None:
    assert (
        classify_event({"type": EVENT_FUNCTION_CALL_ARGS_DONE})
        == EVENT_FUNCTION_CALL_ARGS_DONE
    )


class _FakeItems:
    def __init__(self) -> None:
        self.created: list = []

    async def create(self, item=None) -> None:
        self.created.append(item)


class _FakeSession:
    def __init__(self) -> None:
        self.updates: list = []

    async def update(self, session=None) -> None:
        self.updates.append(session)


class _FakeConn:
    def __init__(self) -> None:
        self.conversation = type("C", (), {"item": _FakeItems()})()
        self.response = type("R", (), {"created": 0, "last": None})()
        self.session = _FakeSession()

        async def _create(response=None) -> None:
            self.response.created += 1
            self.response.last = response

        self.response.create = _create


def test_tool_call_dispatches_with_parsed_args() -> None:
    seen: list = []
    states: list[str] = []

    async def handler(name: str, args: dict) -> str:
        seen.append((name, args))
        return f"Okay, I'll remember {args.get('name')}!"

    async def state_cb(name: str) -> None:
        states.append(name)

    sess = RealtimeSession(api_key="x", tool_handler=handler, state_callback=state_cb)
    conn = _FakeConn()
    sess._conn = conn

    event = {
        "type": EVENT_FUNCTION_CALL_ARGS_DONE,
        "call_id": "call_1",
        "name": "remember_person",
        "arguments": '{"name": "Sarah"}',
    }
    asyncio.run(sess._handle_tool_call(event))

    assert seen == [("remember_person", {"name": "Sarah"})]
    assert sess._handling_tool_call is True
    out = conn.conversation.item.created[0]
    assert out["type"] == "function_call_output"
    assert out["call_id"] == "call_1"
    assert out["output"] == "Okay, I'll remember Sarah!"
    assert conn.response.created == 1
    assert "thinking" in states


def test_tool_call_without_handler_is_noop() -> None:
    sess = RealtimeSession(api_key="x", tool_handler=None)
    conn = _FakeConn()
    sess._conn = conn
    event = {"type": EVENT_FUNCTION_CALL_ARGS_DONE, "call_id": "c1", "name": "look_and_describe"}
    asyncio.run(sess._handle_tool_call(event))
    assert not conn.conversation.item.created
    assert conn.response.created == 0
    assert sess._handling_tool_call is False


def test_tool_call_survives_handler_failure() -> None:
    async def handler(name: str, args: dict) -> str:
        raise RuntimeError("tool exploded")

    sess = RealtimeSession(api_key="x", tool_handler=handler)
    conn = _FakeConn()
    sess._conn = conn
    event = {"type": EVENT_FUNCTION_CALL_ARGS_DONE, "call_id": "c9", "name": "look_and_describe"}
    asyncio.run(sess._handle_tool_call(event))
    # Still reports something back so the model isn't left hanging.
    assert conn.conversation.item.created
    assert conn.conversation.item.created[0]["call_id"] == "c9"
    assert conn.response.created == 1


def test_trigger_greeting_when_idle_creates_response_with_instruction() -> None:
    sess = RealtimeSession(api_key="x")
    sess._running = True
    conn = _FakeConn()
    sess._conn = conn

    sent = asyncio.run(sess.trigger_greeting("Say hi to Sarah."))
    assert sent is True
    assert conn.response.created == 1
    # Per-response instruction carried the greeting.
    assert conn.response.last == {"instructions": "Say hi to Sarah."}


def test_trigger_greeting_skipped_when_turn_in_flight() -> None:
    conn = _FakeConn()
    for flag in ("_assistant_speaking", "_user_speaking", "_handling_tool_call", "_mic_muted"):
        sess = RealtimeSession(api_key="x")
        sess._running = True
        sess._conn = conn
        setattr(sess, flag, True)
        sent = asyncio.run(sess.trigger_greeting("hello"))
        assert sent is False, f"should not greet while {flag} is set"
    assert conn.response.created == 0  # never issued a response


def test_set_context_line_updates_instructions_and_debounces() -> None:
    sess = RealtimeSession(api_key="x", instructions="You are Maxwell.")
    sess._base_instructions = "You are Maxwell."
    conn = _FakeConn()
    sess._conn = conn

    async def go():
        await sess.set_context_line("You are looking at Sarah.")
        await sess.set_context_line("You are looking at Sarah.")  # unchanged -> no-op
        await sess.set_context_line("")  # cleared -> update back to base

    asyncio.run(go())

    # Two updates: set, then clear (the duplicate in the middle is debounced).
    assert len(conn.session.updates) == 2
    assert "Sarah" in conn.session.updates[0]["instructions"]
    assert conn.session.updates[1]["instructions"] == "You are Maxwell."

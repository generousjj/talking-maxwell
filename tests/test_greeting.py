"""Proactive greeting: instruction selection + cooldown + gating."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.pipeline import ConversationPipeline
from conversation.llm import StubLLM
from motion.state_machine import ConversationStateMachine
from vision.recognition import RecognitionContext


def _pipeline(greet_on_sight=True, cooldown=600.0):
    cfg = SimpleNamespace(
        recognition_enabled=True,
        greet_on_sight=greet_on_sight,
        greeting_cooldown_s=cooldown,
    )
    return ConversationPipeline(
        stt=None, llm=StubLLM(), tts=None, backend=None,
        state_machine=ConversationStateMachine(),
        jaw_calibration=None, behavior_gains=None, vision_config=cfg,
    )


class _FakeRecognition:
    """Stands in for RecognitionTracker: canned last-seen timestamp per context."""

    def __init__(self, last_seen=None):
        self.is_running = True
        self._last_seen = last_seen  # monotonic ts or None

    def last_seen_at(self, ctx):
        return self._last_seen


class _FakeSession:
    def __init__(self, idle=True):
        self.is_running = True
        self._idle = idle
        self.greetings: list[str] = []
        self.context_lines: list[str] = []

    async def set_context_line(self, text):
        self.context_lines.append(text)

    async def trigger_greeting(self, instruction):
        if not self._idle:
            return False
        self.greetings.append(instruction)
        return True


def test_greeting_for_known_and_unknown_and_none():
    key, instr = ConversationPipeline._greeting_for(RecognitionContext(name="Sarah"))
    assert key == "Sarah" and "Sarah" in instr
    key, instr = ConversationPipeline._greeting_for(RecognitionContext(is_unknown=True))
    assert key == "__unknown__" and "name" in instr.lower()
    key, instr = ConversationPipeline._greeting_for(RecognitionContext())
    assert key is None


def test_maybe_greet_greets_first_sighting():
    # Never seen before (last_seen is None) -> greet.
    p = _pipeline()
    sess = _FakeSession()
    p._rt_session = sess
    p._recognition = _FakeRecognition(last_seen=None)
    asyncio.run(p._maybe_greet(RecognitionContext(name="Sarah")))
    assert len(sess.greetings) == 1
    assert "Sarah" in sess.greetings[0]


def test_maybe_greet_suppressed_when_seen_recently():
    # Seen 5 seconds ago (< 600s cooldown) -> same encounter, no re-greet even
    # though a change event fired (detection flickered).
    import time
    p = _pipeline(cooldown=600.0)
    sess = _FakeSession()
    p._rt_session = sess
    p._recognition = _FakeRecognition(last_seen=time.monotonic() - 5.0)
    asyncio.run(p._maybe_greet(RecognitionContext(name="Sarah")))
    assert sess.greetings == []


def test_maybe_greet_greets_after_long_absence():
    # Last seen 11 minutes ago (> 600s) -> genuine return, greet again.
    import time
    p = _pipeline(cooldown=600.0)
    sess = _FakeSession()
    p._rt_session = sess
    p._recognition = _FakeRecognition(last_seen=time.monotonic() - 660.0)
    asyncio.run(p._maybe_greet(RecognitionContext(name="Sarah")))
    assert len(sess.greetings) == 1


def test_maybe_greet_disabled_by_config():
    p = _pipeline(greet_on_sight=False)
    sess = _FakeSession()
    p._rt_session = sess
    p._recognition = _FakeRecognition(last_seen=None)
    asyncio.run(p._maybe_greet(RecognitionContext(name="Sarah")))
    assert sess.greetings == []


def test_maybe_greet_not_sent_when_session_busy():
    p = _pipeline()
    busy = _FakeSession(idle=False)
    p._rt_session = busy
    p._recognition = _FakeRecognition(last_seen=None)
    asyncio.run(p._maybe_greet(RecognitionContext(name="Sarah")))
    assert busy.greetings == []

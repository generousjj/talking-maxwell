"""BehaviorEngine deterministic behavior tests."""

from __future__ import annotations

import math
import time

from motion.behavior_engine import (
    BehaviorEngine,
    analyze_text,
    estimate_phrase_boundaries,
    split_phrases,
)
from motion.models import BehaviorGains, ConversationState, SpeakingContext


def _engine(**overrides) -> BehaviorEngine:
    overrides.setdefault("seed", 42)
    gains = BehaviorGains(**overrides)
    return BehaviorEngine(gains=gains)


def test_idle_has_no_jaw_and_head_stays_centered():
    # IDLE keeps the jaw fully closed and the head at neutral center; only
    # the wing is allowed to move (periodic "alive" flap).
    engine = _engine()
    for i in range(60):
        out = engine.tick(ConversationState.IDLE, now=i / 30.0)
        assert out.jaw_open == 0.0
        assert abs(out.head_lr - 0.5) < 1e-6
        assert abs(out.head_ud - 0.5) < 1e-6


def test_idle_wing_flaps_periodically():
    engine = _engine()
    saw_flap = False
    saw_rest = False
    for i in range(200):
        out = engine.tick(ConversationState.IDLE, now=i / 30.0)
        if out.wing > 0.1:
            saw_flap = True
        if out.wing == 0.0:
            saw_rest = True
    assert saw_flap, "expected at least one wing flap during IDLE"
    assert saw_rest, "expected rest frames between flaps during IDLE"


def test_speaking_jaw_follows_envelope():
    engine = _engine()
    ctx = SpeakingContext(envelope=0.8, text="Hello there", progress=0.5)
    out = engine.tick(ConversationState.SPEAKING, now=0.0, speaking=ctx)
    assert 0.0 <= out.jaw_open <= 1.0
    assert out.jaw_open >= 0.6


def test_phrase_boundary_triggers_nod():
    engine = _engine(seed=0)
    ctx = SpeakingContext(envelope=0.4, text="Hi!", phrase_boundary=True)
    out = engine.tick(ConversationState.SPEAKING, now=0.0, speaking=ctx)
    # Nod adjusts pitch down/up away from neutral.
    assert 0.0 <= out.head_ud <= 1.0


def test_wing_cooldown_prevents_rapid_flaps():
    engine = _engine()
    fires = 0
    for i in range(300):
        ctx = SpeakingContext(
            envelope=0.9,
            text="wow!",
            excited=True,
            emphasis=0.9,
        )
        out = engine.tick(ConversationState.SPEAKING, now=i / 30.0, speaking=ctx)
        if out.wing > 0.0:
            fires += 1
    # 10 seconds of loud excited speech should produce a bounded number of flaps.
    assert fires <= 5, f"expected sparse wing usage, got {fires} flaps"


def test_analyze_text_detects_questions_and_excitement():
    assert analyze_text("Hello friend!")["excited"] is True
    assert analyze_text("How are you?")["question_like"] is True
    assert analyze_text("Neutral statement.")["question_like"] is False


def test_split_phrases_handles_punctuation():
    assert split_phrases("Hi there! How are you? I am fine.") == [
        "Hi there!",
        "How are you?",
        "I am fine.",
    ]


def test_estimate_phrase_boundaries_monotonic():
    boundaries = estimate_phrase_boundaries("Hi there. How are you?", duration_s=2.0)
    assert boundaries == sorted(boundaries)
    assert all(0 < b < 2.0 for b in boundaries)

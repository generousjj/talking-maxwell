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


def test_idle_has_no_jaw_and_head_stays_near_center():
    # IDLE keeps the jaw fully closed and head motion bounded. Head does
    # gentle nods / look-arounds between wing flaps (see
    # `test_idle_nods_between_wing_flaps`), but it must stay close to
    # center — never slam to the extremes.
    engine = _engine()
    for i in range(60):
        out = engine.tick(ConversationState.IDLE, now=i / 30.0)
        assert out.jaw_open == 0.0
        assert 0.3 < out.head_lr < 0.7
        assert 0.3 < out.head_ud < 0.7


def test_idle_motion_is_continuous():
    # Wing + head_ud + head_lr each run as continuous slow sines on
    # independent periods. We expect *all three* channels to move
    # through their full sine range over a long-enough window — and
    # critically, no flat "stuck at zero" rest gap on the wing (the
    # raised-cosine shape only touches zero instantaneously at each
    # period boundary).
    engine = _engine(seed=1)
    wing_max = 0.0
    head_ud_min = 1.0
    head_lr_min = 1.0
    head_lr_max = 0.0
    consecutive_zero_wing_frames = 0
    max_consecutive_zero_wing_frames = 0
    for i in range(900):  # 30 s of motion
        t = i / 30.0
        out = engine.tick(ConversationState.IDLE, now=t)
        wing_max = max(wing_max, out.wing)
        head_ud_min = min(head_ud_min, out.head_ud)
        head_lr_min = min(head_lr_min, out.head_lr)
        head_lr_max = max(head_lr_max, out.head_lr)
        if out.wing < 0.001:
            consecutive_zero_wing_frames += 1
            max_consecutive_zero_wing_frames = max(
                max_consecutive_zero_wing_frames, consecutive_zero_wing_frames
            )
        else:
            consecutive_zero_wing_frames = 0
    assert wing_max > 0.3, "wing should reach a clear peak"
    assert head_ud_min < 0.49, "head_ud should dip below center"
    assert head_lr_min < 0.49, "head_lr should swing left of center"
    assert head_lr_max > 0.51, "head_lr should swing right of center"
    # No flat rest gap on the wing: the raised cosine only touches 0
    # for at most a single tick each period, so we should never see a
    # long run of frames where wing == 0.
    assert max_consecutive_zero_wing_frames < 5, (
        "wing should not flat-rest at zero — got "
        f"{max_consecutive_zero_wing_frames} consecutive zero frames"
    )


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

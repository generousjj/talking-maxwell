"""Gaze target integration into the behavior engine.

A confident gaze target should re-base Maxwell's head so he looks at the detected
face, while the existing speech / idle offsets still layer on top (no separate
head-motion path). A zero-confidence gaze must not perturb motion at all.
"""

from __future__ import annotations

from motion.behavior_engine import BehaviorEngine
from motion.models import (
    BehaviorGains,
    ConversationState,
    GazeContext,
    SpeakingContext,
)


def _engine(**overrides) -> BehaviorEngine:
    overrides.setdefault("seed", 7)
    return BehaviorEngine(gains=BehaviorGains(**overrides))


def test_zero_confidence_gaze_is_a_noop():
    a = _engine()
    b = _engine()
    empty_gaze = GazeContext(target_lr=0.9, target_ud=0.9, confidence=0.0)
    for i in range(30):
        t = i / 30.0
        out_a = a.tick(ConversationState.IDLE, now=t)
        out_b = b.tick(ConversationState.IDLE, now=t, gaze=empty_gaze)
        assert abs(out_a.head_lr - out_b.head_lr) < 1e-9
        assert abs(out_a.head_ud - out_b.head_ud) < 1e-9


def test_confident_gaze_pulls_head_toward_target_while_idle():
    engine = _engine()
    gaze = GazeContext(target_lr=0.85, target_ud=0.2, confidence=1.0)
    out = None
    for i in range(60):  # let the output lowpass settle
        out = engine.tick(ConversationState.IDLE, now=i / 30.0, gaze=gaze)
    # Head should sit clearly right-of-center and above center (toward the face),
    # not near the neutral 0.5 it would hold with no gaze.
    assert out.head_lr > 0.65
    assert out.head_ud < 0.4


def _idle_wander_range(engine, gaze, ticks=1800):
    """Peak-to-peak excursion of head_lr / head_ud over a long idle window."""
    lr_min = ud_min = 1.0
    lr_max = ud_max = 0.0
    for i in range(ticks):
        out = engine.tick(ConversationState.IDLE, now=i / 30.0, gaze=gaze)
        lr_min, lr_max = min(lr_min, out.head_lr), max(lr_max, out.head_lr)
        ud_min, ud_max = min(ud_min, out.head_ud), max(ud_max, out.head_ud)
    return (lr_max - lr_min), (ud_max - ud_min)


def test_confident_gaze_suppresses_idle_wander():
    # A confident lock should hold a steady look: the idle head wander shrinks a
    # lot compared to no face, while still leaving a little residual life.
    locked = _engine()
    free = _engine()
    gaze = GazeContext(target_lr=0.5, target_ud=0.5, confidence=1.0)

    locked_lr, locked_ud = _idle_wander_range(locked, gaze)
    free_lr, free_ud = _idle_wander_range(free, None)

    # Locked wander is much smaller than free-running idle wander...
    assert locked_lr < free_lr * 0.4
    assert locked_ud < free_ud * 0.4
    # ...but not dead-frozen (default suppression 0.85 keeps ~15% residual).
    assert locked_lr > 0.0
    assert locked_ud > 0.0


def test_suppression_disabled_keeps_full_wander():
    # gaze_idle_suppression=0 restores the pre-vision behavior: idle wander is
    # unaffected by a confident gaze (target at center so only the sines move it).
    off = _engine(gaze_idle_suppression=0.0)
    free = _engine(gaze_idle_suppression=0.0)
    gaze = GazeContext(target_lr=0.5, target_ud=0.5, confidence=1.0)
    off_lr, off_ud = _idle_wander_range(off, gaze)
    free_lr, free_ud = _idle_wander_range(free, None)
    assert abs(off_lr - free_lr) < 1e-9
    assert abs(off_ud - free_ud) < 1e-9


def test_gaze_tracks_vertical_up_and_down():
    # Looking up at a high face vs down at a low face should clearly separate
    # head_ud, and the idle nod sine must not wash that out.
    up = _engine()
    down = _engine()
    gaze_up = GazeContext(target_lr=0.5, target_ud=0.15, confidence=1.0)
    gaze_down = GazeContext(target_lr=0.5, target_ud=0.85, confidence=1.0)
    up_out = down_out = None
    for i in range(60):
        t = i / 30.0
        up_out = up.tick(ConversationState.IDLE, now=t, gaze=gaze_up)
        down_out = down.tick(ConversationState.IDLE, now=t, gaze=gaze_down)
    assert up_out.head_ud < 0.3
    assert down_out.head_ud > 0.7
    assert (down_out.head_ud - up_out.head_ud) > 0.4


def test_gaze_holds_while_speaking_offsets_still_apply():
    # With a confident gaze the head tracks the face; a phrase-boundary nod should
    # still perturb head_ud on top of the gaze base rather than being ignored.
    base_engine = _engine()
    nod_engine = _engine()
    gaze = GazeContext(target_lr=0.8, target_ud=0.5, confidence=1.0)

    # Warm both to the gaze base with a plain speaking context.
    plain = SpeakingContext(envelope=0.2, text="hello")
    for i in range(30):
        t = i / 30.0
        base_out = base_engine.tick(ConversationState.SPEAKING, now=t, speaking=plain, gaze=gaze)
        nod_out = nod_engine.tick(ConversationState.SPEAKING, now=t, speaking=plain, gaze=gaze)

    # Head aims at the face (right of center) on both.
    assert base_out.head_lr > 0.6
    assert nod_out.head_lr > 0.6

    # Fire a phrase-boundary nod on one engine, then step both through the
    # ~0.4s nod window; the nod should visibly move head_ud vs the baseline.
    max_diff = 0.0
    for i in range(1, 13):  # ~0.4s at 30 Hz
        t = 1.0 + i / 30.0
        b = base_engine.tick(ConversationState.SPEAKING, now=t, speaking=plain, gaze=gaze)
        ctx = SpeakingContext(
            envelope=0.2, text="Hi there!", phrase_boundary=(i == 1)
        )
        n = nod_engine.tick(ConversationState.SPEAKING, now=t, speaking=ctx, gaze=gaze)
        # Head keeps aiming at the face throughout the nod.
        assert n.head_lr > 0.6
        max_diff = max(max_diff, abs(n.head_ud - b.head_ud))
    assert max_diff > 1e-2, f"nod should perturb head_ud on top of gaze, got {max_diff}"

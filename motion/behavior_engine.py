"""Deterministic behavior engine.

Takes the current conversation state plus optional SPEAKING context and
produces a BehaviorOutput each tick. Randomness is seeded so unit tests can
make exact assertions about motion outputs.
"""

from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    BehaviorGains,
    BehaviorOutput,
    ConversationState,
    SpeakingContext,
)


_EXCITED_WORDS = {
    "wow",
    "hello",
    "hi",
    "hey",
    "yay",
    "awesome",
    "amazing",
    "great",
    "whoa",
    "squawk",
    "polly",
    "cracker",
}


def analyze_text(text: str) -> dict:
    """Lightweight text analysis that feeds into SpeakingContext heuristics."""
    stripped = (text or "").strip()
    lowered = stripped.lower()
    question = stripped.endswith("?")
    exclamation = stripped.endswith("!")
    words = re.findall(r"[a-z']+", lowered)
    excited = exclamation or any(w in _EXCITED_WORDS for w in words)
    return {
        "question_like": question,
        "excited": excited,
        "words": words,
    }


def split_phrases(text: str) -> list[str]:
    """Split text into rough phrase segments using punctuation boundaries."""
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!,;:])\s+", text.strip())
    return [p for p in parts if p]


def estimate_phrase_boundaries(text: str, duration_s: float) -> list[float]:
    """Estimate timestamps (seconds) where phrase boundaries occur.

    Distributes boundaries proportional to phrase character length. Used only
    as a lightweight heuristic when richer TTS timing data is unavailable.
    """
    phrases = split_phrases(text)
    if not phrases or duration_s <= 0:
        return []
    total_chars = sum(len(p) for p in phrases) or 1
    boundaries: list[float] = []
    elapsed = 0.0
    for phrase in phrases[:-1]:
        elapsed += duration_s * (len(phrase) / total_chars)
        boundaries.append(elapsed)
    return boundaries


@dataclass
class BehaviorEngine:
    """Deterministic behavior engine.

    The engine is stateful to support slow drift and cooldowns. Call ``tick``
    at a fixed rate (matching the scheduler) with the current state and, when
    speaking, the latest SpeakingContext.
    """

    gains: BehaviorGains = field(default_factory=BehaviorGains)

    _rng: random.Random = field(init=False)
    _last_wing_time: float = 0.0
    _yaw_drift: float = 0.5
    _yaw_target: float = 0.5
    _pitch_drift: float = 0.5
    _pitch_target: float = 0.5
    _last_tick: float = 0.0
    _nod_until: float = 0.0
    _nod_strength: float = 0.0
    _tilt_until: float = 0.0
    _tilt_direction: float = 1.0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.gains.seed)

    def reset(self) -> None:
        self._rng = random.Random(self.gains.seed)
        self._last_wing_time = 0.0
        self._yaw_drift = 0.5
        self._yaw_target = 0.5
        self._pitch_drift = 0.5
        self._pitch_target = 0.5
        self._last_tick = 0.0
        self._nod_until = 0.0
        self._nod_strength = 0.0
        self._tilt_until = 0.0
        self._tilt_direction = 1.0

    def tick(
        self,
        state: ConversationState,
        now: float,
        speaking: Optional[SpeakingContext] = None,
    ) -> BehaviorOutput:
        dt = max(0.0, min(0.5, now - self._last_tick)) if self._last_tick else 1 / 30
        self._last_tick = now
        self._update_drift(dt, state)

        if state == ConversationState.SPEAKING:
            return self._speaking(now, dt, speaking or SpeakingContext())
        if state == ConversationState.LISTENING:
            return self._listening(now, dt)
        if state == ConversationState.THINKING:
            return self._thinking(now, dt)
        return self._idle(now, dt)

    def _update_drift(self, dt: float, state: ConversationState) -> None:
        # Non-speaking states hold dead-center: Maxwell should look calm and
        # attentive while waiting for the user to talk or while processing a
        # reply, not "drifting for no reason". Snap drift targets back to
        # center quickly so entering SPEAKING starts from a neutral pose.
        if state != ConversationState.SPEAKING:
            self._yaw_target = 0.5
            self._pitch_target = 0.5
            follow_tau = 0.35
            follow = 1.0 - math.exp(-dt / follow_tau)
            self._yaw_drift += (0.5 - self._yaw_drift) * follow
            self._pitch_drift += (0.5 - self._pitch_drift) * follow
            return

        change_rate = self.gains.speaking_drift_rate
        if self._rng.random() < change_rate * dt:
            magnitude = self.gains.head_lr_drift
            self._yaw_target = _clamp01(0.5 + self._rng.uniform(-magnitude, magnitude))
        if self._rng.random() < change_rate * dt * 0.7:
            magnitude = self.gains.head_ud_drift
            self._pitch_target = _clamp01(
                0.5 + self._rng.uniform(-magnitude, magnitude)
            )

        follow = 1.0 - math.exp(-dt / 0.45)
        self._yaw_drift += (self._yaw_target - self._yaw_drift) * follow
        self._pitch_drift += (self._pitch_target - self._pitch_drift) * follow

    def _waiting_wing(self, now: float) -> float:
        """Periodic wing flap used in IDLE / LISTENING / THINKING states.

        Keeps Maxwell looking alive without adding head motion (important
        during LISTENING so mic recording stays clean). Each period the
        wing does one half-sine flap over ``duty`` of the period, then
        rests at zero for the remainder.
        """
        period = max(0.2, self.gains.waiting_wing_period_s)
        duty = min(0.95, max(0.05, self.gains.waiting_wing_duty))
        phase = (now % period) / period
        if phase >= duty:
            return 0.0
        flap = math.sin(math.pi * (phase / duty))
        return self.gains.waiting_wing_strength * flap

    def _idle(self, now: float, dt: float) -> BehaviorOutput:
        # Head and jaw completely still between turns; only the wing
        # wiggles so Maxwell doesn't look frozen.
        return BehaviorOutput(
            jaw_open=0.0,
            head_lr=_clamp01(self._yaw_drift),
            head_ud=_clamp01(self._pitch_drift),
            wing=_clamp01(self._waiting_wing(now)),
        )

    def _listening(self, now: float, dt: float) -> BehaviorOutput:
        # Hold head and jaw still while the mic is recording so servo
        # chatter doesn't leak into the captured audio. Wings are far
        # enough from the bird's mic path that a periodic flap is fine.
        return BehaviorOutput(
            jaw_open=0.0,
            head_lr=_clamp01(self._yaw_drift),
            head_ud=_clamp01(self._pitch_drift),
            wing=_clamp01(self._waiting_wing(now)),
        )

    def _thinking(self, now: float, dt: float) -> BehaviorOutput:
        # Thinking is usually brief; keep the same waiting flap going so
        # there's no jarring wing stop just because STT/LLM is running.
        return BehaviorOutput(
            jaw_open=0.0,
            head_lr=_clamp01(self._yaw_drift),
            head_ud=_clamp01(self._pitch_drift),
            wing=_clamp01(self._waiting_wing(now)),
        )

    def _speaking(
        self, now: float, dt: float, context: SpeakingContext
    ) -> BehaviorOutput:
        if context.phrase_boundary:
            self._nod_until = now + 0.4
            self._nod_strength = self.gains.nod_strength
            if context.question_like and self._rng.random() < 0.6:
                self._tilt_until = now + 0.7
                self._tilt_direction = 1.0 if self._rng.random() < 0.5 else -1.0

        emphasis_bump = 0.0
        if context.emphasis > 0.45:
            emphasis_bump = self.gains.emphasis_strength * context.emphasis

        nod = 0.0
        if now < self._nod_until:
            remaining = max(0.0, self._nod_until - now)
            nod = -self._nod_strength * math.sin(math.pi * (1.0 - remaining / 0.4))

        tilt = 0.0
        if now < self._tilt_until:
            remaining = max(0.0, self._tilt_until - now)
            tilt = (
                self.gains.question_tilt
                * self._tilt_direction
                * math.sin(math.pi * (1.0 - remaining / 0.7))
            )

        # Continuous envelope-driven head bob: louder syllables tip the head up
        # slightly. This is what makes the bird look like it's actually
        # following its own speech instead of just drifting randomly.
        envelope_bob = -self.gains.envelope_head_bob * context.envelope

        # Wings: fire more readily but still cooldown-gated. We drop the
        # "excited word" requirement because excited-only flaps made them so
        # rare they felt broken; any loud syllable past the threshold with the
        # cooldown expired is now eligible.
        wing = 0.0
        can_wing = (
            context.envelope > 0.45
            and (now - self._last_wing_time) > self.gains.wing_cooldown_s
        )
        if can_wing:
            excited_boost = 0.35 if context.excited else 0.0
            if self._rng.random() < 0.20 + excited_boost:
                wing = self.gains.wing_strength
                self._last_wing_time = now

        return BehaviorOutput(
            jaw_open=_clamp01(context.envelope + emphasis_bump),
            head_lr=_clamp01(self._yaw_drift + tilt),
            head_ud=_clamp01(
                self._pitch_drift + envelope_bob - emphasis_bump * 0.35 + nod
            ),
            wing=_clamp01(wing),
        )


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)

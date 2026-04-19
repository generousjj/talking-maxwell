"""Core data models shared across the motion pipeline.

All normalized motion values are in the inclusive range [0.0, 1.0]. Centered
channels (head_lr, head_ud) default to 0.5 which means "neutral / centered".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConversationState(str, Enum):
    """High-level conversation state that drives motion behavior profiles."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class MotionFrame:
    """A single normalized motion frame sent to the transport backend.

    All values must be clamped to [0.0, 1.0]. Centered channels default to 0.5.
    """

    jaw_open: float = 0.0
    head_lr: float = 0.5
    head_ud: float = 0.5
    wing: float = 0.0
    timestamp: float = 0.0

    def clamped(self) -> "MotionFrame":
        return MotionFrame(
            jaw_open=_clamp01(self.jaw_open),
            head_lr=_clamp01(self.head_lr),
            head_ud=_clamp01(self.head_ud),
            wing=_clamp01(self.wing),
            timestamp=self.timestamp,
        )

    def as_dict(self) -> dict:
        return {
            "jaw_open": self.jaw_open,
            "head_lr": self.head_lr,
            "head_ud": self.head_ud,
            "wing": self.wing,
            "timestamp": self.timestamp,
        }


@dataclass
class SpeakingContext:
    """Context supplied to the behavior engine while the bot is SPEAKING.

    The envelope is the current normalized loudness [0..1] from the TTS audio.
    ``text`` is the full utterance being spoken; ``progress`` is 0..1 through
    playback. ``phrase_boundary`` signals a phrase-start nod opportunity.
    """

    envelope: float = 0.0
    text: str = ""
    progress: float = 0.0
    phrase_boundary: bool = False
    emphasis: float = 0.0
    question_like: bool = False
    excited: bool = False


@dataclass
class BehaviorOutput:
    """Raw behavior engine outputs prior to final clamping."""

    jaw_open: float = 0.0
    head_lr: float = 0.5
    head_ud: float = 0.5
    wing: float = 0.0

    def to_frame(self, timestamp: float = 0.0) -> MotionFrame:
        return MotionFrame(
            jaw_open=_clamp01(self.jaw_open),
            head_lr=_clamp01(self.head_lr),
            head_ud=_clamp01(self.head_ud),
            wing=_clamp01(self.wing),
            timestamp=timestamp,
        ).clamped()


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


@dataclass
class JawCalibration:
    """Jaw envelope shaping parameters.

    Attack/release are per-frame smoothing coefficients in [0, 1]; higher means
    faster response. ``noise_floor`` is the RMS threshold below which the jaw
    is treated as silent. ``floor`` / ``ceiling`` bound the final output so the
    jaw never fully slams shut or over-extends.
    """

    floor: float = 0.0
    """Output value when the signal is below noise_floor. 0.0 means fully
    closed between utterances (the servo parks at the closed PWM endpoint)."""
    ceiling: float = 1.0
    """Output value on loud peaks. 1.0 means peaks drive to the opposite PWM
    endpoint, i.e. fully open given the configured min_pwm/max_pwm range."""
    noise_floor: float = 0.010
    """RMS threshold below which the jaw is treated as silent."""
    attack: float = 0.55
    """Per-frame rise coefficient. Higher means the jaw snaps open faster on
    syllable onsets. Kept at 0.55 (was 0.85) so the target doesn't slam
    back and forth 20 times a second — that looked fine on a short-wire
    bench but starves a jaw servo with extension wires of a chance to
    actually reach each target before we flip direction."""
    release: float = 0.22
    """Per-frame fall coefficient. Keep this below attack so the jaw glides
    closed rather than snapping shut — looks much more natural."""
    peak_hold_ms: float = 60.0
    """Briefly hold the current peak so short loud spikes stay visible.
    Widened from 40ms so the jaw has time to physically open at peaks
    before the envelope starts decaying back toward rest."""
    gain: float = 7.5
    """Multiplies RMS before mapping to [floor, ceiling]. Higher = jaw opens
    more during quiet speech. Tune so typical speech sits roughly mid-range
    and peaks reliably clamp to ceiling."""


@dataclass
class BehaviorGains:
    """Tunable magnitudes and cooldowns for behavior heuristics.

    Amplitudes are fractions of each channel's full normalized range. For
    example, ``head_lr_drift = 0.25`` means the yaw target wanders up to
    ±25% of full travel from center. Defaults are tuned for a clearly
    visible, expressive parrot — dial them down in config.yaml if you want
    a calmer vibe.
    """

    head_lr_drift: float = 0.25
    head_ud_drift: float = 0.18
    nod_strength: float = 0.28
    emphasis_strength: float = 0.18
    question_tilt: float = 0.22
    wing_strength: float = 0.85
    wing_cooldown_s: float = 2.0
    idle_jitter: float = 0.03
    listening_jitter: float = 0.05
    thinking_tilt: float = 0.12
    envelope_head_bob: float = 0.12
    """How strongly loud syllables push the head up. Separate from nods
    (which fire at phrase starts) so the head moves continuously while
    Maxwell talks."""
    speaking_drift_rate: float = 0.8
    """Probability-per-second of picking a new random head-drift target
    while speaking. Higher = more restless head motion."""
    waiting_wing_period_s: float = 2.16
    """Seconds per wing flap-cycle while waiting (IDLE / LISTENING /
    THINKING). Head stays still so mic recording isn't polluted by servo
    noise, but a periodic wing flap keeps Maxwell looking alive. With the
    default duty of 0.52 this is a ~1.12s flap and a ~1.04s rest."""
    waiting_wing_strength: float = 0.55
    """Peak amplitude of the waiting flap. Lower than the full excited
    SPEAKING flap (wing_strength) so it reads as a "wiggle" rather than a
    big emphasis gesture."""
    waiting_wing_duty: float = 0.52
    """Fraction of each wing period spent actively flapping (rest of the
    period the wing sits at zero). Raised from 0.35 so the gap between
    flaps is roughly half what it used to be — Maxwell feels much more
    alive without the servo buzzing continuously."""
    seed: Optional[int] = None

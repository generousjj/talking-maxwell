"""YAML + environment configuration loader.

Uses only the stdlib plus PyYAML to stay light. Environment variables (loaded
from ``.env`` if python-dotenv is installed) take precedence for secrets such
as ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required. Install it via `pip install pyyaml`."
    ) from exc

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore[misc]
        return False

from motion.models import BehaviorGains, JawCalibration


@dataclass
class ProvidersConfig:
    stt: str = "openai_whisper"
    llm: str = "openai"
    tts: str = "openai"
    llm_model: str = "gpt-4o-mini"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "ballad"
    tts_instructions: str = (
        "Voice: a cheerful, animated cartoon parrot. Slightly nasal, bright, "
        "with a playful warble on vowels and a light British-pirate swagger. "
        "Pace: lively, varied, with small pauses for personality. Do not "
        "overact; keep phrases clear."
    )
    stt_model: str = "whisper-1"


@dataclass
class AudioConfig:
    input_device: Optional[str | int] = None
    input_sample_rate: int = 16000
    playback_device: Optional[str | int] = None
    frame_ms: float = 20.0
    vad_threshold: float = 0.012
    vad_silence_hangover_s: float = 1.2
    max_utterance_s: float = 15.0


@dataclass
class MotionConfig:
    rate_hz: float = 30.0
    jaw: JawCalibration = field(default_factory=JawCalibration)
    behavior: BehaviorGains = field(default_factory=BehaviorGains)


@dataclass
class BottangoServoChannel:
    """Per-channel servo calibration for the serial backend.

    Defaults match the Maxwell reference project values. Min/max PWM and
    max slew-rate are enforced by the firmware, giving us cheap safety
    rails regardless of what the behavior engine commands.
    """

    pin: int
    min_pwm: int
    max_pwm: int
    max_pwm_per_sec: int = 2500
    starting_pwm: Optional[int] = None
    invert: bool = False


@dataclass
class BottangoSerialConfig:
    port: Optional[str] = None
    baud: int = 115200
    auto_detect: bool = True
    handshake_timeout_s: float = 6.0
    command_timeout_s: float = 1.5
    compressed_signal_max: int = 8192
    min_delta_for_send: float = 0.004
    jaw_min_delta: float = 0.020
    jaw_min_send_interval_s: float = 0.08
    jaw: BottangoServoChannel = field(
        default_factory=lambda: BottangoServoChannel(
            pin=9, min_pwm=1450, max_pwm=1775, max_pwm_per_sec=1200
        )
    )
    head_lr: BottangoServoChannel = field(
        default_factory=lambda: BottangoServoChannel(
            pin=5, min_pwm=1275, max_pwm=1725, max_pwm_per_sec=1800
        )
    )
    head_ud: BottangoServoChannel = field(
        default_factory=lambda: BottangoServoChannel(
            pin=6, min_pwm=850, max_pwm=2100, max_pwm_per_sec=1800
        )
    )
    wing: BottangoServoChannel = field(
        default_factory=lambda: BottangoServoChannel(
            pin=3, min_pwm=1500, max_pwm=2000, max_pwm_per_sec=3000
        )
    )


@dataclass
class BottangoConfig:
    """Configuration for Bottango motion transports.

    ``transport`` selects how we reach the hardware:

    * ``serial``  — talk Bottango's firmware protocol directly to the ESP32
                    over USB (preferred; requires no desktop app).
    * ``http``    — legacy HTTP transport (kept for reference; most Bottango
                    builds expose the live API over WebSocket instead, so
                    this only works with custom plugins).
    """

    enabled: bool = False
    transport: str = "serial"
    serial: BottangoSerialConfig = field(default_factory=BottangoSerialConfig)

    # Legacy HTTP transport settings (kept for completeness).
    base_url: str = "http://localhost:59224"
    path_template: str = "/setInputValue/{identifier}/{value}"
    value_scale: float = 1.0
    request_timeout_s: float = 0.25
    health_path: str = "/"
    jaw_identifier: str = "jaw_api"
    head_lr_identifier: str = "head_lr_api"
    head_ud_identifier: str = "head_ud_api"
    wing_identifier: str = "wing_api"


@dataclass
class RealtimeConfig:
    """OpenAI Realtime API mode (low-latency speech-in / speech-out).

    Independent of the regular STT / LLM / TTS providers; toggled at
    runtime via the webapp. ``instructions`` falls back to
    :class:`AppConfig.personality` if left blank, so by default the
    Realtime voice inherits the same personality as the typed-text mode.

    The VAD knobs default to noisy-environment-friendly values:
    far-field server-side noise reduction (good for a laptop mic at
    events), a higher activation threshold (0.7 instead of OpenAI's
    default 0.5), and a longer trailing silence window so the user can
    pause mid-sentence without getting cut off.
    """

    enabled: bool = True
    model: str = "gpt-realtime"
    voice: str = "ballad"
    instructions: str = ""

    # ---- Voice activity detection ----
    vad_type: str = "server_vad"  # "server_vad" | "semantic_vad"
    vad_threshold: float = 0.7
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 700
    vad_eagerness: str = "low"  # only used when vad_type == "semantic_vad"

    # ---- Server-side noise reduction ----
    # "off" | "near_field" (headsets/AirPods) | "far_field" (laptop /
    # conference mic in noisy room). Far-field is the right pick for a
    # laptop running at an event.
    noise_reduction: str = "far_field"

    # ---- Half-duplex echo suppression ----
    # OpenAI's Realtime API has no built-in AEC. With Maxwell's speaker
    # near the laptop mic the assistant's voice loops back into the mic
    # and he runs away talking to himself. Half-duplex mode stops
    # forwarding mic frames to the server while playback is in flight,
    # plus a tail to swallow speaker reverb. Disable only if the mic is
    # genuinely isolated from playback (headset / AirPods).
    half_duplex: bool = True
    playback_tail_ms: int = 400

    # Smart barge-in: while half_duplex has the mic muted, watch local
    # mic RMS; if the user is clearly louder than speaker leakage the
    # mic un-mutes and the in-flight server response is cancelled so
    # the user can interrupt naturally. Disable to revert to "wait
    # your turn" half-duplex.
    barge_in_enabled: bool = True
    barge_in_rms_threshold: float = 0.06
    barge_in_above_ambient_factor: float = 5.0
    barge_in_min_frames: int = 4

    # Push-to-talk: when true, server VAD is disabled and the mic only
    # streams while the user holds the PTT key/button. Useful in noisy
    # rooms (booths, demos) where any auto-VAD would constantly false-
    # trigger. Mutually exclusive with barge-in (PTT does its own
    # bargein on key-down).
    push_to_talk: bool = False


@dataclass
class VisionConfig:
    """Webcam vision: face tracking (fast, local, free) + scene understanding
    (slow, paid, on-demand).

    Face tracking runs a webcam through a face detector at ``tracking_fps`` and
    turns Maxwell's head toward the nearest face. It's off by default so the app
    still runs with no camera; enable it here or toggle it live from the operator
    UI. The ``gaze_*`` knobs shape how face position maps to head motion — the
    ``invert_*`` flags exist because whether a mirrored webcam / a given servo
    mounting turns "the right way" can't be known ahead of time (flip them if
    Maxwell looks *away* from you instead of *at* you). Use
    ``python tools/vision_preview.py`` to tune these before running the bird.

    Scene understanding is only ever called on demand (operator button or a spoken
    request), so it has no idle cost. ``scene_min_interval_s`` coalesces rapid
    re-triggers, and ``scene_max_tokens`` caps the reply length — both cost rails.
    """

    enabled: bool = False
    camera_index: int = 0
    tracking_fps: float = 12.0
    detector: str = "auto"  # "auto" | "mediapipe" | "opencv"
    """Face detector backend. ``auto`` prefers MediaPipe when its legacy
    ``solutions`` API is available (Python <=3.12) and otherwise falls back to
    OpenCV's bundled Haar cascade — which needs no model download and works on
    every Python, including 3.13 where the MediaPipe solutions API is gone."""

    # ---- Gaze mapping (face position -> head target) ----
    gaze_gain_lr: float = 1.4
    """How far the head swings left/right for a given horizontal face offset."""
    gaze_gain_ud: float = 1.7
    """How far the head tilts up/down for a given vertical face offset. Higher
    than the horizontal gain so Maxwell clearly tips up/down toward the person's
    face height rather than only swivelling side to side."""
    invert_lr: bool = False
    """Flip if Maxwell turns away from the person horizontally."""
    invert_ud: bool = False
    """Flip if Maxwell tilts the wrong way vertically."""
    deadzone: float = 0.03
    """Face offsets smaller than this (fraction of frame) are treated as centered,
    so a roughly-centered face doesn't make the head hunt. Kept small so modest
    up/down head movements still register."""
    lost_face_timeout_s: float = 1.5
    """Seconds over which confidence decays after a face leaves frame; the head
    eases back to procedural motion as it drops."""

    # ---- Face recognition / memory ----
    recognition_enabled: bool = False
    """Turn on face recognition (remember people by name). Requires insightface +
    onnxruntime; if they're missing the app logs a warning and runs without it."""
    recognition_fps: float = 3.0
    """How often to run recognition, in Hz. Decoupled from (and much slower than)
    the 30 fps head tracking — identity doesn't need every frame."""
    recognition_threshold: float = 0.40
    """Minimum cosine similarity to accept a match. A calibration starting point,
    not a constant — tune with tools/recognition_calibrate.py for your camera."""
    recognition_margin: float = 0.05
    """Rejection band: if the top two candidates are within this of each other,
    return 'unknown' rather than risk a confident wrong name."""
    recognition_votes: int = 8
    """Frames of agreement (temporal voting) before committing to an identity, so a
    single bad frame can't make Maxwell blurt the wrong name."""
    stranger_match_threshold: float = 0.45
    """Cosine similarity for telling unrecognized people apart within a session:
    two unknown faces above this are treated as the same stranger (so a person who
    steps out and back isn't re-greeted), below it as different people (so each new
    stranger is greeted). Higher = more likely to treat similar faces as distinct."""
    min_sharpness: float = 60.0
    """Laplacian-variance floor for enrollment: blurrier face crops (from the moving
    head) are skipped so they don't pollute a person's stored embeddings."""
    max_embeddings_per_person: int = 20
    """Cap on stored embeddings per person (oldest dropped). A spread of views across
    angles/lighting matters more than any single one."""
    enroll_sample_interval_s: float = 1.0
    """While talking to an unrecognized person, sample at most one embedding this
    often into the pending buffer."""
    greet_on_sight: bool = True
    """Make Maxwell speak first when a face is committed — greet remembered people
    by name, and say hello / ask the name of someone new. Only fires in realtime
    (voice) mode and never over an in-progress turn."""
    greeting_cooldown_s: float = 600.0
    """Don't greet a face again until it's been *unseen* for this long (default 10
    min). 'Last seen' is refreshed every frame the face is on screen, so someone in
    an ongoing conversation stays 'seen' and is never re-greeted even when detection
    flickers in and out. Only a genuine return after a long absence greets again."""
    insightface_model: str = "buffalo_l"
    """InsightFace model bundle (SCRFD detector + ArcFace recogniser)."""
    memory_persist: bool = False
    """Persist face memory to disk so people are remembered across restarts. Off by
    default (session-only, nothing biometric written). Enabling this WRITES BIOMETRIC
    DATA to memory_path (gitignored)."""
    memory_path: str = "data/face_memory.json"
    """Where the persisted face memory lives (only used when memory_persist is on)."""

    # ---- Scene understanding (vision LLM) ----
    scene_enabled: bool = False
    """Allow paid image analysis ("what do you see?"). Off by default because every
    describe sends a camera frame to OpenAI. When off, the look_and_describe voice
    tool isn't offered to the model and the operator button is disabled. Face
    tracking and face recognition are local and unaffected by this switch."""
    scene_provider: str = "openai"  # "openai" | "stub"
    scene_model: str = "gpt-4o-mini"  # vision-capable, cheapest suitable model
    scene_max_tokens: int = 120
    scene_min_interval_s: float = 3.0
    scene_prompt: str = (
        "You are Maxwell, a witty animatronic parrot, describing what you see "
        "through your camera. In one short, vivid sentence say what's in front of "
        "you — objects, people, or what someone is showing you. Stay in character."
    )


@dataclass
class LoggingConfig:
    level: str = "INFO"
    motion_csv_path: Optional[str] = None
    plot_after: bool = False
    log_every_n_motion: int = 15


@dataclass
class AppConfig:
    mode: str = "typed"  # "typed" | "live"
    backend: str = "mock"  # "mock" | "bottango"
    personality: str = (
        "You are Maxwell, a cheerful animatronic parrot. Keep replies to one or "
        "two short sentences. Occasionally say 'squawk!' or 'polly!' for flavor."
    )
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    bottango: BottangoConfig = field(default_factory=BottangoConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: Optional[str] = None, *, env_file: Optional[str] = None) -> AppConfig:
    """Load YAML config and merge with defaults.

    ``env_file`` (defaults to ``.env`` next to the config) is loaded into the
    process environment; values that are not present there are not overridden.
    Config resolution order (later wins): defaults < YAML file.
    """
    config = AppConfig()

    if env_file is None:
        # Try ./.env first, then project root .env next to this file.
        candidate = Path.cwd() / ".env"
        if candidate.exists():
            env_file = str(candidate)
        else:
            here = Path(__file__).resolve().parent.parent / ".env"
            if here.exists():
                env_file = str(here)
    if env_file:
        load_dotenv(env_file, override=False)

    if path is None:
        for candidate in ("config.yaml", "config.example.yaml"):
            cand = Path(candidate)
            if cand.exists():
                path = str(cand)
                break
    if path and Path(path).exists():
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        _apply(config, raw)
    return config


def _apply(config: AppConfig, raw: dict[str, Any]) -> None:
    for key, value in raw.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply_dc(current, value)
        else:
            setattr(config, key, value)


def _apply_dc(target: Any, raw: dict[str, Any]) -> None:
    for key, value in raw.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply_dc(current, value)
        else:
            setattr(target, key, value)


def dump_defaults_yaml() -> str:
    """Useful when regenerating config.example.yaml."""
    return yaml.safe_dump(asdict(AppConfig()), sort_keys=False)

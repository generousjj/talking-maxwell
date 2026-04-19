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

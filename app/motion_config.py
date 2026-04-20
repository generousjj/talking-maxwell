"""Single source of truth for motion config shared by local + browser modes.

Reads the tuned values out of ``config.yaml`` (Maxwell's hardware +
behavior calibration) and returns a JSON-serializable dict suitable
for shipping to the browser. If the file is missing or unparseable,
falls back to the values that matched the reference hardware at the
time this module was written — the browser will still run, just with
defaults that may not perfectly match *your* Maxwell.

Why this exists:
    Before this module existed, ``static/web/js/serial.js`` and
    ``static/web/js/behavior.js`` each hardcoded their own copies of
    pin numbers, PWM ranges, and behavior gains. Those copies drifted
    from ``config.yaml`` (the values that actually drive the working
    Python operator mode), and the browser mode silently sent head
    commands to the wrong GPIO pins. Keeping one Python-side parser
    and one JSON payload fixes that permanently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


# Browser-facing defaults. These MUST match ``config.yaml`` so the
# browser behaves identically to the operator build when config.yaml
# isn't available (e.g. a fresh clone, or a Vercel build where
# includeFiles somehow dropped it).
DEFAULT_CHANNELS: Dict[str, Dict[str, Any]] = {
    "jaw":     {"pin": 9, "minPwm": 1450, "maxPwm": 1800, "maxPwmPerSec": 3000, "inverted": True},
    "head_lr": {"pin": 5, "minPwm": 1275, "maxPwm": 1725, "maxPwmPerSec": 1800, "inverted": False},
    "head_ud": {"pin": 6, "minPwm": 850,  "maxPwm": 2100, "maxPwmPerSec": 1800, "inverted": False},
    "wing":    {"pin": 3, "minPwm": 1500, "maxPwm": 2000, "maxPwmPerSec": 3000, "inverted": False},
}

DEFAULT_GAINS: Dict[str, Any] = {
    # Mirrors motion.BehaviorGains + motion.behavior overrides from
    # config.yaml, translated to camelCase for JS. Every field in the
    # Python dataclass is represented here so the browser engine can
    # execute the same algorithm path-for-path (see behavior.js).
    "headLrDrift":           0.32,   # head_lr_drift
    "headUdDrift":           0.26,   # head_ud_drift
    "nodStrength":           0.45,   # nod_strength (phrase-boundary nod)
    "emphasisStrength":      0.26,   # emphasis_strength (loud-syllable bump)
    "questionTilt":          0.32,   # question_tilt (head_lr tilt on ?)
    "wingStrength":          1.0,    # wing_strength (speaking flap)
    "wingCooldownS":         2.0,    # wing_cooldown_s
    "waitingWingStrength":   0.69,   # waiting_wing_strength (idle flap)
    "waitingWingPeriodS":    2.16,   # waiting_wing_period_s
    "envelopeHeadBob":       0.22,   # envelope_head_bob
    "speakingDriftRate":     0.8,    # speaking_drift_rate
    "idleNodStrength":       0.30,   # idle_nod_strength
    "idleTiltStrength":      0.20,   # idle_tilt_strength
    "idleNodPeriodS":        3.7,    # idle_nod_period_s
    "idleTiltPeriodS":       5.1,    # idle_tilt_period_s
    "headSmoothingTauS":     0.08,   # head_smoothing_tau_s (output lowpass)
    "seed":                  None,   # int for deterministic motion, null for random

    # Legacy aliases (older JS builds read these names). Python has
    # no separate "speakingWingStrength"/"speakingBobStrength"/
    # "idleWingPeriodS" knobs — those were invented in the old JS port
    # — so we mirror them from the canonical Python fields above to
    # keep any still-cached browser bundles working during rollout.
    "speakingWingStrength":  1.0,    # == wing_strength
    "speakingBobStrength":   0.22,   # == envelope_head_bob
    "idleWingPeriodS":       2.16,   # == waiting_wing_period_s
}

DEFAULT_JAW_CALIBRATION: Dict[str, float] = {
    "floor":      0.08,
    "ceiling":    0.90,
    "noiseFloor": 0.010,
    "attack":     0.55,
    "release":    0.20,
    "peakHoldMs": 40.0,
    "gain":       1.6,
}


def load_motion_config(repo_root: Path) -> Dict[str, Any]:
    """Return ``{channels, gains, jaw_calibration, source}`` for the browser.

    ``source`` is the literal string "config.yaml" when the file was
    read, or "defaults" when we fell back.
    """
    channels = {k: dict(v) for k, v in DEFAULT_CHANNELS.items()}
    gains = dict(DEFAULT_GAINS)
    jaw_cal = dict(DEFAULT_JAW_CALIBRATION)

    yaml_path = repo_root / "config.yaml"
    if not yaml_path.is_file():
        return {
            "channels": channels,
            "gains": gains,
            "jaw_calibration": jaw_cal,
            "source": "defaults",
        }

    try:
        with yaml_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception:
        return {
            "channels": channels,
            "gains": gains,
            "jaw_calibration": jaw_cal,
            "source": "defaults",
        }

    # Channels live under bottango.serial.channels in config.yaml.
    serial_cfg = ((raw.get("bottango") or {}).get("serial") or {}).get("channels") or {}
    for name, cfg in serial_cfg.items():
        if name not in channels or not isinstance(cfg, dict):
            continue
        ch = channels[name]
        if "pin" in cfg:
            ch["pin"] = _as_int(cfg["pin"], ch["pin"])
        if "min_pwm" in cfg:
            ch["minPwm"] = _as_int(cfg["min_pwm"], ch["minPwm"])
        if "max_pwm" in cfg:
            ch["maxPwm"] = _as_int(cfg["max_pwm"], ch["maxPwm"])
        if "max_pwm_per_sec" in cfg:
            ch["maxPwmPerSec"] = _as_int(cfg["max_pwm_per_sec"], ch["maxPwmPerSec"])
        if "invert" in cfg:
            ch["inverted"] = bool(cfg["invert"])
        if "starting_pwm" in cfg and cfg["starting_pwm"] is not None:
            ch["startingPwm"] = _as_int(cfg["starting_pwm"], 1500)

    # Mirror transport/bottango_serial_backend.resolved_start: when
    # startingPwm is missing or null, default to the midpoint of
    # (minPwm, maxPwm). Prevents sending startingPwm=0 to the firmware,
    # which would park servos at a bogus PWM before the first sCI.
    for name, ch in channels.items():
        if "startingPwm" not in ch or ch.get("startingPwm") is None:
            try:
                ch["startingPwm"] = int(round((ch["minPwm"] + ch["maxPwm"]) / 2))
            except (TypeError, ValueError, KeyError):
                ch["startingPwm"] = 1500

    # Behavior gains live under motion.behavior. Every snake_case key
    # in the yaml maps to a camelCase key in the JS-facing payload so
    # the browser engine reads the same tuned values the Python engine
    # reads. Both the canonical camelCase key and its legacy alias
    # (for speaking wing/bob/period) are updated when present.
    beh = (raw.get("motion") or {}).get("behavior") or {}
    _map_float(beh, "head_lr_drift",         gains, "headLrDrift")
    _map_float(beh, "head_ud_drift",         gains, "headUdDrift")
    _map_float(beh, "nod_strength",          gains, "nodStrength")
    _map_float(beh, "emphasis_strength",     gains, "emphasisStrength")
    _map_float(beh, "question_tilt",         gains, "questionTilt")
    _map_float(beh, "wing_strength",         gains, "wingStrength")
    _map_float(beh, "wing_strength",         gains, "speakingWingStrength")
    _map_float(beh, "wing_cooldown_s",       gains, "wingCooldownS")
    _map_float(beh, "waiting_wing_strength", gains, "waitingWingStrength")
    _map_float(beh, "waiting_wing_period_s", gains, "waitingWingPeriodS")
    _map_float(beh, "waiting_wing_period_s", gains, "idleWingPeriodS")
    _map_float(beh, "envelope_head_bob",     gains, "envelopeHeadBob")
    _map_float(beh, "envelope_head_bob",     gains, "speakingBobStrength")
    _map_float(beh, "speaking_drift_rate",   gains, "speakingDriftRate")
    _map_float(beh, "idle_nod_strength",     gains, "idleNodStrength")
    _map_float(beh, "idle_tilt_strength",    gains, "idleTiltStrength")
    _map_float(beh, "idle_nod_period_s",     gains, "idleNodPeriodS")
    _map_float(beh, "idle_tilt_period_s",    gains, "idleTiltPeriodS")
    _map_float(beh, "head_smoothing_tau_s",  gains, "headSmoothingTauS")
    # seed can be an int or null; preserve either explicitly.
    if "seed" in beh:
        seed_val = beh["seed"]
        try:
            gains["seed"] = int(seed_val) if seed_val is not None else None
        except (TypeError, ValueError):
            gains["seed"] = None

    # Jaw calibration lives under motion.jaw.
    jaw = (raw.get("motion") or {}).get("jaw") or {}
    _map_float(jaw, "floor",        jaw_cal, "floor")
    _map_float(jaw, "ceiling",      jaw_cal, "ceiling")
    _map_float(jaw, "noise_floor",  jaw_cal, "noiseFloor")
    _map_float(jaw, "attack",       jaw_cal, "attack")
    _map_float(jaw, "release",      jaw_cal, "release")
    _map_float(jaw, "peak_hold_ms", jaw_cal, "peakHoldMs")
    _map_float(jaw, "gain",         jaw_cal, "gain")

    return {
        "channels": channels,
        "gains": gains,
        "jaw_calibration": jaw_cal,
        "source": "config.yaml",
    }


def _as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _map_float(src: dict, src_key: str, dst: dict, dst_key: str) -> None:
    if src_key not in src:
        return
    try:
        dst[dst_key] = float(src[src_key])
    except (TypeError, ValueError):
        pass

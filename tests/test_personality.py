"""Personality loading parity between config.yaml and the two web builds.

Regression guard: previously both the aiohttp server and the Vercel
FastAPI server shipped a compact placeholder prompt that stripped
out Stanford TEA specifics (weekly meeting time, admit weekend fair,
LA trip, alumni placements, etc.). These tests assert config.yaml
wins at runtime on both builds.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _has_full_personality(prompt: str) -> list[str]:
    """Return the list of expected keywords missing from `prompt`.

    Each keyword is something that lives in config.yaml's
    ``personality:`` block but NOT in the old fallback prompt, so
    hitting any of them proves config.yaml is being read.
    """
    needles = [
        "admit weekend",     # config-yaml-specific audience line
        "Wednesday",         # weekly meeting schedule
        "Los Angeles",       # annual trip
        "Imagineer",         # alumni placements
    ]
    return [n for n in needles if n.lower() not in prompt.lower()]


def test_config_yaml_exposes_full_personality():
    from app.personality import load_personality

    prompt = load_personality(REPO_ROOT)
    missing = _has_full_personality(prompt)
    assert not missing, (
        f"config.yaml personality missing expected keywords: {missing}"
    )


def test_aiohttp_web_app_uses_full_personality():
    # Import lazily so the whole suite doesn't depend on the
    # aiohttp bootstrap.
    from app import web_app

    missing = _has_full_personality(web_app.SYSTEM_PROMPT)
    assert not missing, (
        "app/web_app.py is still shipping the compact fallback prompt"
    )


def test_vercel_api_index_uses_full_personality(monkeypatch):
    # The FastAPI module has global side effects (logging, env checks),
    # so load it via its file path the same way the integration tests
    # do. This way we don't accidentally import twice when running the
    # full suite.
    path = REPO_ROOT / "api" / "index.py"
    spec = importlib.util.spec_from_file_location("api_index_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    missing = _has_full_personality(mod.SYSTEM_PROMPT)
    assert not missing, (
        "api/index.py is still shipping the compact fallback prompt"
    )

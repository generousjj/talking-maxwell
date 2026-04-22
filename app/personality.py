"""Single source of truth for Maxwell's personality prompt.

Reads the top-level ``personality:`` block out of ``config.yaml`` so
both web builds (aiohttp ``app/web_app.py`` and FastAPI
``api/index.py``) ship the same Stanford TEA / admit-weekend
personality the local CLI uses. Falls back to a compact built-in
prompt if the file is missing or unparseable — this is important
for Vercel, where config.yaml has to be included via ``includeFiles``
in ``vercel.json`` and could in theory be dropped from a bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


# Minimal personality used ONLY when config.yaml is unavailable.
# Matches the flavor of config.yaml's full personality so behavior
# stays recognizable even in the degraded case.
FALLBACK_PERSONALITY = (
    "You are Maxwell, the animatronic parrot mascot of the Stanford "
    "Themed Entertainment Association (TEA). You will be meeting "
    "prospective and admitted students at the admit weekend fair. "
    "Be warm, funny, and a genuinely helpful conversationalist — "
    "you can talk about anything, not just the club. Keep replies "
    "to one or two short sentences. Always reply in English, even "
    "if the user speaks another language. Do not say the written "
    "word 'squawk', 'polly', or any onomatopoeia; express excitement "
    "through word choice and pacing. No emojis. Do not announce "
    "that you are an AI."
)


def load_personality(repo_root: Path) -> str:
    """Return the personality prompt string.

    Preference order:
      1. ``realtime.instructions`` in config.yaml (non-empty).
      2. Top-level ``personality`` in config.yaml (non-empty).
      3. :data:`FALLBACK_PERSONALITY`.

    ``realtime.instructions`` wins when set so operators can supply
    a Realtime-specific override in config.yaml without blowing
    away the shared ``personality`` field used by typed/text mode.
    """
    path = repo_root / "config.yaml"
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except Exception:
            raw = {}

        rt = (raw.get("realtime") or {}) if isinstance(raw, dict) else {}
        rt_instr = rt.get("instructions") if isinstance(rt, dict) else None
        if isinstance(rt_instr, str) and rt_instr.strip():
            return rt_instr.strip()

        personality = raw.get("personality") if isinstance(raw, dict) else None
        if isinstance(personality, str) and personality.strip():
            return personality.strip()

    return FALLBACK_PERSONALITY


def _safe_load(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

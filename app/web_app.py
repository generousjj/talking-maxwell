"""Hosted browser-mode entrypoint for Maxwell.

This is the public HTTPS-deployable server. It intentionally does NOT
boot a ``ConversationPipeline``, open a serial port, or do anything
hardware-related — all hardware work happens in the end user's browser
via Web Serial on their own laptop. The server's only jobs are:

  1. Serve the password-gated web UI (``static/web/``).
  2. Mint short-lived OpenAI Realtime ephemeral tokens so the browser
     can open WebRTC directly to OpenAI without ever seeing the raw
     ``OPENAI_API_KEY``.
  3. Run a tiny "typed-turn" fallback for devices without Realtime so
     users can still type a prompt and get a spoken/animated response.

The existing local operator workflow (``python -m app.webapp``) is
completely unaffected — this is a separate, additive process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import aiohttp
from aiohttp import web
from dotenv import load_dotenv

from app.web_auth import (
    AuthConfig,
    LoginLimiter,
    auth_middleware,
    current_session,
    handle_login,
    handle_logout,
    handle_me,
)
from app.personality import load_personality

log = logging.getLogger(__name__)

DEFAULT_REALTIME_MODEL = "gpt-4o-realtime-preview"
DEFAULT_REALTIME_VOICE = "ballad"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "ballad"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# System prompt is pulled from config.yaml (the same file the local
# CLI reads) so both web builds and the local operator build share
# one personality. Without this, the hosted web UI was shipping a
# compact placeholder that didn't mention Stanford TEA's meeting
# time, the admit weekend fair, the LA trip, alumni placements, etc.
_REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT = load_personality(_REPO_ROOT)

STATIC_WEB_SUBDIR = "web"


def _web_root() -> Path:
    return Path(__file__).resolve().parent.parent / "static" / STATIC_WEB_SUBDIR


# --------------------------------------------------------------------
# HTML page handlers
# --------------------------------------------------------------------

async def handle_login_page(request: web.Request) -> web.Response:
    if current_session(request) is not None:
        raise web.HTTPFound("/")
    return _send_static_file(request, "login.html")


async def handle_index(request: web.Request) -> web.Response:
    return _send_static_file(request, "index.html")


async def handle_admits(request: web.Request) -> web.Response:
    # Cartoon-bubble, end-user-friendly UI. Same session gate as /
    # (the auth middleware already covered that); renders the static
    # asset, which pulls the browser realtime + Web Serial modules.
    return _send_static_file(request, "admits.html")


async def handle_health(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def _send_static_file(request: web.Request, name: str) -> web.Response:
    path = _web_root() / name
    if not path.exists():
        return web.Response(
            status=500,
            text=f"missing static asset: {name}",
            content_type="text/plain",
        )
    body = path.read_bytes()
    if name.endswith(".html"):
        resp = web.Response(body=body, content_type="text/html", charset="utf-8")
        # Client-side apps change often during dev; avoid stale caches on HTML.
        resp.headers["Cache-Control"] = "no-store"
        return resp
    return web.Response(body=body, content_type="application/octet-stream")


# --------------------------------------------------------------------
# OpenAI endpoints
# --------------------------------------------------------------------

async def handle_realtime_session(request: web.Request) -> web.Response:
    """Mint a short-lived OpenAI Realtime client-secret for this browser.

    The raw ``OPENAI_API_KEY`` stays server-side; the browser only sees
    a temporary ``ek_...`` token (valid ~60 s) scoped to this session.
    The browser then opens WebRTC to ``api.openai.com/v1/realtime``
    directly using that ephemeral token.
    """
    api_key = _require_openai_key(request)
    if isinstance(api_key, web.Response):
        return api_key

    try:
        body: dict = await request.json()
    except Exception:  # noqa: BLE001
        body = {}

    voice = (body.get("voice") or DEFAULT_REALTIME_VOICE).strip()
    model = (body.get("model") or DEFAULT_REALTIME_MODEL).strip()

    # OpenAI's documented ephemeral-session endpoint. Parameters we
    # pass here become the defaults for the WebRTC session.
    payload = {
        "model": model,
        "voice": voice,
        "modalities": ["audio", "text"],
        "instructions": SYSTEM_PROMPT,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {"model": "whisper-1"},
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.65,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 650,
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "realtime=v1",
    }
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers=headers,
            data=json.dumps(payload),
        ) as resp:
            raw = await resp.text()
            if resp.status >= 400:
                log.warning(
                    "OpenAI realtime session failed: %s %s", resp.status, raw[:400]
                )
                return web.json_response(
                    {"ok": False, "error": "openai_error", "status": resp.status},
                    status=502,
                )
            try:
                data = json.loads(raw)
            except Exception:  # noqa: BLE001
                return web.json_response(
                    {"ok": False, "error": "openai_parse"}, status=502
                )

    client_secret = (data.get("client_secret") or {}).get("value")
    if not client_secret:
        return web.json_response(
            {"ok": False, "error": "no_ephemeral"}, status=502
        )

    return web.json_response(
        {
            "ok": True,
            "client_secret": client_secret,
            "expires_at": (data.get("client_secret") or {}).get("expires_at"),
            "model": model,
            "voice": voice,
        }
    )


async def handle_typed_turn(request: web.Request) -> web.Response:
    """Typed fallback: run an LLM + TTS turn entirely server-side.

    Returns ``{ok, text, audio_b64, mime}``. The browser plays the
    audio and drives its own local envelope follower off the decoded
    samples for jaw motion. This keeps the raw API key server-only
    even for devices without Realtime/WebRTC support.
    """
    api_key = _require_openai_key(request)
    if isinstance(api_key, web.Response):
        return api_key
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    user_text = str(body.get("text") or "").strip()
    if not user_text:
        return web.json_response(
            {"ok": False, "error": "empty"}, status=400
        )
    voice = (body.get("voice") or DEFAULT_TTS_VOICE).strip()
    llm_model = os.environ.get("OPENAI_LLM_MODEL", DEFAULT_LLM_MODEL)
    tts_model = os.environ.get("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)

    reply_text = await _chat_completion(api_key, llm_model, user_text)
    audio_b64, mime = await _tts_speak(api_key, tts_model, voice, reply_text)
    return web.json_response(
        {
            "ok": True,
            "text": reply_text,
            "audio_b64": audio_b64,
            "mime": mime,
        }
    )


async def _chat_completion(api_key: str, model: str, user_text: str) -> str:
    timeout = aiohttp.ClientTimeout(total=25)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.8,
    }
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
        ) as resp:
            raw = await resp.text()
            if resp.status >= 400:
                log.warning("chat.completions failed: %s %s", resp.status, raw[:300])
                raise web.HTTPBadGateway(reason="llm_error")
            data = json.loads(raw)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        log.exception("bad chat completion shape: %s", exc)
        raise web.HTTPBadGateway(reason="llm_parse")


async def _tts_speak(api_key: str, model: str, voice: str, text: str) -> tuple[str, str]:
    import base64

    timeout = aiohttp.ClientTimeout(total=25)
    payload = {
        "model": model,
        "voice": voice,
        "input": text,
        "format": "mp3",
    }
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                log.warning("tts failed: %s %s", resp.status, body[:300])
                raise web.HTTPBadGateway(reason="tts_error")
            audio = await resp.read()
    return base64.b64encode(audio).decode("ascii"), "audio/mpeg"


def _require_openai_key(request: web.Request) -> Any:
    api_key = request.app["openai_api_key"]
    if not api_key:
        return web.json_response(
            {"ok": False, "error": "openai_key_missing"}, status=503
        )
    return api_key


# --------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------

def build_app(
    *,
    auth_config: AuthConfig,
    openai_api_key: Optional[str],
    realtime_voice: str = DEFAULT_REALTIME_VOICE,
    realtime_model: str = DEFAULT_REALTIME_MODEL,
) -> web.Application:
    app = web.Application(middlewares=[auth_middleware])
    app["auth_config"] = auth_config
    app["login_limiter"] = LoginLimiter()
    app["openai_api_key"] = openai_api_key
    app["realtime_voice"] = realtime_voice
    app["realtime_model"] = realtime_model

    app.router.add_get("/healthz", handle_health)
    app.router.add_get("/login", handle_login_page)
    app.router.add_get("/", handle_index)
    app.router.add_get("/admits", handle_admits)

    app.router.add_post("/api/auth/login", handle_login)
    app.router.add_post("/api/auth/logout", handle_logout)
    app.router.add_get("/api/auth/me", handle_me)

    app.router.add_post("/api/web/realtime/session", handle_realtime_session)
    app.router.add_post("/api/web/typed", handle_typed_turn)

    # Expose a tiny config endpoint so the browser knows what the
    # server-configured defaults are (without baking them into HTML).
    async def handle_config(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "realtime_model": realtime_model,
                "realtime_voice": realtime_voice,
                "has_openai_key": bool(openai_api_key),
            }
        )

    app.router.add_get("/api/web/config", handle_config)

    # Motion/channel/gain tuning, live from config.yaml so the browser
    # always sees the same values the Python operator build uses.
    async def handle_motion_config(_request: web.Request) -> web.Response:
        from app.motion_config import load_motion_config

        try:
            data = load_motion_config(Path(__file__).resolve().parent.parent)
        except Exception as exc:  # noqa: BLE001
            log.exception("motion_config load failed: %s", exc)
            return web.json_response(
                {"ok": False, "error": "motion_config_load_failed"}, status=500
            )
        return web.json_response({"ok": True, **data})

    app.router.add_get("/api/web/motion-config", handle_motion_config)

    web_root = _web_root()
    if web_root.is_dir():
        app.router.add_static(
            "/static", str(web_root.parent), show_index=False
        )
    else:
        log.warning("browser web assets dir not found: %s", web_root)

    return app


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.web_app",
        description=(
            "Hosted browser-mode Maxwell server. Serves the password-"
            "gated web UI and mints OpenAI Realtime ephemeral tokens. "
            "Hardware I/O stays in the operator's browser via Web Serial."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default 8080)")
    parser.add_argument("--env-file", default=".env", help=".env to load (default .env)")
    parser.add_argument(
        "--realtime-voice", default=DEFAULT_REALTIME_VOICE, help="Default voice"
    )
    parser.add_argument(
        "--realtime-model", default=DEFAULT_REALTIME_MODEL, help="Realtime model id"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    env_path = Path(args.env_file)
    if env_path.is_file():
        load_dotenv(env_path)
        log.info("loaded env from %s", env_path)

    auth_config = AuthConfig.from_env()
    openai_api_key = os.environ.get("OPENAI_API_KEY") or None
    if not openai_api_key:
        log.warning("OPENAI_API_KEY is not set; OpenAI endpoints will return 503.")

    app = build_app(
        auth_config=auth_config,
        openai_api_key=openai_api_key,
        realtime_voice=args.realtime_voice,
        realtime_model=args.realtime_model,
    )

    async def _run() -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.port)
        await site.start()
        log.info("Maxwell hosted web mode on http://%s:%d", args.host, args.port)
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

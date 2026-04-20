"""Vercel entrypoint — FastAPI version of the hosted browser app.

Mirrors ``app/web_app.py`` (the aiohttp server used by
Docker/Fly/Render) but built on FastAPI so Vercel's Python runtime
can invoke it directly. Both builds share the same pure auth
primitives from :mod:`app.auth_core`, so password hashing, session
cookies, rate limiting, and Origin checks are byte-for-byte identical
across deploy targets.

Routing:
    ``vercel.json`` rewrites every URL to this function (``/api``),
    and FastAPI does the actual routing. A small per-path public/
    protected split is enforced in the middleware exactly the way
    the aiohttp version does it.

What is NOT here:
    Hardware. Same as the aiohttp server — all Web Serial + motion
    logic runs in the end-user's browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# Make the project root importable so we can reuse app.auth_core.
# Vercel bundles files reached via imports + anything in includeFiles.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.auth_core import (
    AuthConfig,
    LoginLimiter,
    SESSION_COOKIE,
    build_session_payload,
    sign_session,
    verify_session,
)

log = logging.getLogger("maxwell.vercel")
logging.basicConfig(level=logging.INFO)

# ---- Config (shared across deploy targets) ----------------------------

DEFAULT_REALTIME_MODEL = "gpt-4o-realtime-preview"
DEFAULT_REALTIME_VOICE = "ballad"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "ballad"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are Maxwell, the cheerful parrot mascot of Stanford's Themed "
    "Entertainment Association (TEA). You live on a perch and meet "
    "guests at themed entertainment events. Always reply in English "
    "only — never use any other language, even if the user does. "
    "Be warm, funny, thoughtful, and conversational. Keep replies "
    "short, punchy, and easy to say aloud. Never use the words "
    "'squawk' or 'polly'. Ask a curious follow-up question when "
    "natural. You are knowledgeable about theme parks, immersive "
    "theatre, haunts, dark rides, and experiential design, but happy "
    "to chat about anything."
)

STATIC_WEB_ROOT = ROOT / "static" / "web"
STATIC_DIR = ROOT / "static"

AUTH_CONFIG = AuthConfig.from_env()
LOGIN_LIMITER = LoginLimiter()
OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY") or None
if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY is not set; OpenAI endpoints will 503.")


# --------------------------------------------------------------------
# App
# --------------------------------------------------------------------

app = FastAPI(
    title="Maxwell hosted web mode (Vercel)",
    docs_url=None,        # no auto /docs on the public booth URL
    redoc_url=None,
    openapi_url=None,
)


def _current_session(request: Request) -> Optional[dict]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return verify_session(AUTH_CONFIG.session_secret, token)


def _set_session_cookie(response: Response, payload: dict) -> None:
    token = sign_session(AUTH_CONFIG.session_secret, payload)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=AUTH_CONFIG.session_ttl_s,
        path="/",
        secure=AUTH_CONFIG.secure_cookie,
        httponly=True,
        samesite="lax",
    )


def _clear_session_cookie(response: Response) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value="",
        max_age=0,
        path="/",
        secure=AUTH_CONFIG.secure_cookie,
        httponly=True,
        samesite="lax",
    )


def _client_ip(request: Request) -> str:
    # Vercel always terminates at its edge, so trusting XFF is the
    # right call in production (and opt-in locally via the env flag).
    if os.environ.get("MAXWELL_TRUST_FORWARDED_FOR") == "1" or os.environ.get("VERCEL") == "1":
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return (request.client.host if request.client else "unknown")


class AuthAndOriginMiddleware(BaseHTTPMiddleware):
    """Session gate + Origin check, mirroring the aiohttp middleware.

    Public paths: ``/login``, ``/api/auth/login``, ``/api/auth/me``,
    ``/healthz``, and anything under ``/static/``. Everything else
    requires a valid session cookie. State-changing API calls must
    come from the configured Origin (or same-origin when unset).
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        public = (
            path == "/login"
            or path == "/healthz"
            or path == "/api/auth/login"
            or path == "/api/auth/me"
            or path.startswith("/static/")
        )
        if not public:
            if _current_session(request) is None:
                if path.startswith("/api/"):
                    return JSONResponse(
                        {"ok": False, "error": "unauthorized"}, status_code=401
                    )
                return RedirectResponse(url="/login", status_code=302)

        if request.method not in ("GET", "HEAD", "OPTIONS") and path.startswith("/api/"):
            origin = request.headers.get("origin") or request.headers.get("referer")
            if AUTH_CONFIG.allowed_origin:
                allowed = AUTH_CONFIG.allowed_origin
            else:
                host = request.headers.get("host", "")
                scheme = request.headers.get("x-forwarded-proto", request.url.scheme or "http")
                allowed = f"{scheme}://{host}"
            if origin and not origin.startswith(allowed):
                log.warning(
                    "rejecting cross-origin %s %s (origin=%s allowed=%s)",
                    request.method, path, origin, allowed,
                )
                return JSONResponse(
                    {"ok": False, "error": "forbidden_origin"}, status_code=403
                )
        return await call_next(request)


app.add_middleware(AuthAndOriginMiddleware)


# --------------------------------------------------------------------
# Static files
# --------------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    log.warning("static dir missing: %s", STATIC_DIR)


# --------------------------------------------------------------------
# Page routes
# --------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _current_session(request) is not None:
        return RedirectResponse(url="/", status_code=302)
    return _serve_html("login.html")


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return _serve_html("index.html")


def _serve_html(name: str) -> Response:
    path = STATIC_WEB_ROOT / name
    if not path.is_file():
        return Response(f"missing {name}", status_code=500, media_type="text/plain")
    return FileResponse(
        str(path),
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


# --------------------------------------------------------------------
# Auth endpoints
# --------------------------------------------------------------------

@app.post("/api/auth/login")
async def auth_login(request: Request):
    ip = _client_ip(request)
    allowed, retry_after = await LOGIN_LIMITER.check(ip)
    if not allowed:
        log.info("rate-limiting login from %s (retry after %.0fs)", ip, retry_after)
        return JSONResponse(
            {"ok": False, "error": "rate_limited", "retry_after": int(retry_after)},
            status_code=429,
        )
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        await LOGIN_LIMITER.record_failure(ip)
        return JSONResponse({"ok": False, "error": "bad_request"}, status_code=400)

    candidate = str(body.get("password") or "")
    ok = await asyncio.to_thread(AUTH_CONFIG.password_ok, candidate)
    if not ok:
        await LOGIN_LIMITER.record_failure(ip)
        return JSONResponse(
            {"ok": False, "error": "invalid_credentials"}, status_code=401
        )
    await LOGIN_LIMITER.reset(ip)
    payload = build_session_payload("operator", AUTH_CONFIG.session_ttl_s)
    resp = JSONResponse({"ok": True})
    _set_session_cookie(resp, payload)
    log.info("login ok from %s", ip)
    return resp


@app.post("/api/auth/logout")
async def auth_logout():
    resp = JSONResponse({"ok": True})
    _clear_session_cookie(resp)
    return resp


@app.get("/api/auth/me")
async def auth_me(request: Request):
    session = _current_session(request)
    if session is None:
        return JSONResponse({"ok": False, "authed": False})
    return JSONResponse(
        {
            "ok": True,
            "authed": True,
            "uid": session.get("uid"),
            "exp": session.get("exp"),
        }
    )


# --------------------------------------------------------------------
# OpenAI endpoints
# --------------------------------------------------------------------

@app.get("/api/web/config")
async def web_config():
    return JSONResponse(
        {
            "ok": True,
            "realtime_model": DEFAULT_REALTIME_MODEL,
            "realtime_voice": DEFAULT_REALTIME_VOICE,
            "has_openai_key": bool(OPENAI_API_KEY),
        }
    )


@app.post("/api/web/realtime/session")
async def realtime_session(request: Request):
    if not OPENAI_API_KEY:
        return JSONResponse(
            {"ok": False, "error": "openai_key_missing"}, status_code=503
        )
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    voice = (body.get("voice") or DEFAULT_REALTIME_VOICE).strip()
    model = (body.get("model") or DEFAULT_REALTIME_MODEL).strip()

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
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "realtime=v1",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers=headers,
                content=json.dumps(payload),
            )
    except httpx.HTTPError as exc:
        log.warning("OpenAI transport error: %s", exc)
        return JSONResponse(
            {"ok": False, "error": "openai_transport"}, status_code=502
        )
    if resp.status_code >= 400:
        log.warning("OpenAI realtime session failed: %s %s", resp.status_code, resp.text[:400])
        return JSONResponse(
            {"ok": False, "error": "openai_error", "status": resp.status_code},
            status_code=502,
        )
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "openai_parse"}, status_code=502)
    client_secret = (data.get("client_secret") or {}).get("value")
    if not client_secret:
        return JSONResponse({"ok": False, "error": "no_ephemeral"}, status_code=502)
    return JSONResponse(
        {
            "ok": True,
            "client_secret": client_secret,
            "expires_at": (data.get("client_secret") or {}).get("expires_at"),
            "model": model,
            "voice": voice,
        }
    )


@app.post("/api/web/typed")
async def typed_turn(request: Request):
    if not OPENAI_API_KEY:
        return JSONResponse(
            {"ok": False, "error": "openai_key_missing"}, status_code=503
        )
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    user_text = str(body.get("text") or "").strip()
    if not user_text:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
    voice = (body.get("voice") or DEFAULT_TTS_VOICE).strip()
    llm_model = os.environ.get("OPENAI_LLM_MODEL", DEFAULT_LLM_MODEL)
    tts_model = os.environ.get("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)

    reply_text = await _chat_completion(llm_model, user_text)
    if isinstance(reply_text, JSONResponse):
        return reply_text
    audio_b64_mime = await _tts_speak(tts_model, voice, reply_text)
    if isinstance(audio_b64_mime, JSONResponse):
        return audio_b64_mime
    audio_b64, mime = audio_b64_mime
    return JSONResponse(
        {"ok": True, "text": reply_text, "audio_b64": audio_b64, "mime": mime}
    )


async def _chat_completion(model: str, user_text: str):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.8,
    }
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(payload),
            )
    except httpx.HTTPError as exc:
        log.warning("chat.completions transport error: %s", exc)
        return JSONResponse({"ok": False, "error": "llm_transport"}, status_code=502)
    if resp.status_code >= 400:
        log.warning("chat.completions failed: %s %s", resp.status_code, resp.text[:300])
        return JSONResponse({"ok": False, "error": "llm_error"}, status_code=502)
    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        log.exception("bad chat completion shape: %s", exc)
        return JSONResponse({"ok": False, "error": "llm_parse"}, status_code=502)


async def _tts_speak(model: str, voice: str, text: str):
    import base64

    payload = {"model": model, "voice": voice, "input": text, "format": "mp3"}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(payload),
            )
    except httpx.HTTPError as exc:
        log.warning("tts transport error: %s", exc)
        return JSONResponse({"ok": False, "error": "tts_transport"}, status_code=502)
    if resp.status_code >= 400:
        log.warning("tts failed: %s %s", resp.status_code, resp.text[:300])
        return JSONResponse({"ok": False, "error": "tts_error"}, status_code=502)
    return base64.b64encode(resp.content).decode("ascii"), "audio/mpeg"


# --------------------------------------------------------------------
# Test hook: reset the in-memory limiter between tests. Not part of
# the public API; safe on serverless where each invocation starts
# fresh anyway.
# --------------------------------------------------------------------

def _reset_for_tests(*, api_key: Optional[str] = None, allowed_origin: Optional[str] = None) -> None:
    global LOGIN_LIMITER, OPENAI_API_KEY, AUTH_CONFIG
    LOGIN_LIMITER = LoginLimiter()
    if api_key is not None:
        OPENAI_API_KEY = api_key
    if allowed_origin is not None:
        AUTH_CONFIG = AuthConfig(
            password_hash=AUTH_CONFIG.password_hash,
            dev_plain_password=AUTH_CONFIG.dev_plain_password,
            session_secret=AUTH_CONFIG.session_secret,
            allowed_origin=allowed_origin or None,
            secure_cookie=AUTH_CONFIG.secure_cookie,
            session_ttl_s=AUTH_CONFIG.session_ttl_s,
        )

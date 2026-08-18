"""Vercel entrypoint — FastAPI version of the hosted browser app.

Mirrors ``app/web_app.py`` (the aiohttp server used by
Docker/Fly/Render) but built on FastAPI so Vercel's Python runtime
can invoke it directly. Both builds share the same pure auth
primitives from :mod:`app.auth_core`, so password hashing, session
cookies, rate limiting, and Origin checks are byte-for-byte identical
across deploy targets.

Routing:
    Vercel FastAPI compiles this module into a function at ``/api``.
    ``vercel.json`` routes every browser URL there and a ``request.path``
    transform restores ``/login``, ``/relic``, etc. before FastAPI routes.

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
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
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
from app.motion_config import load_motion_config
from app.personality import load_personality

log = logging.getLogger("maxwell.vercel")
logging.basicConfig(level=logging.INFO)

# ---- Config (shared across deploy targets) ----------------------------

DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_REALTIME_VOICE = "ballad"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "ballad"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


# Pulled from config.yaml (same file the local CLI reads). Makes the
# Vercel build use the full Stanford TEA + admit-weekend-fair
# personality instead of the compact placeholder we used to ship —
# so Maxwell actually knows about the club, the meeting time, the
# annual LA trip, alumni, etc. Falls back to a built-in short prompt
# only if config.yaml is somehow unbundled.
SYSTEM_PROMPT = load_personality(ROOT)
log.info(
    "personality loaded (%d chars, source=%s)",
    len(SYSTEM_PROMPT),
    "config.yaml" if (ROOT / "config.yaml").is_file() else "fallback",
)

# Per-session extra context (the event Maxwell is at, or an operator's
# custom prompt). Appended to the base personality so the club/TEA
# personality is always preserved. Capped so a malicious client can't
# blow up the instruction size or smuggle in a giant prompt.
MAX_CONTEXT_CHARS = 2000


def compose_instructions(base: str, context: str | None) -> str:
    """Return the base personality with an optional event/custom
    context appended. Keeps the personality intact (we append rather
    than replace) and bounds the context length."""
    ctx = (context or "").strip()
    if not ctx:
        return base
    ctx = ctx[:MAX_CONTEXT_CHARS]
    return (
        f"{base}\n\n"
        "## Right now\n"
        f"{ctx}\n"
        "Weave this into how you greet and talk with people when relevant, "
        "but stay in character as Maxwell."
    )

STATIC_WEB_ROOT = ROOT / "static" / "web"
STATIC_DIR = ROOT / "static"

AUTH_CONFIG = AuthConfig.from_env()
LOGIN_LIMITER = LoginLimiter()
OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY") or None
if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY is not set; OpenAI endpoints will 503.")

# ---- Song lip-sync ("jukebox") config ---------------------------------
# Spotify is used for search only (rich metadata + album art). The
# secret stays server-side exactly like OPENAI_API_KEY — the browser
# never sees it. App tokens are minted with the client-credentials
# flow (no user OAuth) and cached until they expire.
SPOTIFY_CLIENT_ID: Optional[str] = os.environ.get("SPOTIFY_CLIENT_ID") or None
SPOTIFY_CLIENT_SECRET: Optional[str] = os.environ.get("SPOTIFY_CLIENT_SECRET") or None
if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
    log.info("SPOTIFY_CLIENT_ID/SECRET not set; /api/web/song/search will 503.")

# Only ever proxy 30s preview clips from these known-good CDNs. This
# is both an anti-SSRF allowlist and a guard against turning the booth
# into an open proxy. Spotify previews live on *.scdn.co; Apple/iTunes
# previews live on *.itunes.apple.com / *.mzstatic.com.
SONG_AUDIO_HOST_SUFFIXES = ("scdn.co", "mzstatic.com", "itunes.apple.com")
# 30s AAC/MP3 previews are ~0.5-1.5 MB; cap well above that but bounded
# so a redirect to something huge can't blow up the function.
MAX_SONG_AUDIO_BYTES = 12 * 1024 * 1024

# Cached Spotify app token: {"value": str|None, "expires_at": epoch}.
_spotify_token_cache: dict[str, Any] = {"value": None, "expires_at": 0.0}


def _song_host_allowed(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return any(
        host == suffix or host.endswith("." + suffix)
        for suffix in SONG_AUDIO_HOST_SUFFIXES
    )


async def _spotify_token() -> Optional[str]:
    """Return a cached Spotify app token, minting a new one if needed.

    Uses the client-credentials grant (server-to-server, no user login).
    Returns ``None`` when credentials are unset or minting fails so the
    caller can surface a clean 503 instead of crashing.
    """
    import base64 as _b64

    now = time.time()
    cached = _spotify_token_cache.get("value")
    if cached and float(_spotify_token_cache.get("expires_at", 0)) > now + 5:
        return cached
    if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
        return None
    basic = _b64.b64encode(
        f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()
    ).decode()
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.post(
                "https://accounts.spotify.com/api/token",
                headers={
                    "Authorization": f"Basic {basic}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                content="grant_type=client_credentials",
            )
    except httpx.HTTPError as exc:
        log.warning("spotify token transport error: %s", exc)
        return None
    if resp.status_code >= 400:
        log.warning("spotify token failed: %s %s", resp.status_code, resp.text[:200])
        return None
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None
    token = data.get("access_token")
    ttl = float(data.get("expires_in", 3600) or 3600)
    if token:
        _spotify_token_cache["value"] = token
        _spotify_token_cache["expires_at"] = now + ttl
    return token


async def _itunes_preview(client: "httpx.AsyncClient", term: str) -> Optional[str]:
    """Resolve a 30s preview URL for ``term`` via the tokenless iTunes
    Search API. Used to fill in previews for tracks where Spotify
    returns ``preview_url: null`` (common since Spotify trimmed preview
    coverage)."""
    if not term:
        return None
    try:
        resp = await client.get(
            "https://itunes.apple.com/search",
            params={"term": term, "media": "music", "entity": "song", "limit": 1},
        )
    except httpx.HTTPError:
        return None
    if resp.status_code >= 400:
        return None
    try:
        results = resp.json().get("results") or []
    except Exception:  # noqa: BLE001
        return None
    if not results:
        return None
    return results[0].get("previewUrl")


def _spotify_album_art(track: dict) -> Optional[str]:
    images = ((track.get("album") or {}).get("images")) or []
    if not images:
        return None
    # Spotify returns images largest-first; prefer a ~300px middle size
    # so result cards stay light.
    return (images[1] if len(images) >= 2 else images[0]).get("url")


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
            or path == "/api"
            or path == "/index"
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
    # Subclass StaticFiles so JS/CSS files never get stuck in stale
    # browser/CDN caches. We change these files on most deploys and
    # the symptom of a cached bundle is genuinely confusing
    # ("realtime is broken!" when in fact the new build fixed it).
    class _NoStoreStaticFiles(StaticFiles):
        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            if ext in {"js", "mjs", "css", "html"}:
                resp.headers["Cache-Control"] = "no-cache, must-revalidate"
            return resp

    app.mount("/static", _NoStoreStaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    log.warning("static dir missing: %s", STATIC_DIR)


# --------------------------------------------------------------------
# Page routes
# --------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/api", include_in_schema=False)
@app.get("/index", include_in_schema=False)
async def vercel_path_probe(request: Request) -> JSONResponse:
    # Temporary: if Vercel still forwards every URL as /api or /index,
    # this is what the browser will receive. Remove once routing is stable.
    x_headers = {
        k: v for k, v in request.headers.items()
        if k.lower().startswith("x-") or k.lower() in {"host", "forwarded", "url"}
    }
    return JSONResponse({
        "ok": True,
        "path": request.url.path,
        "scope_path": request.scope.get("path"),
        "root_path": request.scope.get("root_path"),
        "url": str(request.url),
        "headers": x_headers,
    })


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _current_session(request) is not None:
        return RedirectResponse(url="/", status_code=302)
    return _serve_html("login.html")


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return _serve_html("index.html")


@app.get("/admits", response_class=HTMLResponse)
async def admits_page(request: Request):
    # End-user-facing cartoon UI. Same auth gate as the operator page;
    # a booth host logs in once and hands the laptop to guests.
    return _serve_html("admits.html")


@app.get("/sing", response_class=HTMLResponse)
async def sing_page(request: Request):
    # Song lip-sync "jukebox" — the booth's headline experience.
    return _serve_html("sing.html")


@app.get("/relic", response_class=HTMLResponse)
async def relic_page(request: Request):
    # Artifact prop lighting control — Web Serial to Sparkle Motion Mini.
    return _serve_html("relic.html")


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


@app.get("/api/web/motion-config")
async def web_motion_config():
    # Pins, PWM ranges, behavior gains, and jaw calibration come from
    # config.yaml so tuning the operator build automatically flows to
    # the browser. Falls back to DEFAULT_* if config.yaml is missing.
    try:
        data = load_motion_config(ROOT)
    except Exception as exc:  # noqa: BLE001
        log.exception("motion_config load failed: %s", exc)
        return JSONResponse(
            {"ok": False, "error": "motion_config_load_failed"}, status_code=500
        )
    return JSONResponse({"ok": True, **data})


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
    instructions = compose_instructions(SYSTEM_PROMPT, body.get("context"))

    # OpenAI deprecated /v1/realtime/sessions in May 2026. The GA flow
    # is now POST /v1/realtime/client_secrets with the session config
    # nested, voice under session.audio.output, and no OpenAI-Beta
    # header. Token comes back as the top-level `value` field.
    payload = {
        "expires_after": {"anchor": "created_at", "seconds": 600},
        "session": {
            "type": "realtime",
            "model": model,
            "instructions": instructions,
            "audio": {
                "input": {
                    "transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.65,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 650,
                    },
                },
                "output": {"voice": voice},
            },
        },
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.openai.com/v1/realtime/client_secrets",
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
        # Surface the OpenAI error body to the browser log so the
        # operator can actually see what failed (auth, deprecated
        # endpoint, model not enabled, etc.) instead of an opaque
        # "openai_error" string.
        detail = resp.text[:400] if resp.text else ""
        try:
            j = resp.json()
            if isinstance(j, dict) and "error" in j and isinstance(j["error"], dict):
                msg = j["error"].get("message") or ""
                code = j["error"].get("code") or j["error"].get("type") or ""
                if msg:
                    detail = f"{code}: {msg}" if code else msg
        except Exception:  # noqa: BLE001
            pass
        return JSONResponse(
            {
                "ok": False,
                "error": "openai_error",
                "status": resp.status_code,
                "detail": detail,
            },
            status_code=502,
        )
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "openai_parse"}, status_code=502)
    # GA shape: { "value": "ek_...", "expires_at": ..., "session": {...} }
    # Pre-GA shape (legacy /sessions): { "client_secret": {"value": "ek_..."} }
    # Tolerate both so an account that hasn't fully rolled to GA still works.
    client_secret = data.get("value")
    expires_at = data.get("expires_at")
    if not client_secret:
        legacy = data.get("client_secret")
        if isinstance(legacy, dict):
            client_secret = legacy.get("value")
            expires_at = legacy.get("expires_at") or expires_at
        elif isinstance(legacy, str):
            client_secret = legacy
    if not client_secret:
        return JSONResponse({"ok": False, "error": "no_ephemeral"}, status_code=502)
    return JSONResponse(
        {
            "ok": True,
            "client_secret": client_secret,
            "expires_at": expires_at,
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
    instructions = compose_instructions(SYSTEM_PROMPT, body.get("context"))

    reply_text = await _chat_completion(llm_model, user_text, instructions)
    if isinstance(reply_text, JSONResponse):
        return reply_text
    audio_b64_mime = await _tts_speak(tts_model, voice, reply_text)
    if isinstance(audio_b64_mime, JSONResponse):
        return audio_b64_mime
    audio_b64, mime = audio_b64_mime
    return JSONResponse(
        {"ok": True, "text": reply_text, "audio_b64": audio_b64, "mime": mime}
    )


@app.post("/api/web/tts")
async def tts_turn(request: Request):
    """Direct text-to-speech: speak the supplied text verbatim (no LLM).

    Used by the sing page's "make Maxwell talk" field so the operator can
    put exact words in his mouth. Returns the same shape as
    ``/api/web/typed`` (``text`` echoes the input) so the browser
    playback path is identical.
    """
    if not OPENAI_API_KEY:
        return JSONResponse(
            {"ok": False, "error": "openai_key_missing"}, status_code=503
        )
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    text = str(body.get("text") or "").strip()[:600]
    if not text:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
    voice = (body.get("voice") or DEFAULT_TTS_VOICE).strip()
    tts_model = os.environ.get("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)

    audio_b64_mime = await _tts_speak(tts_model, voice, text)
    if isinstance(audio_b64_mime, JSONResponse):
        return audio_b64_mime
    audio_b64, mime = audio_b64_mime
    return JSONResponse(
        {"ok": True, "text": text, "audio_b64": audio_b64, "mime": mime}
    )


# --------------------------------------------------------------------
# Song lip-sync ("jukebox") endpoints
# --------------------------------------------------------------------

@app.get("/api/web/song/search")
async def song_search(request: Request, q: str = "", limit: int = 8):
    query = (q or "").strip()[:120]
    if not query:
        return JSONResponse({"ok": False, "error": "empty_query"}, status_code=400)
    token = await _spotify_token()
    if not token:
        return JSONResponse(
            {
                "ok": False,
                "error": "spotify_not_configured",
                "detail": "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.",
            },
            status_code=503,
        )
    n = max(1, min(int(limit or 8), 12))
    async with httpx.AsyncClient(timeout=12) as client:
        try:
            resp = await client.get(
                "https://api.spotify.com/v1/search",
                headers={"Authorization": f"Bearer {token}"},
                params={"q": query, "type": "track", "limit": n},
            )
        except httpx.HTTPError as exc:
            log.warning("spotify search transport error: %s", exc)
            return JSONResponse({"ok": False, "error": "spotify_transport"}, status_code=502)
        if resp.status_code >= 400:
            log.warning("spotify search failed: %s %s", resp.status_code, resp.text[:200])
            return JSONResponse(
                {"ok": False, "error": "spotify_error", "status": resp.status_code},
                status_code=502,
            )
        try:
            items = (resp.json().get("tracks") or {}).get("items") or []
        except Exception:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": "spotify_parse"}, status_code=502)

        tracks = []
        for it in items:
            tracks.append(
                {
                    "id": it.get("id"),
                    "title": it.get("name"),
                    "artist": ", ".join(
                        a.get("name") for a in (it.get("artists") or []) if a.get("name")
                    ),
                    "art": _spotify_album_art(it),
                    "preview_url": it.get("preview_url"),
                    "duration_ms": it.get("duration_ms"),
                }
            )

        # Spotify increasingly returns preview_url: null. Fall back to
        # the tokenless iTunes preview for those, concurrently so the
        # whole search still resolves in well under the function budget.
        missing = [t for t in tracks if not t.get("preview_url")]
        if missing:
            async def _fill(t: dict) -> None:
                term = f"{t.get('artist') or ''} {t.get('title') or ''}".strip()
                t["preview_url"] = await _itunes_preview(client, term)

            await asyncio.gather(*[_fill(t) for t in missing], return_exceptions=True)

    # Only return tracks we can actually play + analyze.
    playable = [t for t in tracks if t.get("preview_url")]
    return JSONResponse({"ok": True, "tracks": playable})


@app.get("/api/web/song/audio")
async def song_audio(request: Request, url: str = ""):
    target = (url or "").strip()
    if not target or not _song_host_allowed(target):
        return JSONResponse({"ok": False, "error": "bad_url"}, status_code=400)

    client = httpx.AsyncClient(timeout=httpx.Timeout(20.0), follow_redirects=True)
    try:
        upstream = await client.send(client.build_request("GET", target), stream=True)
    except httpx.HTTPError as exc:
        log.warning("song audio transport error: %s", exc)
        await client.aclose()
        return JSONResponse({"ok": False, "error": "audio_transport"}, status_code=502)
    if upstream.status_code >= 400:
        await upstream.aclose()
        await client.aclose()
        return JSONResponse(
            {"ok": False, "error": "audio_error", "status": upstream.status_code},
            status_code=502,
        )
    media_type = upstream.headers.get("content-type", "audio/mpeg")

    async def _gen():
        sent = 0
        try:
            async for chunk in upstream.aiter_bytes(65536):
                sent += len(chunk)
                if sent > MAX_SONG_AUDIO_BYTES:
                    break
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        _gen(),
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )


async def _chat_completion(model: str, user_text: str, instructions: str | None = None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions or SYSTEM_PROMPT},
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

def _reset_for_tests(
    *,
    api_key: Optional[str] = None,
    allowed_origin: Optional[str] = None,
    spotify_id: Optional[str] = None,
    spotify_secret: Optional[str] = None,
) -> None:
    global LOGIN_LIMITER, OPENAI_API_KEY, AUTH_CONFIG
    global SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
    LOGIN_LIMITER = LoginLimiter()
    _spotify_token_cache["value"] = None
    _spotify_token_cache["expires_at"] = 0.0
    if spotify_id is not None:
        SPOTIFY_CLIENT_ID = spotify_id or None
    if spotify_secret is not None:
        SPOTIFY_CLIENT_SECRET = spotify_secret or None
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

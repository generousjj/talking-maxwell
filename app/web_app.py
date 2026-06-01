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

from urllib.parse import urlparse

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

DEFAULT_REALTIME_MODEL = "gpt-realtime"
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

# Per-session extra context (event Maxwell is at / custom prompt),
# appended to the base personality. Bounded so a client can't blow
# up the instruction size. Mirrors api/index.py.
MAX_CONTEXT_CHARS = 2000


def compose_instructions(base: str, context: Optional[str]) -> str:
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

STATIC_WEB_SUBDIR = "web"

# ---- Song lip-sync ("jukebox") config (mirrors api/index.py) ----------
# Spotify is search-only; the secret stays server-side. App tokens use
# the client-credentials flow and are cached until expiry. Audio is only
# proxied from known preview CDNs (anti-SSRF + not an open proxy).
SONG_AUDIO_HOST_SUFFIXES = ("scdn.co", "mzstatic.com", "itunes.apple.com")
MAX_SONG_AUDIO_BYTES = 12 * 1024 * 1024
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


async def _spotify_token(client_id: Optional[str], client_secret: Optional[str]) -> Optional[str]:
    """Cached Spotify app token via client-credentials. Returns None if
    creds are unset or minting fails."""
    import base64
    import time

    now = time.time()
    cached = _spotify_token_cache.get("value")
    if cached and float(_spotify_token_cache.get("expires_at", 0)) > now + 5:
        return cached
    if not (client_id and client_secret):
        return None
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    timeout = aiohttp.ClientTimeout(total=12)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://accounts.spotify.com/api/token",
                headers={
                    "Authorization": f"Basic {basic}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data="grant_type=client_credentials",
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    log.warning("spotify token failed: %s %s", resp.status, body[:200])
                    return None
                data = await resp.json()
    except aiohttp.ClientError as exc:
        log.warning("spotify token transport error: %s", exc)
        return None
    token = data.get("access_token")
    ttl = float(data.get("expires_in", 3600) or 3600)
    if token:
        _spotify_token_cache["value"] = token
        _spotify_token_cache["expires_at"] = now + ttl
    return token


async def _itunes_preview(session: "aiohttp.ClientSession", term: str) -> Optional[str]:
    """Tokenless iTunes preview fallback for tracks Spotify returns with
    preview_url: null."""
    if not term:
        return None
    try:
        async with session.get(
            "https://itunes.apple.com/search",
            params={"term": term, "media": "music", "entity": "song", "limit": 1},
        ) as resp:
            if resp.status >= 400:
                return None
            data = await resp.json(content_type=None)
    except aiohttp.ClientError:
        return None
    except Exception:  # noqa: BLE001
        return None
    results = data.get("results") or []
    if not results:
        return None
    return results[0].get("previewUrl")


def _spotify_album_art(track: dict) -> Optional[str]:
    images = ((track.get("album") or {}).get("images")) or []
    if not images:
        return None
    return (images[1] if len(images) >= 2 else images[0]).get("url")


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


async def handle_sing(request: web.Request) -> web.Response:
    # Song lip-sync "jukebox" page — the booth's headline experience.
    # Same session gate as the other pages.
    return _send_static_file(request, "sing.html")


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
    instructions = compose_instructions(SYSTEM_PROMPT, body.get("context"))

    # GA Realtime API (May 2026+): POST /v1/realtime/client_secrets,
    # session config nested, voice under session.audio.output, and no
    # OpenAI-Beta header. The pre-GA /v1/realtime/sessions endpoint
    # returns HTTP 400 "Invalid URL" after the May 2026 deprecation.
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
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "https://api.openai.com/v1/realtime/client_secrets",
            headers=headers,
            data=json.dumps(payload),
        ) as resp:
            raw = await resp.text()
            if resp.status >= 400:
                log.warning(
                    "OpenAI realtime session failed: %s %s", resp.status, raw[:400]
                )
                detail = raw[:400] if raw else ""
                try:
                    j = json.loads(raw)
                    if isinstance(j, dict) and isinstance(j.get("error"), dict):
                        msg = j["error"].get("message") or ""
                        code = j["error"].get("code") or j["error"].get("type") or ""
                        if msg:
                            detail = f"{code}: {msg}" if code else msg
                except Exception:  # noqa: BLE001
                    pass
                return web.json_response(
                    {
                        "ok": False,
                        "error": "openai_error",
                        "status": resp.status,
                        "detail": detail,
                    },
                    status=502,
                )
            try:
                data = json.loads(raw)
            except Exception:  # noqa: BLE001
                return web.json_response(
                    {"ok": False, "error": "openai_parse"}, status=502
                )

    # GA shape puts token at top-level `value`; legacy /sessions nested
    # under client_secret.value. Accept both for resilience.
    client_secret = data.get("value")
    if not client_secret:
        legacy = data.get("client_secret")
        if isinstance(legacy, dict):
            client_secret = legacy.get("value")
        elif isinstance(legacy, str):
            client_secret = legacy
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
    instructions = compose_instructions(SYSTEM_PROMPT, body.get("context"))

    reply_text = await _chat_completion(api_key, llm_model, user_text, instructions)
    audio_b64, mime = await _tts_speak(api_key, tts_model, voice, reply_text)
    return web.json_response(
        {
            "ok": True,
            "text": reply_text,
            "audio_b64": audio_b64,
            "mime": mime,
        }
    )


async def handle_tts_turn(request: web.Request) -> web.Response:
    """Direct text-to-speech: speak the supplied text verbatim (no LLM).

    Used by the sing page so the operator can put exact words in
    Maxwell's mouth. Returns the same ``{ok, text, audio_b64, mime}``
    shape as ``/api/web/typed`` (``text`` echoes the input).
    """
    api_key = _require_openai_key(request)
    if isinstance(api_key, web.Response):
        return api_key
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    text = str(body.get("text") or "").strip()[:600]
    if not text:
        return web.json_response({"ok": False, "error": "empty"}, status=400)
    voice = (body.get("voice") or DEFAULT_TTS_VOICE).strip()
    tts_model = os.environ.get("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)

    audio_b64, mime = await _tts_speak(api_key, tts_model, voice, text)
    return web.json_response(
        {"ok": True, "text": text, "audio_b64": audio_b64, "mime": mime}
    )


async def _chat_completion(
    api_key: str, model: str, user_text: str, instructions: Optional[str] = None
) -> str:
    timeout = aiohttp.ClientTimeout(total=25)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions or SYSTEM_PROMPT},
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
# Song lip-sync ("jukebox") endpoints
# --------------------------------------------------------------------

async def handle_song_search(request: web.Request) -> web.Response:
    query = (request.query.get("q") or "").strip()[:120]
    if not query:
        return web.json_response({"ok": False, "error": "empty_query"}, status=400)
    cid = request.app.get("spotify_client_id")
    csecret = request.app.get("spotify_client_secret")
    token = await _spotify_token(cid, csecret)
    if not token:
        return web.json_response(
            {
                "ok": False,
                "error": "spotify_not_configured",
                "detail": "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.",
            },
            status=503,
        )
    try:
        n = max(1, min(int(request.query.get("limit") or 8), 12))
    except ValueError:
        n = 8

    timeout = aiohttp.ClientTimeout(total=12)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(
                "https://api.spotify.com/v1/search",
                headers={"Authorization": f"Bearer {token}"},
                params={"q": query, "type": "track", "limit": n},
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    log.warning("spotify search failed: %s %s", resp.status, body[:200])
                    return web.json_response(
                        {"ok": False, "error": "spotify_error", "status": resp.status},
                        status=502,
                    )
                data = await resp.json(content_type=None)
        except aiohttp.ClientError as exc:
            log.warning("spotify search transport error: %s", exc)
            return web.json_response(
                {"ok": False, "error": "spotify_transport"}, status=502
            )

        items = (data.get("tracks") or {}).get("items") or []
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

        missing = [t for t in tracks if not t.get("preview_url")]
        if missing:
            async def _fill(t: dict) -> None:
                term = f"{t.get('artist') or ''} {t.get('title') or ''}".strip()
                t["preview_url"] = await _itunes_preview(session, term)

            await asyncio.gather(*[_fill(t) for t in missing], return_exceptions=True)

    playable = [t for t in tracks if t.get("preview_url")]
    return web.json_response({"ok": True, "tracks": playable})


async def handle_song_audio(request: web.Request) -> web.StreamResponse:
    target = (request.query.get("url") or "").strip()
    if not target or not _song_host_allowed(target):
        return web.json_response({"ok": False, "error": "bad_url"}, status=400)

    timeout = aiohttp.ClientTimeout(total=20)
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        upstream = await session.get(target, allow_redirects=True)
    except aiohttp.ClientError as exc:
        log.warning("song audio transport error: %s", exc)
        await session.close()
        return web.json_response({"ok": False, "error": "audio_transport"}, status=502)
    if upstream.status >= 400:
        upstream.release()
        await session.close()
        return web.json_response(
            {"ok": False, "error": "audio_error", "status": upstream.status}, status=502
        )

    media_type = upstream.headers.get("Content-Type", "audio/mpeg")
    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": media_type, "Cache-Control": "no-store"},
    )
    await resp.prepare(request)
    sent = 0
    try:
        async for chunk in upstream.content.iter_chunked(65536):
            sent += len(chunk)
            if sent > MAX_SONG_AUDIO_BYTES:
                break
            await resp.write(chunk)
    finally:
        upstream.release()
        await session.close()
    await resp.write_eof()
    return resp


# --------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------

def build_app(
    *,
    auth_config: AuthConfig,
    openai_api_key: Optional[str],
    realtime_voice: str = DEFAULT_REALTIME_VOICE,
    realtime_model: str = DEFAULT_REALTIME_MODEL,
    spotify_client_id: Optional[str] = None,
    spotify_client_secret: Optional[str] = None,
) -> web.Application:
    app = web.Application(middlewares=[auth_middleware])
    app["auth_config"] = auth_config
    app["login_limiter"] = LoginLimiter()
    app["openai_api_key"] = openai_api_key
    app["realtime_voice"] = realtime_voice
    app["realtime_model"] = realtime_model
    app["spotify_client_id"] = spotify_client_id
    app["spotify_client_secret"] = spotify_client_secret

    app.router.add_get("/healthz", handle_health)
    app.router.add_get("/login", handle_login_page)
    app.router.add_get("/", handle_index)
    app.router.add_get("/admits", handle_admits)
    app.router.add_get("/sing", handle_sing)

    app.router.add_post("/api/auth/login", handle_login)
    app.router.add_post("/api/auth/logout", handle_logout)
    app.router.add_get("/api/auth/me", handle_me)

    app.router.add_post("/api/web/realtime/session", handle_realtime_session)
    app.router.add_post("/api/web/typed", handle_typed_turn)
    app.router.add_post("/api/web/tts", handle_tts_turn)
    app.router.add_get("/api/web/song/search", handle_song_search)
    app.router.add_get("/api/web/song/audio", handle_song_audio)

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
    spotify_client_id = os.environ.get("SPOTIFY_CLIENT_ID") or None
    spotify_client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET") or None
    if not (spotify_client_id and spotify_client_secret):
        log.info("SPOTIFY_CLIENT_ID/SECRET not set; song search will return 503.")

    app = build_app(
        auth_config=auth_config,
        openai_api_key=openai_api_key,
        realtime_voice=args.realtime_voice,
        realtime_model=args.realtime_model,
        spotify_client_id=spotify_client_id,
        spotify_client_secret=spotify_client_secret,
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

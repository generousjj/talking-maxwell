"""Integration tests for the hosted browser-mode aiohttp app.

We intentionally avoid pytest-aiohttp / pytest-asyncio so the test
suite keeps running on the stock ``pytest`` already in
``requirements.txt``. Each async test is wrapped in ``asyncio.run``
and boots an ephemeral ``TestServer`` around ``web_app.build_app``.

Covered:
  - unauthenticated requests are redirected (HTML) or 401 (JSON)
  - login + me + logout flow works end-to-end
  - bad origins are rejected on state-changing calls
  - /api/web/realtime/session returns only the ephemeral token
    (never the raw API key) and surfaces OpenAI errors cleanly
  - static frontend files exist
  - existing local ``app.webapp`` still imports
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import json
import sys
from pathlib import Path
from typing import Callable

import pytest
from aiohttp.test_utils import TestClient, TestServer

from app import web_app, web_auth

PASSWORD = "correcthorsebatterystaple"
RAW_API_KEY = "sk-test-NEVER-LEAK-ME-1234567890"
EPHEMERAL_TOKEN = "ek_test_shortlived_0001"


def _auth_config() -> web_auth.AuthConfig:
    return web_auth.AuthConfig(
        password_hash=web_auth.hash_password(PASSWORD, iterations=10_000),
        dev_plain_password=None,
        session_secret=b"session-secret-bytes-0123456789abc",
        allowed_origin=None,
        secure_cookie=False,
        session_ttl_s=300,
    )


def _build_client_factory(openai_api_key: str | None):
    async def factory() -> tuple[TestClient, Callable[[], "asyncio.Future[None]"]]:
        app = web_app.build_app(
            auth_config=_auth_config(),
            openai_api_key=openai_api_key,
        )
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        return client, client.close
    return factory


def _async(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        return asyncio.run(fn(*a, **kw))
    return wrapper


# ---- tests ----

@_async
async def test_health_is_public():
    client, close = await _build_client_factory(RAW_API_KEY)()
    try:
        resp = await client.get("/healthz")
        assert resp.status == 200
        assert (await resp.json())["ok"] is True
    finally:
        await close()


@_async
async def test_index_redirects_when_unauthenticated():
    client, close = await _build_client_factory(RAW_API_KEY)()
    try:
        resp = await client.get("/", allow_redirects=False)
        assert resp.status == 302
        assert resp.headers["Location"] == "/login"
    finally:
        await close()


@_async
async def test_admits_page_requires_auth_and_renders():
    client, close = await _build_client_factory(RAW_API_KEY)()
    try:
        resp = await client.get("/admits", allow_redirects=False)
        assert resp.status == 302
        assert resp.headers["Location"] == "/login"

        await client.post("/api/auth/login", json={"password": PASSWORD})
        page = await client.get("/admits")
        assert page.status == 200
        body = await page.text()
        assert "Talk to Maxwell" in body
        assert "/static/web/js/admits.js" in body
    finally:
        await close()


@_async
async def test_api_returns_401_when_unauthenticated():
    client, close = await _build_client_factory(RAW_API_KEY)()
    try:
        resp = await client.post("/api/web/realtime/session", json={})
        assert resp.status == 401
        body = await resp.json()
        assert body["error"] == "unauthorized"
    finally:
        await close()


@_async
async def test_login_flow_and_me():
    client, close = await _build_client_factory(RAW_API_KEY)()
    try:
        resp = await client.post("/api/auth/login", json={"password": "nope"})
        assert resp.status == 401

        resp = await client.post("/api/auth/login", json={"password": PASSWORD})
        assert resp.status == 200
        assert (await resp.json())["ok"] is True

        me = await client.get("/api/auth/me")
        body = await me.json()
        assert body["authed"] is True
        assert body["uid"] == "operator"

        logout = await client.post("/api/auth/logout")
        assert logout.status == 200

        me_after = await client.get("/api/auth/me")
        assert (await me_after.json())["authed"] is False
    finally:
        await close()


@_async
async def test_origin_check_rejects_cross_site():
    client, close = await _build_client_factory(RAW_API_KEY)()
    try:
        await client.post("/api/auth/login", json={"password": PASSWORD})
        resp = await client.post(
            "/api/web/realtime/session",
            json={},
            headers={"Origin": "https://evil.example.com"},
        )
        assert resp.status == 403
        assert (await resp.json())["error"] == "forbidden_origin"
    finally:
        await close()


@_async
async def test_realtime_session_hides_raw_api_key(monkeypatch_env=None):
    # Patch web_app.aiohttp.ClientSession for the duration of this test.
    captured: dict = {}

    class FakeCtx:
        def __init__(self, status, payload):
            self._status = status
            self._payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        @property
        def status(self):
            return self._status

        async def text(self):
            return json.dumps(self._payload)

    class FakeSession:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        def post(self, url, *, headers=None, data=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["data"] = data
            # GA Realtime: response shape has `value` at top level.
            return FakeCtx(
                200,
                {
                    "value": EPHEMERAL_TOKEN,
                    "expires_at": 1234567890,
                    "session": {"id": "sess_test", "type": "realtime"},
                },
            )

    real_client_session = web_app.aiohttp.ClientSession
    web_app.aiohttp.ClientSession = FakeSession
    client, close = await _build_client_factory(RAW_API_KEY)()
    try:
        await client.post("/api/auth/login", json={"password": PASSWORD})
        resp = await client.post("/api/web/realtime/session", json={"voice": "ballad"})
        assert resp.status == 200
        body = await resp.json()
        assert body["client_secret"] == EPHEMERAL_TOKEN
        assert captured["url"].endswith("/v1/realtime/client_secrets")
        assert "OpenAI-Beta" not in (captured["headers"] or {})
        assert RAW_API_KEY in captured["headers"]["Authorization"]
        assert RAW_API_KEY not in json.dumps(body)
    finally:
        await close()
        web_app.aiohttp.ClientSession = real_client_session


@_async
async def test_realtime_session_returns_503_when_key_missing():
    client, close = await _build_client_factory(None)()
    try:
        await client.post("/api/auth/login", json={"password": PASSWORD})
        resp = await client.post("/api/web/realtime/session", json={})
        assert resp.status == 503
        assert (await resp.json())["error"] == "openai_key_missing"
    finally:
        await close()


SPOTIFY_ID = "spotify-client-id"
SPOTIFY_SECRET = "spotify-client-secret-NEVER-LEAK"


def _build_song_client_factory(spotify_id, spotify_secret):
    async def factory():
        app = web_app.build_app(
            auth_config=_auth_config(),
            openai_api_key=RAW_API_KEY,
            spotify_client_id=spotify_id,
            spotify_client_secret=spotify_secret,
        )
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        return client, client.close
    return factory


class _FakeReqCtx:
    """Mimics aiohttp's request context manager, which is both awaitable
    (``await session.get(...)``) and an async context manager
    (``async with session.get(...) as resp``)."""

    def __init__(self, resp):
        self._resp = resp

    def __await__(self):
        async def _coro():
            return self._resp
        return _coro().__await__()

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *a):
        return None


class _FakeContent:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_chunked(self, n):
        for chunk in self._chunks:
            yield chunk


class _FakeResp:
    def __init__(self, status, payload=None, *, chunks=None, headers=None):
        self.status = status
        self._payload = payload
        self._chunks = chunks or []
        self.headers = headers or {}

    async def json(self, content_type=None):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)

    def release(self):
        return None

    @property
    def content(self):
        return _FakeContent(self._chunks)


def _reset_song_cache():
    web_app._spotify_token_cache["value"] = None
    web_app._spotify_token_cache["expires_at"] = 0.0


@_async
async def test_song_search_503_without_credentials():
    _reset_song_cache()
    client, close = await _build_song_client_factory("", "")()
    try:
        await client.post("/api/auth/login", json={"password": PASSWORD})
        resp = await client.get("/api/web/song/search?q=test")
        assert resp.status == 503
        assert (await resp.json())["error"] == "spotify_not_configured"
    finally:
        await close()


@_async
async def test_song_search_spotify_with_itunes_fallback():
    _reset_song_cache()
    calls = {"itunes": 0}

    class FakeSession:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def close(self):
            return None

        def post(self, url, *, headers=None, data=None, **kw):
            assert "accounts.spotify.com" in url
            return _FakeReqCtx(_FakeResp(200, {"access_token": "app-token", "expires_in": 3600}))

        def get(self, url, *, headers=None, params=None, allow_redirects=None, **kw):
            if "api.spotify.com" in url:
                return _FakeReqCtx(_FakeResp(200, {
                    "tracks": {"items": [
                        {
                            "id": "1", "name": "Has Preview",
                            "artists": [{"name": "Artist A"}],
                            "album": {"images": [{"url": "big"}, {"url": "mid"}]},
                            "preview_url": "https://p.scdn.co/mp3/abc",
                            "duration_ms": 30000,
                        },
                        {
                            "id": "2", "name": "No Preview",
                            "artists": [{"name": "Artist B"}],
                            "album": {"images": [{"url": "big2"}]},
                            "preview_url": None, "duration_ms": 30000,
                        },
                    ]}
                }))
            if "itunes.apple.com" in url:
                calls["itunes"] += 1
                return _FakeReqCtx(_FakeResp(200, {
                    "results": [{"previewUrl": "https://audio-ssl.itunes.apple.com/x.m4a"}]
                }))
            return _FakeReqCtx(_FakeResp(404, {}))

    real = web_app.aiohttp.ClientSession
    web_app.aiohttp.ClientSession = FakeSession
    client, close = await _build_song_client_factory(SPOTIFY_ID, SPOTIFY_SECRET)()
    try:
        await client.post("/api/auth/login", json={"password": PASSWORD})
        resp = await client.get("/api/web/song/search?q=hello")
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True
        tracks = body["tracks"]
        assert len(tracks) == 2
        assert calls["itunes"] == 1
        by_title = {t["title"]: t for t in tracks}
        assert by_title["Has Preview"]["preview_url"] == "https://p.scdn.co/mp3/abc"
        assert by_title["No Preview"]["preview_url"].startswith(
            "https://audio-ssl.itunes.apple.com"
        )
        assert by_title["Has Preview"]["art"] == "mid"
        assert SPOTIFY_SECRET not in json.dumps(body)
    finally:
        await close()
        web_app.aiohttp.ClientSession = real


@_async
async def test_song_audio_rejects_disallowed_hosts():
    client, close = await _build_song_client_factory(SPOTIFY_ID, SPOTIFY_SECRET)()
    try:
        await client.post("/api/auth/login", json={"password": PASSWORD})
        bad = await client.get("/api/web/song/audio?url=https://evil.example.com/x.mp3")
        assert bad.status == 400
        assert (await bad.json())["error"] == "bad_url"
        insecure = await client.get("/api/web/song/audio?url=http://p.scdn.co/x.mp3")
        assert insecure.status == 400
    finally:
        await close()


@_async
async def test_song_audio_streams_allowed_host():
    class FakeSession:
        def __init__(self, *a, **kw):
            pass

        def get(self, url, *, allow_redirects=None, **kw):
            return _FakeReqCtx(
                _FakeResp(200, chunks=[b"ID3", b"audio-bytes"], headers={"Content-Type": "audio/mpeg"})
            )

        async def close(self):
            return None

    real = web_app.aiohttp.ClientSession
    web_app.aiohttp.ClientSession = FakeSession
    client, close = await _build_song_client_factory(SPOTIFY_ID, SPOTIFY_SECRET)()
    try:
        await client.post("/api/auth/login", json={"password": PASSWORD})
        resp = await client.get("/api/web/song/audio?url=https://p.scdn.co/mp3/abc")
        assert resp.status == 200
        body = await resp.read()
        assert body == b"ID3audio-bytes"
        assert resp.headers["Content-Type"].startswith("audio/mpeg")
    finally:
        await close()
        web_app.aiohttp.ClientSession = real


@_async
async def test_sing_page_requires_auth_and_renders():
    client, close = await _build_song_client_factory(SPOTIFY_ID, SPOTIFY_SECRET)()
    try:
        resp = await client.get("/sing", allow_redirects=False)
        assert resp.status == 302
        assert resp.headers["Location"] == "/login"
        await client.post("/api/auth/login", json={"password": PASSWORD})
        page = await client.get("/sing")
        assert page.status == 200
        assert "/static/web/js/sing.js" in await page.text()
    finally:
        await close()


@_async
async def test_relic_page_requires_auth_and_renders():
    client, close = await _build_song_client_factory(SPOTIFY_ID, SPOTIFY_SECRET)()
    try:
        resp = await client.get("/relic", allow_redirects=False)
        assert resp.status == 302
        assert resp.headers["Location"] == "/login"
        await client.post("/api/auth/login", json={"password": PASSWORD})
        page = await client.get("/relic")
        assert page.status == 200
        body = await page.text()
        assert "/static/web/js/relic.js" in body
        assert "BTF-LIGHTING" in body
        assert "The Relic" in body
    finally:
        await close()


def test_frontend_assets_present():
    root = Path(__file__).resolve().parent.parent / "static" / "web"
    for name in (
        "login.html",
        "index.html",
        "admits.html",
        "sing.html",
        "relic.html",
        "css/app.css",
        "css/relic.css",
        "js/app.js",
        "js/admits.js",
        "js/sing.js",
        "js/relic.js",
        "js/auth.js",
        "js/serial.js",
        "js/bottango.js",
        "js/envelope.js",
        "js/behavior.js",
        "js/motion.js",
        "js/realtime.js",
        "js/typed.js",
        "js/login.js",
    ):
        assert (root / name).is_file(), f"missing frontend asset: {name}"


def test_local_webapp_still_imports():
    """Smoke test: importing the existing local operator app must
    still work (we're strictly additive)."""
    # Reload in case pytest has already imported it.
    sys.modules.pop("app.webapp", None)
    mod = importlib.import_module("app.webapp")
    assert hasattr(mod, "main")
    assert callable(mod.main)

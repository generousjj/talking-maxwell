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
            return FakeCtx(
                200,
                {
                    "id": "sess_test",
                    "client_secret": {"value": EPHEMERAL_TOKEN, "expires_at": 1234567890},
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


def test_frontend_assets_present():
    root = Path(__file__).resolve().parent.parent / "static" / "web"
    for name in (
        "login.html",
        "index.html",
        "css/app.css",
        "js/app.js",
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

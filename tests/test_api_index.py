"""Integration tests for the Vercel/FastAPI build of the hosted app.

Mirrors ``tests/test_web_app.py`` endpoint-for-endpoint against
``api/index.py`` so any drift between the two deploy targets trips a
test here. Like the aiohttp tests, this avoids pytest-asyncio /
pytest-aiohttp: Starlette's ``TestClient`` is synchronous.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


PASSWORD = "correcthorsebatterystaple"
RAW_API_KEY = "sk-test-NEVER-LEAK-ME-1234567890"
EPHEMERAL_TOKEN = "ek_vercel_test_0001"

ROOT = Path(__file__).resolve().parent.parent
API_INDEX_PATH = ROOT / "api" / "index.py"


fastapi = pytest.importorskip("fastapi")
starlette_test = pytest.importorskip("starlette.testclient")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient  # noqa: E402

# Set env BEFORE importing api.index so its module-level AuthConfig
# sees the right values.
import os  # noqa: E402

from app.auth_core import hash_password  # noqa: E402


def _install_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "MAXWELL_WEB_PASSWORD_HASH",
        hash_password(PASSWORD, iterations=10_000),
    )
    monkeypatch.delenv("MAXWELL_WEB_PASSWORD", raising=False)
    monkeypatch.setenv("SESSION_SECRET", "vercel-test-secret-bytes-0123456789abc")
    monkeypatch.setenv("MAXWELL_WEB_INSECURE_COOKIE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", RAW_API_KEY)
    monkeypatch.delenv("MAXWELL_ALLOWED_ORIGIN", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("MAXWELL_TRUST_FORWARDED_FOR", raising=False)


def _fresh_app():
    """Load api/index.py directly from disk as a one-off module.

    We avoid ``import api.index`` because Vercel's Python runtime
    refuses to detect ``api/index.py`` as a serverless function when
    ``api/`` is a package (i.e. contains ``__init__.py``). So this
    repo intentionally keeps ``api/`` un-packaged, and the test suite
    loads the entrypoint via the importlib file-location API to
    match.
    """
    sys.modules.pop("_api_index_test_module", None)
    spec = importlib.util.spec_from_file_location(
        "_api_index_test_module", str(API_INDEX_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_api_index_test_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def client(monkeypatch):
    _install_env(monkeypatch)
    mod = _fresh_app()
    mod._reset_for_tests(api_key=RAW_API_KEY, allowed_origin="")
    with TestClient(mod.app) as c:
        yield c, mod


def test_health_is_public(client):
    c, _ = client
    resp = c.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_index_redirects_when_unauthenticated(client):
    c, _ = client
    resp = c.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/login"


def test_api_returns_401_when_unauthenticated(client):
    c, _ = client
    resp = c.post("/api/web/realtime/session", json={})
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_login_flow_and_me(client):
    c, _ = client
    bad = c.post("/api/auth/login", json={"password": "nope"})
    assert bad.status_code == 401

    good = c.post("/api/auth/login", json={"password": PASSWORD})
    assert good.status_code == 200
    assert good.json()["ok"] is True

    me = c.get("/api/auth/me")
    body = me.json()
    assert body["authed"] is True
    assert body["uid"] == "operator"

    assert c.post("/api/auth/logout").status_code == 200
    assert c.get("/api/auth/me").json()["authed"] is False


def test_origin_check_rejects_cross_site(client, monkeypatch):
    c, mod = client
    # Tighten allowed_origin to simulate a real deploy on a single domain.
    mod._reset_for_tests(allowed_origin="https://maxwell.example.com")
    c.post("/api/auth/login", json={"password": PASSWORD})
    resp = c.post(
        "/api/web/realtime/session",
        json={},
        headers={"origin": "https://evil.example.com"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "forbidden_origin"


def test_realtime_session_hides_raw_api_key(client, monkeypatch):
    c, mod = client
    captured: Dict[str, Any] = {}

    class FakeResponse:
        def __init__(self, status_code: int, payload: Dict[str, Any]):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)
            self.content = self.text.encode("utf-8")

        def json(self):
            return self._payload

    class FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def post(self, url, *, headers=None, content=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["content"] = content
            return FakeResponse(
                200,
                {
                    "id": "sess_test",
                    "client_secret": {
                        "value": EPHEMERAL_TOKEN,
                        "expires_at": 1234567890,
                    },
                },
            )

    monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)

    c.post("/api/auth/login", json={"password": PASSWORD})
    resp = c.post("/api/web/realtime/session", json={"voice": "ballad"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["client_secret"] == EPHEMERAL_TOKEN
    # Raw key goes server -> OpenAI, never in the response body.
    assert RAW_API_KEY in captured["headers"]["Authorization"]
    assert RAW_API_KEY not in json.dumps(body)


def test_realtime_session_returns_503_when_key_missing(client):
    c, mod = client
    mod._reset_for_tests(api_key="")
    c.post("/api/auth/login", json={"password": PASSWORD})
    resp = c.post("/api/web/realtime/session", json={})
    assert resp.status_code == 503
    assert resp.json()["error"] == "openai_key_missing"


def test_vercel_config_present():
    root = Path(__file__).resolve().parent.parent
    vjson = root / "vercel.json"
    assert vjson.is_file()
    data = json.loads(vjson.read_text())
    assert "rewrites" in data
    # Every request must flow through the single function entrypoint.
    assert any(r.get("destination", "").startswith("/api") for r in data["rewrites"])


def test_api_requirements_are_minimal():
    root = Path(__file__).resolve().parent.parent
    raw = (root / "api" / "requirements.txt").read_text().lower()
    # Strip comment lines so the "explanation" comment doesn't trip
    # the banned-substring check below.
    pkg_lines = [
        ln.strip() for ln in raw.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    pkgs = "\n".join(pkg_lines)
    # Don't ship the heavy local-mode deps to Vercel.
    for banned in ("numpy", "sounddevice", "soundfile", "pyserial", "matplotlib"):
        assert banned not in pkgs, f"{banned} shouldn't be in api/requirements.txt"
    for needed in ("fastapi", "httpx"):
        assert needed in pkgs

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


def test_admits_page_requires_auth_and_renders(client):
    c, _ = client
    # Unauthed: redirect to /login like every other non-public page.
    resp = c.get("/admits", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/login"

    # Authed: serves the cartoon-bubble admits HTML with the expected
    # title and a script import for admits.js. If these go missing, the
    # booth guest UI silently breaks, so we fail loudly here instead.
    c.post("/api/auth/login", json={"password": PASSWORD})
    page = c.get("/admits")
    assert page.status_code == 200
    body = page.text
    assert "Talk to Maxwell" in body
    assert "/static/web/js/admits.js" in body


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
            # GA Realtime API returns the token at the top-level
            # `value` field.
            return FakeResponse(
                200,
                {
                    "value": EPHEMERAL_TOKEN,
                    "expires_at": 1234567890,
                    "session": {"id": "sess_test", "type": "realtime"},
                },
            )

    monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)

    c.post("/api/auth/login", json={"password": PASSWORD})
    resp = c.post("/api/web/realtime/session", json={"voice": "ballad"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["client_secret"] == EPHEMERAL_TOKEN
    # Hit the GA client_secrets endpoint, not the deprecated /sessions.
    assert captured["url"].endswith("/v1/realtime/client_secrets")
    # No OpenAI-Beta header on GA — the API rejects it now.
    assert "OpenAI-Beta" not in (captured["headers"] or {})
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


def test_compose_instructions_appends_context(client):
    c, mod = client
    base = "BASE PERSONALITY"
    # No context -> unchanged base.
    assert mod.compose_instructions(base, None) == base
    assert mod.compose_instructions(base, "   ") == base
    # With context -> base preserved AND context present.
    out = mod.compose_instructions(base, "You are at NSO today.")
    assert base in out
    assert "You are at NSO today." in out
    # Context is length-capped so a client can't blow up instructions.
    # (Use a char that doesn't appear in the base/framing text — "z" —
    # so the count reflects only the injected context.)
    huge = "z" * 5000
    capped = mod.compose_instructions(base, huge)
    assert capped.count("z") == mod.MAX_CONTEXT_CHARS


def test_vercel_config_present():
    root = Path(__file__).resolve().parent.parent
    vjson = root / "vercel.json"
    assert vjson.is_file()
    data = json.loads(vjson.read_text())
    assert "rewrites" in data
    # Every request must flow through the single function entrypoint.
    assert any(r.get("destination", "").startswith("/api") for r in data["rewrites"])


def test_motion_config_exposes_correct_pins(client):
    # Regression: before the motion-config endpoint existed, the JS
    # transport hardcoded head_lr=pin10/head_ud=pin11, which didn't
    # match config.yaml's head_lr=5/head_ud=6 and silently sent every
    # head command to empty GPIO pins. Only wings (pin 3, matched)
    # ever visibly moved.
    c, _ = client
    c.post("/api/auth/login", json={"password": PASSWORD})
    resp = c.get("/api/web/motion-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    chans = data["channels"]
    assert chans["head_lr"]["pin"] == 5
    assert chans["head_ud"]["pin"] == 6
    assert chans["wing"]["pin"] == 3
    assert chans["jaw"]["pin"] == 9
    assert chans["jaw"]["inverted"] is True
    assert data["source"] == "config.yaml"


def test_motion_config_exposes_full_behavior_gains(client):
    # Every Python BehaviorGains field the JS engine executes must
    # ship over the wire — missing any of them silently reverts the
    # browser to baked-in JS defaults that drift from config.yaml.
    c, _ = client
    c.post("/api/auth/login", json={"password": PASSWORD})
    resp = c.get("/api/web/motion-config")
    assert resp.status_code == 200
    gains = resp.json()["gains"]
    required = {
        "headLrDrift", "headUdDrift",
        "nodStrength", "emphasisStrength", "questionTilt",
        "wingStrength", "wingCooldownS",
        "waitingWingStrength", "waitingWingPeriodS",
        "envelopeHeadBob", "speakingDriftRate",
        "idleNodStrength", "idleTiltStrength",
        "idleNodPeriodS", "idleTiltPeriodS",
        "headSmoothingTauS",
    }
    missing = required - gains.keys()
    assert not missing, f"motion-config is missing gains: {missing}"
    # seed is allowed to be null, but the key must exist so the JS
    # engine can wire its PRNG deterministically when set.
    assert "seed" in gains
    # startingPwm must be filled in even when config.yaml says null —
    # otherwise the browser would register servos at PWM=0.
    for name, ch in resp.json()["channels"].items():
        assert "startingPwm" in ch and isinstance(ch["startingPwm"], int), (
            f"channel {name} missing startingPwm midpoint fallback"
        )


def test_motion_config_gated_by_auth(client):
    c, _ = client
    resp = c.get("/api/web/motion-config")
    assert resp.status_code == 401


SPOTIFY_ID = "spotify-client-id"
SPOTIFY_SECRET = "spotify-client-secret-NEVER-LEAK"


class _FakeSongResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_song_search_requires_auth(client):
    c, _ = client
    resp = c.get("/api/web/song/search?q=test")
    assert resp.status_code == 401


def test_song_search_503_without_credentials(client):
    c, mod = client
    mod._reset_for_tests(spotify_id="", spotify_secret="")
    c.post("/api/auth/login", json={"password": PASSWORD})
    resp = c.get("/api/web/song/search?q=test")
    assert resp.status_code == 503
    assert resp.json()["error"] == "spotify_not_configured"


def test_song_search_spotify_with_itunes_fallback(client, monkeypatch):
    c, mod = client
    mod._reset_for_tests(spotify_id=SPOTIFY_ID, spotify_secret=SPOTIFY_SECRET)
    calls = {"itunes": 0}

    class FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def aclose(self):
            return None

        async def post(self, url, *, headers=None, content=None, **kw):
            # Spotify app-token mint (client-credentials).
            assert "accounts.spotify.com" in url
            return _FakeSongResponse(200, {"access_token": "app-token", "expires_in": 3600})

        async def get(self, url, *, headers=None, params=None, **kw):
            if "api.spotify.com" in url:
                return _FakeSongResponse(
                    200,
                    {
                        "tracks": {
                            "items": [
                                {
                                    "id": "1",
                                    "name": "Has Preview",
                                    "artists": [{"name": "Artist A"}],
                                    "album": {"images": [{"url": "big"}, {"url": "mid"}]},
                                    "preview_url": "https://p.scdn.co/mp3/abc",
                                    "duration_ms": 30000,
                                },
                                {
                                    "id": "2",
                                    "name": "No Preview",
                                    "artists": [{"name": "Artist B"}],
                                    "album": {"images": [{"url": "big2"}]},
                                    "preview_url": None,
                                    "duration_ms": 30000,
                                },
                            ]
                        }
                    },
                )
            if "itunes.apple.com" in url:
                calls["itunes"] += 1
                return _FakeSongResponse(
                    200,
                    {"results": [{"previewUrl": "https://audio-ssl.itunes.apple.com/x.m4a"}]},
                )
            return _FakeSongResponse(404, {})

    monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
    c.post("/api/auth/login", json={"password": PASSWORD})
    resp = c.get("/api/web/song/search?q=hello")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    tracks = body["tracks"]
    assert len(tracks) == 2
    # iTunes fallback resolved a preview for the track Spotify left null.
    assert calls["itunes"] == 1
    by_title = {t["title"]: t for t in tracks}
    assert by_title["Has Preview"]["preview_url"] == "https://p.scdn.co/mp3/abc"
    assert by_title["No Preview"]["preview_url"].startswith(
        "https://audio-ssl.itunes.apple.com"
    )
    # Mid-size album art is preferred over the largest image.
    assert by_title["Has Preview"]["art"] == "mid"
    # The Spotify secret must never reach the browser.
    assert SPOTIFY_SECRET not in json.dumps(body)


def test_song_audio_rejects_disallowed_hosts(client):
    c, _ = client
    c.post("/api/auth/login", json={"password": PASSWORD})
    bad = c.get("/api/web/song/audio?url=https://evil.example.com/x.mp3")
    assert bad.status_code == 400
    assert bad.json()["error"] == "bad_url"
    # Non-HTTPS is rejected even for an allowed host.
    insecure = c.get("/api/web/song/audio?url=http://p.scdn.co/x.mp3")
    assert insecure.status_code == 400


def test_song_audio_streams_allowed_host(client, monkeypatch):
    c, mod = client

    class FakeUpstream:
        def __init__(self):
            self.status_code = 200
            self.headers = {"content-type": "audio/mpeg"}

        async def aiter_bytes(self, n=65536):
            for chunk in (b"ID3", b"audio-bytes"):
                yield chunk

        async def aclose(self):
            return None

    class FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

        def build_request(self, method, url):
            return (method, url)

        async def send(self, request, stream=False):
            return FakeUpstream()

        async def aclose(self):
            return None

    monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
    c.post("/api/auth/login", json={"password": PASSWORD})
    resp = c.get("/api/web/song/audio?url=https://p.scdn.co/mp3/abc")
    assert resp.status_code == 200
    assert resp.content == b"ID3audio-bytes"
    assert resp.headers["content-type"].startswith("audio/mpeg")


def test_sing_page_requires_auth_and_renders(client):
    c, _ = client
    resp = c.get("/sing", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/login"
    c.post("/api/auth/login", json={"password": PASSWORD})
    page = c.get("/sing")
    assert page.status_code == 200
    assert "/static/web/js/sing.js" in page.text


def test_relic_page_requires_auth_and_renders(client):
    c, _ = client
    resp = c.get("/relic", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/login"
    c.post("/api/auth/login", json={"password": PASSWORD})
    page = c.get("/relic")
    assert page.status_code == 200
    assert "/static/web/js/relic.js" in page.text
    assert "BTF-LIGHTING" in page.text
    assert "The Relic" in page.text


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

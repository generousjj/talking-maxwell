"""Unit tests for the password-gated web auth module.

These tests do not boot an aiohttp app — they cover the pure helper
functions (password hashing, session signing, rate limiting) that the
integration tests in ``test_web_app.py`` rely on.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time

import pytest

from app import web_auth


def test_hash_and_verify_password_roundtrip():
    encoded = web_auth.hash_password("s3cret!", iterations=50_000)
    assert encoded.startswith("pbkdf2_sha256$50000$")
    assert web_auth.verify_password("s3cret!", encoded)
    assert not web_auth.verify_password("wrong", encoded)


def test_verify_rejects_bad_format():
    assert not web_auth.verify_password("s", "not-a-hash")
    assert not web_auth.verify_password("s", "pbkdf2_sha256$bad$$")


def test_session_sign_and_verify_roundtrip():
    secret = b"test-secret-bytes-0123456789abcd"
    now = int(time.time())
    payload = {"uid": "op", "iat": now, "exp": now + 600}
    token = web_auth.sign_session(secret, payload)
    got = web_auth.verify_session(secret, token)
    assert got == payload


def test_session_verify_rejects_expired_token():
    secret = b"s" * 32
    payload = {"uid": "op", "iat": 0, "exp": int(time.time()) - 10}
    token = web_auth.sign_session(secret, payload)
    assert web_auth.verify_session(secret, token) is None


def test_session_verify_rejects_tampered_payload():
    secret = b"s" * 32
    now = int(time.time())
    token = web_auth.sign_session(secret, {"uid": "op", "exp": now + 60})
    body, sig = token.rsplit(".", 1)
    bad_payload = base64.urlsafe_b64encode(b'{"uid":"admin","exp":9999999999}').decode("ascii").rstrip("=")
    bad = f"{bad_payload}.{sig}"
    assert web_auth.verify_session(secret, bad) is None


def test_login_limiter_locks_after_max_attempts():
    async def run() -> None:
        limiter = web_auth.LoginLimiter(window_s=60, max_attempts=3)
        for _ in range(3):
            allowed, _ = await limiter.check("1.2.3.4")
            assert allowed
            await limiter.record_failure("1.2.3.4")
        allowed, retry_after = await limiter.check("1.2.3.4")
        assert not allowed
        assert retry_after > 0

        # Other IPs unaffected.
        allowed_other, _ = await limiter.check("5.6.7.8")
        assert allowed_other

    asyncio.run(run())


def test_auth_config_generates_plain_password_when_none_set(monkeypatch):
    for var in (
        "MAXWELL_WEB_PASSWORD",
        "MAXWELL_WEB_PASSWORD_HASH",
        "SESSION_SECRET",
        "MAXWELL_ALLOWED_ORIGIN",
        "MAXWELL_WEB_INSECURE_COOKIE",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = web_auth.AuthConfig.from_env()
    assert cfg.password_hash is None
    assert cfg.dev_plain_password  # auto-generated
    assert cfg.password_ok(cfg.dev_plain_password)
    assert not cfg.password_ok("nope")


def test_auth_config_prefers_hash(monkeypatch):
    hashed = web_auth.hash_password("hunter2", iterations=10_000)
    monkeypatch.setenv("MAXWELL_WEB_PASSWORD_HASH", hashed)
    monkeypatch.delenv("MAXWELL_WEB_PASSWORD", raising=False)
    monkeypatch.setenv("SESSION_SECRET", "a" * 32)
    cfg = web_auth.AuthConfig.from_env()
    assert cfg.password_hash == hashed
    assert cfg.password_ok("hunter2")
    assert not cfg.password_ok("hunter3")

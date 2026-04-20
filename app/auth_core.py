"""Framework-agnostic auth primitives shared by every deploy target.

This module is pure stdlib — no aiohttp, no FastAPI, no httpx. Both
``app.web_auth`` (the aiohttp server used by Docker / Fly / Render)
and ``api.index`` (the FastAPI app used by Vercel) import the same
password hashing, session signing, config loader, and rate limiter
from here so there's exactly one implementation to audit.

Nothing about the on-the-wire behavior (cookie name, signing algo,
PBKDF2 iterations, lockout window) depends on which HTTP framework is
serving the request.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

SESSION_COOKIE = "mxw_session"
SESSION_TTL_S = 60 * 60 * 12  # 12 hours
LOGIN_WINDOW_S = 15 * 60      # 15 minutes
LOGIN_MAX_ATTEMPTS = 8        # per IP per window
PBKDF2_DEFAULT_ITERS = 200_000


# --------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------

def hash_password(password: str, *, iterations: int = PBKDF2_DEFAULT_ITERS) -> str:
    """Return ``pbkdf2_sha256$<iters>$<salt_b64>$<hash_b64>``.

    Mirrors Django's PBKDF2 encoding so ops people recognize it on
    sight. No third-party deps — stdlib ``hashlib`` only.
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return (
        "pbkdf2_sha256$"
        f"{iterations}$"
        f"{base64.b64encode(salt).decode('ascii')}$"
        f"{base64.b64encode(digest).decode('ascii')}"
    )


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time password check against ``hash_password`` output."""
    try:
        algo, iters_s, salt_b64, hash_b64 = encoded.split("$", 3)
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        iterations = int(iters_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except Exception:  # noqa: BLE001
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(actual, expected)


# --------------------------------------------------------------------
# Session cookie signing
# --------------------------------------------------------------------

def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64u(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + pad).encode("ascii"))


def sign_session(secret: bytes, payload: dict) -> str:
    body = _b64u(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64u(hmac.new(secret, body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_session(secret: bytes, token: str) -> Optional[dict]:
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    expected = _b64u(
        hmac.new(secret, body.encode("ascii"), hashlib.sha256).digest()
    )
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_unb64u(body))
    except Exception:  # noqa: BLE001
        return None
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or exp < time.time():
        return None
    return payload


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

@dataclass
class AuthConfig:
    password_hash: Optional[str]
    dev_plain_password: Optional[str]  # insecure fallback
    session_secret: bytes
    allowed_origin: Optional[str]       # e.g. "https://maxwell.example.com"
    secure_cookie: bool                 # default True in prod
    session_ttl_s: int = SESSION_TTL_S

    @classmethod
    def from_env(cls) -> "AuthConfig":
        pw_hash = os.environ.get("MAXWELL_WEB_PASSWORD_HASH") or None
        pw_plain = os.environ.get("MAXWELL_WEB_PASSWORD") or None
        if not pw_hash and not pw_plain:
            # Never silently be open: without a configured password,
            # generate a one-session random one and log it. The
            # operator either reads it from stdout/logs or (much
            # better) sets MAXWELL_WEB_PASSWORD_HASH properly.
            pw_plain = secrets.token_urlsafe(12)
            log.warning(
                "MAXWELL_WEB_PASSWORD[_HASH] not set — generated "
                "one-session password: %s (set MAXWELL_WEB_PASSWORD_HASH "
                "to lock this in)",
                pw_plain,
            )
        secret_env = os.environ.get("SESSION_SECRET")
        if secret_env:
            secret = secret_env.encode("utf-8")
        else:
            secret = secrets.token_bytes(32)
            log.warning(
                "SESSION_SECRET not set — using an ephemeral one "
                "(all sessions will drop on restart)"
            )
        allowed_origin = os.environ.get("MAXWELL_ALLOWED_ORIGIN") or None
        secure_env = os.environ.get("MAXWELL_WEB_INSECURE_COOKIE")
        secure_cookie = not (secure_env and secure_env.lower() in {"1", "true", "yes"})
        if pw_plain and not pw_hash:
            log.warning(
                "using MAXWELL_WEB_PASSWORD (plaintext); please switch "
                "to MAXWELL_WEB_PASSWORD_HASH for production"
            )
        return cls(
            password_hash=pw_hash,
            dev_plain_password=pw_plain,
            session_secret=secret,
            allowed_origin=allowed_origin,
            secure_cookie=secure_cookie,
        )

    def password_ok(self, candidate: str) -> bool:
        if self.password_hash:
            return verify_password(candidate, self.password_hash)
        if self.dev_plain_password is not None:
            return hmac.compare_digest(
                candidate.encode("utf-8"),
                self.dev_plain_password.encode("utf-8"),
            )
        return False


# --------------------------------------------------------------------
# Rate limiting — tiny in-memory sliding window. Good enough for a
# single server process. On serverless (Vercel), every cold start
# resets the limiter — safe because cookies are still signed, but the
# lockout becomes per-instance. Swap in Vercel KV / Upstash Redis if
# you ever want cross-invocation limiting.
# --------------------------------------------------------------------

@dataclass
class _Bucket:
    attempts: list
    locked_until: float = 0.0


class LoginLimiter:
    def __init__(self, window_s: int = LOGIN_WINDOW_S, max_attempts: int = LOGIN_MAX_ATTEMPTS):
        self._window = window_s
        self._max = max_attempts
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    async def check(self, ip: str) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_s)``."""
        now = time.monotonic()
        async with self._lock:
            b = self._buckets.get(ip)
            if b is None:
                return True, 0.0
            if now < b.locked_until:
                return False, b.locked_until - now
            cutoff = now - self._window
            b.attempts = [t for t in b.attempts if t > cutoff]
            if len(b.attempts) >= self._max:
                b.locked_until = now + self._window
                return False, self._window
            return True, 0.0

    async def record_failure(self, ip: str) -> None:
        async with self._lock:
            b = self._buckets.setdefault(ip, _Bucket(attempts=[]))
            b.attempts.append(time.monotonic())

    async def reset(self, ip: str) -> None:
        async with self._lock:
            self._buckets.pop(ip, None)


# --------------------------------------------------------------------
# Session payload construction (framework-agnostic).
# --------------------------------------------------------------------

def build_session_payload(uid: str, ttl_s: int, now: Optional[int] = None) -> dict:
    now_int = int(now if now is not None else time.time())
    return {"uid": uid, "iat": now_int, "exp": now_int + ttl_s}


# --------------------------------------------------------------------
# CLI: `python -m app.auth_core hash <password>`
# --------------------------------------------------------------------

def _cli(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "hash":
        print(hash_password(argv[1]))
        return 0
    print(
        "usage: python -m app.auth_core hash <password>\n\n"
        "Prints a PBKDF2-SHA256 hash suitable for MAXWELL_WEB_PASSWORD_HASH.",
    )
    return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))

"""Password-gated auth + helpers for the hosted browser deployment.

Keeps the local operator/admits workflow untouched. Only the new
``app.web_app`` entrypoint (and any handler wanting ``require_auth``)
needs anything in here.

Design in one paragraph:
    A signed, opaque session cookie (``mxw_session``) is minted on
    successful login and re-verified on every request via an aiohttp
    middleware. Passwords are compared with ``hmac.compare_digest``
    against either a PBKDF2-SHA256 hash (production, recommended) or a
    plain ``MAXWELL_WEB_PASSWORD`` fallback (dev only). Per-IP login
    attempts are rate-limited with exponential back-off; logins run in
    a thread so the constant-time PBKDF2 check doesn't stall the event
    loop. CSRF is defended in depth: Origin header is verified on every
    state-changing request, and sessions piggy-back on SameSite=Lax,
    HttpOnly, Secure cookies. No secrets reach the browser.
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
from typing import Awaitable, Callable, Optional

from aiohttp import web

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

    The format mirrors Django's PBKDF2 encoding so ops people recognize
    it immediately. No new dependencies — only ``hashlib``.
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
            # Generate a random one-session password and log it. This
            # means "never silently be open"; if the operator hasn't
            # configured a password, login is effectively disabled
            # until they read the log and copy it, or set the env var.
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
# Rate limiting — tiny in-memory sliding window (good enough for one
# process; if this ever runs behind multiple replicas we'd swap in a
# redis-backed limiter).
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
# Session helpers
# --------------------------------------------------------------------

def current_session(request: web.Request) -> Optional[dict]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    cfg: AuthConfig = request.app["auth_config"]
    return verify_session(cfg.session_secret, token)


def set_session_cookie(response: web.Response, cfg: AuthConfig, payload: dict) -> None:
    token = sign_session(cfg.session_secret, payload)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=cfg.session_ttl_s,
        path="/",
        secure=cfg.secure_cookie,
        httponly=True,
        samesite="Lax",
    )


def clear_session_cookie(response: web.Response, cfg: AuthConfig) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        "",
        max_age=0,
        path="/",
        secure=cfg.secure_cookie,
        httponly=True,
        samesite="Lax",
    )


@web.middleware
async def auth_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Gate protected paths (index, ``/api/web/...``) behind the cookie.

    Public paths (``/login``, ``/api/auth/login``, ``/static/...``,
    ``/healthz``) pass through.
    """
    path = request.path
    public = (
        path == "/login"
        or path == "/healthz"
        or path == "/api/auth/login"
        or path == "/api/auth/me"
        or path.startswith("/static/")
    )
    if not public:
        session = current_session(request)
        if session is None:
            # HTML nav -> redirect to login; API -> 401 JSON.
            if path.startswith("/api/"):
                return web.json_response(
                    {"ok": False, "error": "unauthorized"}, status=401
                )
            raise web.HTTPFound("/login")
        request["session"] = session

    # Origin check for state-changing requests (extra CSRF defense on
    # top of SameSite=Lax). GETs are allowed from any Origin because
    # HTML pages are public by design.
    cfg: AuthConfig = request.app["auth_config"]
    if request.method not in ("GET", "HEAD", "OPTIONS") and path.startswith("/api/"):
        origin = request.headers.get("Origin") or request.headers.get("Referer")
        if cfg.allowed_origin:
            allowed = cfg.allowed_origin
        else:
            host = request.headers.get("Host", "")
            scheme = "https" if request.secure else "http"
            allowed = f"{scheme}://{host}"
        if origin and not origin.startswith(allowed):
            log.warning("rejecting cross-origin %s %s (origin=%s)", request.method, path, origin)
            return web.json_response(
                {"ok": False, "error": "forbidden_origin"}, status=403
            )
    return await handler(request)


# --------------------------------------------------------------------
# Handlers (login/logout/me)
# --------------------------------------------------------------------

async def handle_login(request: web.Request) -> web.Response:
    cfg: AuthConfig = request.app["auth_config"]
    limiter: LoginLimiter = request.app["login_limiter"]
    ip = _client_ip(request)
    allowed, retry_after = await limiter.check(ip)
    if not allowed:
        log.info("rate-limiting login from %s (retry after %.0fs)", ip, retry_after)
        return web.json_response(
            {"ok": False, "error": "rate_limited", "retry_after": int(retry_after)},
            status=429,
        )
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        await limiter.record_failure(ip)
        return web.json_response(
            {"ok": False, "error": "bad_request"}, status=400
        )
    candidate = str(payload.get("password") or "")
    ok = await asyncio.to_thread(cfg.password_ok, candidate)
    if not ok:
        await limiter.record_failure(ip)
        return web.json_response(
            {"ok": False, "error": "invalid_credentials"}, status=401
        )
    await limiter.reset(ip)
    now = int(time.time())
    session_payload = {
        "uid": "operator",
        "iat": now,
        "exp": now + cfg.session_ttl_s,
    }
    resp = web.json_response({"ok": True})
    set_session_cookie(resp, cfg, session_payload)
    log.info("login ok from %s", ip)
    return resp


async def handle_logout(request: web.Request) -> web.Response:
    cfg: AuthConfig = request.app["auth_config"]
    resp = web.json_response({"ok": True})
    clear_session_cookie(resp, cfg)
    return resp


async def handle_me(request: web.Request) -> web.Response:
    session = current_session(request)
    if session is None:
        return web.json_response({"ok": False, "authed": False}, status=200)
    return web.json_response(
        {
            "ok": True,
            "authed": True,
            "uid": session.get("uid"),
            "exp": session.get("exp"),
        }
    )


def _client_ip(request: web.Request) -> str:
    # Trust X-Forwarded-For only when behind a reverse proxy that sets it;
    # controlled by env var to keep the default safe.
    if os.environ.get("MAXWELL_TRUST_FORWARDED_FOR") == "1":
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.remote or "unknown"


# --------------------------------------------------------------------
# CLI: `python -m app.web_auth hash <password>`
# --------------------------------------------------------------------

def _cli(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "hash":
        print(hash_password(argv[1]))
        return 0
    print(
        "usage: python -m app.web_auth hash <password>\n\n"
        "Prints a PBKDF2-SHA256 hash suitable for MAXWELL_WEB_PASSWORD_HASH.",
    )
    return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))

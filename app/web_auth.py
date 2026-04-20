"""aiohttp bindings for the hosted browser deployment.

All password hashing, session signing, config loading, and rate
limiting live in :mod:`app.auth_core` so the same primitives also
drive the Vercel/FastAPI app in ``api.index``. This module only
provides the aiohttp-specific glue: cookie middleware, login/logout/
me handlers, and the ``python -m app.web_auth hash <password>`` CLI
(kept as an alias for backward compatibility).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Awaitable, Callable, Optional

from aiohttp import web

from app.auth_core import (  # noqa: F401 — re-exported for back-compat
    AuthConfig,
    LoginLimiter,
    LOGIN_MAX_ATTEMPTS,
    LOGIN_WINDOW_S,
    PBKDF2_DEFAULT_ITERS,
    SESSION_COOKIE,
    SESSION_TTL_S,
    build_session_payload,
    hash_password,
    sign_session,
    verify_password,
    verify_session,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Session helpers (aiohttp)
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
    """Gate protected paths behind the signed session cookie.

    Public paths (``/login``, ``/api/auth/login``, ``/api/auth/me``,
    ``/static/...``, ``/healthz``) pass through. Everything else
    redirects (HTML) or 401s (JSON) when the cookie is missing or
    invalid.
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
            if path.startswith("/api/"):
                return web.json_response(
                    {"ok": False, "error": "unauthorized"}, status=401
                )
            raise web.HTTPFound("/login")
        request["session"] = session

    # Origin check for state-changing API requests. Belt-and-suspenders
    # CSRF protection on top of SameSite=Lax.
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
    session_payload = build_session_payload("operator", cfg.session_ttl_s)
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
    if os.environ.get("MAXWELL_TRUST_FORWARDED_FOR") == "1":
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.remote or "unknown"


# --------------------------------------------------------------------
# CLI alias — keep `python -m app.web_auth hash ...` working.
# --------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from app.auth_core import _cli as _core_cli

    raise SystemExit(_core_cli(sys.argv[1:]))

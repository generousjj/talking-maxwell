"""Bottango transport backend.

Bottango Desktop exposes an HTTP endpoint for "API controlled" inputs, which
is the recommended way to drive live motion from an external program. Because
the sample Maxwell project shipped with this prototype has no controlSchemes
configured, the Bottango-side setup is a one-time manual step documented in
the README: create four API-controlled input channels inside Bottango and map
them to the Head Rotation, Head Tilt, Mouth, and Wing servos.

The backend is deliberately resilient: connection failures and transient
HTTP errors are logged and retried with simple backoff. Behavior logic keeps
producing frames even while the backend is temporarily unreachable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

try:  # optional dependency; only required when this backend is selected
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover - aiohttp is imported lazily
    aiohttp = None  # type: ignore

from motion.models import MotionFrame

from .base import MotionBackend

log = logging.getLogger(__name__)


@dataclass
class BottangoIdentifiers:
    """Identifiers for the four API-controlled inputs configured in Bottango.

    The string values should match the "API Identifier" you set on each
    API-controlled input inside Bottango Desktop. Keep them in sync with your
    config.yaml.
    """

    jaw: str = "jaw_api"
    head_lr: str = "head_lr_api"
    head_ud: str = "head_ud_api"
    wing: str = "wing_api"


@dataclass
class BottangoBackend(MotionBackend):
    """HTTP client for Bottango's API-controlled inputs.

    ``base_url`` must point at the Bottango desktop API (default from the
    Bottango docs is http://localhost:59224). ``path_template`` controls the
    URL path used for an update; it must contain ``{identifier}`` and
    ``{value}`` placeholders so it is easy to swap if Bottango's API evolves.

    Bottango historically accepts values in either normalized [0, 1] form or
    percentage [0, 100] form depending on the endpoint. This backend sends
    normalized floats by default; if your Bottango version expects percent,
    set ``value_scale = 100.0`` in the config.
    """

    name: str = "bottango"
    base_url: str = "http://localhost:59224"
    path_template: str = "/setInputValue/{identifier}/{value}"
    identifiers: BottangoIdentifiers = field(default_factory=BottangoIdentifiers)
    request_timeout_s: float = 0.25
    value_scale: float = 1.0
    health_path: str = "/"

    _session: Optional["aiohttp.ClientSession"] = field(default=None, init=False)
    _connected: bool = field(default=False, init=False)
    _last_warn: float = field(default=0.0, init=False)
    _last_values: dict = field(default_factory=dict, init=False)
    _min_delta: float = 0.004

    async def connect(self) -> None:
        if self._connected:
            return
        if aiohttp is None:
            raise RuntimeError(
                "The `aiohttp` package is required for the Bottango backend. "
                "Install it with `pip install aiohttp` and re-run."
            )
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._connected = True
        reachable = await self._health_check()
        if reachable:
            log.info("bottango backend reachable at %s", self.base_url)
        else:
            log.warning(
                "bottango backend at %s is not reachable yet; frames will be "
                "attempted anyway and will succeed once Bottango starts.",
                self.base_url,
            )

    async def send_frame(self, frame: MotionFrame) -> None:
        if not self._connected or self._session is None:
            return
        await asyncio.gather(
            self._send_channel(self.identifiers.jaw, frame.jaw_open),
            self._send_channel(self.identifiers.head_lr, frame.head_lr),
            self._send_channel(self.identifiers.head_ud, frame.head_ud),
            self._send_channel(self.identifiers.wing, frame.wing),
            return_exceptions=True,
        )

    async def close(self) -> None:
        if not self._connected:
            return
        self._connected = False
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:  # noqa: BLE001
                pass
            self._session = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def _send_channel(self, identifier: str, value: float) -> None:
        if not identifier:
            return
        last = self._last_values.get(identifier)
        if last is not None and abs(last - value) < self._min_delta:
            return
        self._last_values[identifier] = value
        url = self.base_url.rstrip("/") + self.path_template.format(
            identifier=identifier,
            value=f"{value * self.value_scale:.4f}",
        )
        try:
            assert self._session is not None
            async with self._session.post(url) as resp:
                if resp.status >= 400:
                    self._warn_throttled(
                        f"Bottango POST {url} returned {resp.status}"
                    )
        except asyncio.TimeoutError:
            self._warn_throttled(f"Bottango POST {url} timed out")
        except Exception as exc:  # noqa: BLE001 - transport errors are non-fatal
            self._warn_throttled(f"Bottango POST {url} failed: {exc}")

    async def _health_check(self) -> bool:
        if aiohttp is None or self._session is None:
            return False
        try:
            url = self.base_url.rstrip("/") + self.health_path
            async with self._session.get(url) as resp:
                return resp.status < 500
        except Exception:  # noqa: BLE001
            return False

    def _warn_throttled(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warn < 2.0:
            return
        self._last_warn = now
        log.warning(message)

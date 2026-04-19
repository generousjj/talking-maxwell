"""State machine coordinating the conversational states.

The machine is intentionally minimal: transitions are driven by the app
pipeline (main loop) rather than inferred from events, which keeps motion
behavior predictable and easy to debug.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from .models import ConversationState

log = logging.getLogger(__name__)

Listener = Callable[[ConversationState, ConversationState], Awaitable[None]]


@dataclass
class ConversationStateMachine:
    """Holds current ConversationState with async-safe transitions."""

    state: ConversationState = ConversationState.IDLE
    _entered_at: float = 0.0
    _listeners: list[Listener] = None  # type: ignore[assignment]
    _lock: asyncio.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._entered_at = time.monotonic()
        self._listeners = []
        self._lock = asyncio.Lock()

    @property
    def time_in_state(self) -> float:
        return time.monotonic() - self._entered_at

    def on_change(self, listener: Listener) -> None:
        self._listeners.append(listener)

    async def transition(self, new_state: ConversationState) -> None:
        async with self._lock:
            if new_state == self.state:
                return
            old = self.state
            self.state = new_state
            self._entered_at = time.monotonic()
            log.debug("state %s -> %s", old.value, new_state.value)
            for listener in list(self._listeners):
                try:
                    await listener(old, new_state)
                except Exception:  # pragma: no cover - defensive
                    log.exception("state listener failed")

    async def idle(self) -> None:
        await self.transition(ConversationState.IDLE)

    async def listening(self) -> None:
        await self.transition(ConversationState.LISTENING)

    async def thinking(self) -> None:
        await self.transition(ConversationState.THINKING)

    async def speaking(self) -> None:
        await self.transition(ConversationState.SPEAKING)

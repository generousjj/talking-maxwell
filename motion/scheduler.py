"""Fixed-rate motion scheduler.

The scheduler ticks the behavior engine at a configurable rate (defaults to
30 Hz) and forwards motion frames to the active transport backend. It is
intentionally small so behavior engine changes stay testable in isolation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from .behavior_engine import BehaviorEngine
from .models import ConversationState, MotionFrame, SpeakingContext
from .state_machine import ConversationStateMachine
from transport.base import MotionBackend

log = logging.getLogger(__name__)

SpeakingContextProvider = Callable[[float], Optional[SpeakingContext]]


@dataclass
class MotionScheduler:
    """Drives the BehaviorEngine at fixed rate and streams frames to transport."""

    behavior: BehaviorEngine
    backend: MotionBackend
    state_machine: ConversationStateMachine
    rate_hz: float = 30.0
    speaking_context_provider: Optional[SpeakingContextProvider] = None

    _task: Optional[asyncio.Task] = None
    _stopped: bool = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopped = False
        await self.backend.connect()
        self._task = asyncio.create_task(self._run(), name="motion-scheduler")

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        await self.backend.close()

    async def _run(self) -> None:
        period = 1.0 / max(1.0, self.rate_hz)
        next_tick = time.monotonic()
        try:
            while not self._stopped:
                now = time.monotonic()
                context: Optional[SpeakingContext] = None
                if (
                    self.state_machine.state == ConversationState.SPEAKING
                    and self.speaking_context_provider is not None
                ):
                    context = self.speaking_context_provider(now)

                output = self.behavior.tick(
                    state=self.state_machine.state,
                    now=now,
                    speaking=context,
                )
                frame = output.to_frame(timestamp=now)
                try:
                    await self.backend.send_frame(frame)
                except Exception:  # noqa: BLE001 - backend failures are non-fatal
                    log.exception("motion backend send failed")

                next_tick += period
                sleep_for = next_tick - time.monotonic()
                if sleep_for < 0:
                    next_tick = time.monotonic()
                    sleep_for = 0
                await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            return


async def run_with_scheduler(
    *,
    scheduler: MotionScheduler,
    main: Callable[[], Awaitable[None]],
) -> None:
    """Run the main coroutine while the scheduler ticks in the background."""
    await scheduler.start()
    try:
        await main()
    finally:
        await scheduler.stop()

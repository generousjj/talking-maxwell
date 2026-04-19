"""Transport backend abstract base class.

All backends must accept normalized MotionFrames with values in [0, 1] and
handle their own connect / close lifecycle. Behavior logic must never reach
past this interface.
"""

from __future__ import annotations

import abc
from typing import Optional

from motion.models import MotionFrame


class MotionBackend(abc.ABC):
    """Pluggable transport for motion frames.

    Implementations should be resilient: ``send_frame`` is expected to be
    called ~30 times per second and should never raise for transient issues.
    Log-and-continue is preferred over propagating exceptions.
    """

    name: str = "abstract"

    @abc.abstractmethod
    async def connect(self) -> None:
        """Perform any connection / handshake. Must be idempotent."""

    @abc.abstractmethod
    async def send_frame(self, frame: MotionFrame) -> None:
        """Send a single normalized motion frame."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release any resources. Must be idempotent."""

    @property
    def is_connected(self) -> bool:
        return True

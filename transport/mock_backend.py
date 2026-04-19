"""Mock motion backend.

Logs motion frames to stdout at a throttled rate (or every frame in verbose
mode), and can optionally write a CSV trace for offline analysis or render a
matplotlib plot after a session.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from motion.models import MotionFrame

from .base import MotionBackend

log = logging.getLogger(__name__)


@dataclass
class MockBackend(MotionBackend):
    """Console/CSV/plot backend for local development without hardware."""

    name: str = "mock"
    csv_path: Optional[Path] = None
    log_every_n: int = 10
    plot_after: bool = False

    _csv_file: Optional[object] = field(default=None, init=False)
    _csv_writer: Optional[object] = field(default=None, init=False)
    _frames: list[MotionFrame] = field(default_factory=list, init=False)
    _tick: int = 0
    _connected: bool = False

    async def connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        if self.csv_path is not None:
            self.csv_path = Path(self.csv_path)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["timestamp", "jaw_open", "head_lr", "head_ud", "wing"]
            )
        log.info("mock backend connected (csv=%s)", self.csv_path)

    async def send_frame(self, frame: MotionFrame) -> None:
        self._tick += 1
        if self.plot_after:
            self._frames.append(frame)
        if self._csv_writer is not None:
            self._csv_writer.writerow(
                [
                    f"{frame.timestamp:.4f}",
                    f"{frame.jaw_open:.4f}",
                    f"{frame.head_lr:.4f}",
                    f"{frame.head_ud:.4f}",
                    f"{frame.wing:.4f}",
                ]
            )
        if self.log_every_n > 0 and (self._tick % self.log_every_n == 0):
            log.info(
                "motion jaw=%.2f head_lr=%.2f head_ud=%.2f wing=%.2f",
                frame.jaw_open,
                frame.head_lr,
                frame.head_ud,
                frame.wing,
            )

    async def close(self) -> None:
        if not self._connected:
            return
        self._connected = False
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:  # noqa: BLE001
                pass
            self._csv_file = None
            self._csv_writer = None
        if self.plot_after and self._frames:
            self._render_plot()

    def _render_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            log.warning("matplotlib not installed; skipping plot")
            return
        if not self._frames:
            return
        t0 = self._frames[0].timestamp
        ts = [f.timestamp - t0 for f in self._frames]
        jaw = [f.jaw_open for f in self._frames]
        lr = [f.head_lr for f in self._frames]
        ud = [f.head_ud for f in self._frames]
        wing = [f.wing for f in self._frames]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, jaw, label="jaw_open")
        ax.plot(ts, lr, label="head_lr")
        ax.plot(ts, ud, label="head_ud")
        ax.plot(ts, wing, label="wing")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("normalized value")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right")
        ax.set_title("Motion channels (mock backend trace)")
        plt.tight_layout()
        plt.show()

    @property
    def is_connected(self) -> bool:
        return self._connected

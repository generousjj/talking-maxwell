"""Tiny, gentle hardware smoke test.

Verifies the serial handshake, effector registration, and a couple of slow
jaw openings. Run before the first audio replay to confirm wire protocol
works without generating noisy motion.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from motion.models import MotionFrame
from transport.bottango_serial_backend import (
    BottangoSerialBackend,
    BottangoServoConfig,
)


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config("config.example.yaml")
    s = config.bottango.serial

    backend = BottangoSerialBackend(
        port=s.port,
        baud=s.baud,
        allow_auto_detect=s.auto_detect,
        handshake_timeout_s=s.handshake_timeout_s,
        command_timeout_s=s.command_timeout_s,
        compressed_signal_max=s.compressed_signal_max,
        min_delta_for_send=0.0,  # force every frame through for this test
        jaw=BottangoServoConfig(
            pin=s.jaw.pin,
            min_pwm=s.jaw.min_pwm,
            max_pwm=s.jaw.max_pwm,
            max_pwm_per_sec=s.jaw.max_pwm_per_sec,
            starting_pwm=s.jaw.starting_pwm,
        ),
        head_lr=BottangoServoConfig(
            pin=s.head_lr.pin,
            min_pwm=s.head_lr.min_pwm,
            max_pwm=s.head_lr.max_pwm,
            max_pwm_per_sec=s.head_lr.max_pwm_per_sec,
            starting_pwm=s.head_lr.starting_pwm,
        ),
        head_ud=BottangoServoConfig(
            pin=s.head_ud.pin,
            min_pwm=s.head_ud.min_pwm,
            max_pwm=s.head_ud.max_pwm,
            max_pwm_per_sec=s.head_ud.max_pwm_per_sec,
            starting_pwm=s.head_ud.starting_pwm,
        ),
        wing=BottangoServoConfig(
            pin=s.wing.pin,
            min_pwm=s.wing.min_pwm,
            max_pwm=s.wing.max_pwm,
            max_pwm_per_sec=s.wing.max_pwm_per_sec,
            starting_pwm=s.wing.starting_pwm,
        ),
    )

    await backend.connect()
    try:
        logging.info("connected; centering, then slow jaw movements")
        await backend.send_frame(
            MotionFrame(jaw_open=0.0, head_lr=0.5, head_ud=0.5, wing=0.0)
        )
        await asyncio.sleep(1.0)

        # Slow jaw open + close, twice.
        steps = 40
        for cycle in range(2):
            for i in range(steps):
                v = i / (steps - 1)
                await backend.send_frame(
                    MotionFrame(jaw_open=v, head_lr=0.5, head_ud=0.5, wing=0.0)
                )
                await asyncio.sleep(1.2 / steps)
            for i in range(steps):
                v = 1.0 - i / (steps - 1)
                await backend.send_frame(
                    MotionFrame(jaw_open=v, head_lr=0.5, head_ud=0.5, wing=0.0)
                )
                await asyncio.sleep(1.2 / steps)

        logging.info("gentle head test")
        for i in range(30):
            t = i / 29.0
            await backend.send_frame(
                MotionFrame(
                    jaw_open=0.0,
                    head_lr=0.5 + 0.25 * (0.5 - abs(0.5 - t)),  # small L-R sweep
                    head_ud=0.5,
                    wing=0.0,
                )
            )
            await asyncio.sleep(2.0 / 30.0)

        await backend.send_frame(
            MotionFrame(jaw_open=0.0, head_lr=0.5, head_ud=0.5, wing=0.0)
        )
        await asyncio.sleep(0.7)
        logging.info("smoke test complete")
    finally:
        await backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

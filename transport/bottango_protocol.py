"""Bottango Arduino Driver wire protocol helpers.

This module implements just the pieces of the Bottango Driver API (API version 8,
documented in ``Readme_DriverAPI.pdf`` and verified against
``BottangoCore.cpp`` / ``BasicCommands.cpp`` on GitHub) that this project
needs to drive Maxwell:

- the handshake + time-sync lifecycle
- registering pin-controlled servos (``rSVPin``)
- issuing instant curves (``sCI``) to set a target position

Every command is terminated with ``\\n`` and every command includes a hash
parameter of the form ``,h<sum>`` where ``sum`` is the sum of ASCII codes of
every character preceding the ``,h`` separator. The firmware rejects commands
whose hash does not match.

Reference: https://github.com/EvanBottango/Bottango
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def compute_hash(command_body: str) -> int:
    """Compute the Bottango command hash.

    ``command_body`` is the command including its arguments but WITHOUT the
    trailing ``,h<sum>`` and without the newline terminator. The firmware
    algorithm sums the ASCII code of every character in this body.
    """
    return sum(ord(c) for c in command_body)


def frame_command(command_body: str) -> bytes:
    """Wrap a command body with its hash and terminator, ready for serial.

    Example::

        >>> frame_command("xUC,4")
        b'xUC,4,h368\\n'
    """
    hashed = f"{command_body},h{compute_hash(command_body)}\n"
    return hashed.encode("ascii")


# ---------- Command builders ----------


def cmd_handshake_request(random_code: int) -> bytes:
    """Build ``hRQ,<random>`` — sent by us to kick off the handshake."""
    return frame_command(f"hRQ,{int(random_code)}")


def cmd_time_sync(ms: int) -> bytes:
    """Build ``tSYN,<current_ms>`` — sync firmware clock to ours."""
    return frame_command(f"tSYN,{int(ms)}")


def cmd_register_pin_servo(
    *,
    pin: int,
    min_pwm: int,
    max_pwm: int,
    max_pwm_per_sec: int,
    starting_pwm: int,
) -> bytes:
    """Build ``rSVPin,<pin>,<min>,<max>,<maxPerSec>,<start>``."""
    return frame_command(
        "rSVPin,"
        f"{int(pin)},"
        f"{int(min_pwm)},"
        f"{int(max_pwm)},"
        f"{int(max_pwm_per_sec)},"
        f"{int(starting_pwm)}"
    )


def cmd_deregister_effector(identifier: str | int) -> bytes:
    return frame_command(f"xUE,{identifier}")


def cmd_deregister_all_effectors() -> bytes:
    return frame_command("xE")


def cmd_clear_all_curves() -> bytes:
    return frame_command("xC")


def cmd_stop() -> bytes:
    return frame_command("STOP")


def cmd_instant_curve(identifier: str | int, compressed_value: int) -> bytes:
    """Build ``sCI,<identifier>,<compressed>``.

    ``compressed_value`` must be in ``[0, COMPRESSED_SIGNAL_MAX]``. The
    firmware's default for recent versions is 8192, overridable in
    ``BottangoArduinoConfig.h``.
    """
    compressed_value = max(0, int(compressed_value))
    return frame_command(f"sCI,{identifier},{compressed_value}")


def normalized_to_compressed(value: float, signal_max: int = 8192) -> int:
    """Convert normalized [0, 1] to Bottango's compressed signal int."""
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0
    return int(round(value * signal_max))


# ---------- Tiny parse helpers used by the backend ----------


def parse_handshake_response(line: str) -> dict | None:
    """Parse a ``btngoHSK,<version>,<code>,<accepting>`` line."""
    parts = line.strip().split(",")
    if not parts or parts[0] != "btngoHSK":
        return None
    if len(parts) < 4:
        return None
    return {
        "version": parts[1],
        "random_code": parts[2],
        "accepting": parts[3] == "1",
    }


def is_ok(line: str) -> bool:
    """Return True if ``line`` is Bottango's ``OK`` readiness marker."""
    stripped = line.strip()
    return stripped == "OK"

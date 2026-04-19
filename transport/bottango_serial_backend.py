"""Bottango serial backend: talk directly to the ESP32 running BottangoArduinoDriver.

This is the preferred hardware path: we skip Bottango's desktop app entirely
and speak the firmware's serial protocol ourselves. That means no GUI setup
is required beyond the normal "upload BottangoArduinoDriver to the ESP32"
step that the user has already done.

Flow on connect:

    1. Open serial at the configured baud.
    2. Wait for ``BOOT`` (or send a wakeup handshake request on a short timeout).
    3. Send ``hRQ,<random>``.
    4. Expect ``btngoHSK,<version>,<random>,<accepting>`` followed by ``OK``.
    5. Send ``tSYN,<ms>``, expect ``OK``.
    6. Optionally deregister any stale effectors (``xE``), expect ``OK``.
    7. Register four pin servos (jaw/head_lr/head_ud/wing), each expecting ``OK``.

During runtime each frame we issue four ``sCI`` (instant curve) commands, one
per channel, serialized through a single background writer that waits for
``OK`` between commands as the firmware expects.

Safety rails:

    * ``min_pwm`` / ``max_pwm`` are honored at the firmware level so the
      servos physically cannot exceed their configured range.
    * ``max_pwm_per_sec`` caps slew rate, preventing sudden jerks even if
      we command big jumps.
    * An explicit ``safe_start`` boolean keeps us from registering / commanding
      if the config is incomplete.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    serial = None  # type: ignore

from motion.models import MotionFrame

from .base import MotionBackend
from .bottango_protocol import (
    cmd_clear_all_curves,
    cmd_deregister_all_effectors,
    cmd_handshake_request,
    cmd_instant_curve,
    cmd_register_pin_servo,
    cmd_stop,
    cmd_time_sync,
    is_ok,
    normalized_to_compressed,
    parse_handshake_response,
)

log = logging.getLogger(__name__)


@dataclass
class BottangoServoConfig:
    """Per-channel servo registration parameters.

    The defaults match the values recorded in the reference Maxwell Bottango
    project. Override any of these in ``config.yaml`` under
    ``bottango.serial.channels``.
    """

    pin: int
    min_pwm: int
    max_pwm: int
    max_pwm_per_sec: int = 2500
    starting_pwm: Optional[int] = None
    invert: bool = False
    """If True, a normalized value of 0.0 maps to max_pwm and 1.0 maps to
    min_pwm. Use when a servo is mounted backwards (e.g., the jaw closes
    when the code asks it to open)."""

    def resolved_start(self) -> int:
        if self.starting_pwm is not None:
            return int(self.starting_pwm)
        return int(round((self.min_pwm + self.max_pwm) / 2))


@dataclass
class BottangoSerialBackend(MotionBackend):
    """Motion backend that drives Maxwell's ESP32 directly over USB serial."""

    name: str = "bottango_serial"
    port: Optional[str] = None  # e.g. /dev/cu.usbmodem... or auto-detected
    baud: int = 115200
    handshake_timeout_s: float = 6.0
    command_timeout_s: float = 1.5
    compressed_signal_max: int = 8192
    allow_auto_detect: bool = True
    min_delta_for_send: float = 0.004
    """Default per-channel send threshold. Below this normalized delta we
    suppress the update. Jaw uses the larger ``jaw_min_delta`` below by
    default since a servo with a long extension wire interprets a flood
    of sub-percent changes as noise and stops tracking them cleanly."""

    jaw_min_delta: float = 0.020
    """Jaw-only send threshold. 2% of the normalized range. Anything smaller
    is treated as flutter and dropped — this matters a lot when the jaw
    servo has extension wires, because degraded signal integrity turns
    fast micro-updates into unreadable PWM."""

    jaw_min_send_interval_s: float = 0.08
    """Minimum time between jaw serial writes. Caps effective jaw update
    rate to ~12 Hz regardless of how fast the motion scheduler runs. The
    firmware interpolates between targets with ``max_pwm_per_sec``, so
    less-frequent but cleaner commands produce *smoother* physical motion
    on a long-wire servo, not jerkier motion."""

    jaw: BottangoServoConfig = field(
        default_factory=lambda: BottangoServoConfig(
            pin=9, min_pwm=1450, max_pwm=1775, max_pwm_per_sec=1200
        )
    )
    head_lr: BottangoServoConfig = field(
        default_factory=lambda: BottangoServoConfig(
            pin=5, min_pwm=1275, max_pwm=1725, max_pwm_per_sec=1800
        )
    )
    head_ud: BottangoServoConfig = field(
        default_factory=lambda: BottangoServoConfig(
            pin=6, min_pwm=850, max_pwm=2100, max_pwm_per_sec=1800
        )
    )
    wing: BottangoServoConfig = field(
        default_factory=lambda: BottangoServoConfig(
            pin=3, min_pwm=1500, max_pwm=2000, max_pwm_per_sec=3000
        )
    )

    _serial: Optional[object] = field(default=None, init=False)
    _writer_task: Optional[asyncio.Task] = field(default=None, init=False)
    _reader_task: Optional[asyncio.Task] = field(default=None, init=False)
    _ok_events: Optional[asyncio.Queue] = field(default=None, init=False)
    _connected: bool = field(default=False, init=False)
    _last_normalized: dict = field(default_factory=dict, init=False)
    _pending_frames: dict = field(default_factory=dict, init=False)
    _pending_signal: Optional[asyncio.Event] = field(default=None, init=False)
    _serial_lock: Optional[asyncio.Lock] = field(default=None, init=False)
    _hsk_line: Optional[str] = field(default=None, init=False)
    _hsk_event: Optional[asyncio.Event] = field(default=None, init=False)
    _channel_pins: dict = field(default_factory=dict, init=False)
    _channel_invert: dict = field(default_factory=dict, init=False)

    motion_intensity: float = 1.0
    """Global motion scale in [0.0, 1.0]. 1.0 = full motion, 0.5 = half
    deviation from rest pose, 0.0 = hold rest pose. Useful for testing
    without drawing as much current from an undersized servo supply."""

    _needs_reregister: Optional[asyncio.Event] = field(default=None, init=False)
    _recovery_task: Optional[asyncio.Task] = field(default=None, init=False)
    _recovering: bool = field(default=False, init=False)
    _missing_servo_count: int = field(default=0, init=False)

    _jaw_sent_count: int = field(default=0, init=False)
    _jaw_last_log_t: float = field(default=0.0, init=False)
    _jaw_last_min: float = field(default=1.0, init=False)
    _jaw_last_max: float = field(default=0.0, init=False)
    _jaw_last_send_t: float = field(default=0.0, init=False)
    _jaw_pending_value: Optional[float] = field(default=None, init=False)
    _jaw_peak_in_window: float = field(default=0.0, init=False)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _channels(self) -> list[tuple[str, BottangoServoConfig]]:
        return [
            ("jaw", self.jaw),
            ("head_lr", self.head_lr),
            ("head_ud", self.head_ud),
            ("wing", self.wing),
        ]

    async def connect(self) -> None:
        if self._connected:
            return
        if serial is None:
            raise RuntimeError(
                "pyserial is not installed. Run `pip install pyserial` to use "
                "the bottango_serial backend."
            )

        port = self.port or (
            _auto_detect_port() if self.allow_auto_detect else None
        )
        if not port:
            raise RuntimeError(
                "Could not locate a Bottango-compatible serial device. "
                "Plug in the ESP32 or set bottango.serial.port explicitly."
            )
        log.info("Opening Bottango serial port %s @ %d baud", port, self.baud)

        # pyserial open is synchronous and can block briefly. Run in a worker
        # so we don't stall the event loop if the OS enumerates slowly.
        def _open() -> "serial.Serial":
            # pyserial >= 3.x: open with a short read timeout for loops.
            return serial.Serial(
                port,
                self.baud,
                timeout=0.05,
                write_timeout=1.0,
                dsrdtr=False,
                rtscts=False,
            )

        self._serial = await asyncio.to_thread(_open)
        self.port = port

        # Many USB-serial bridges reset the target when DTR toggles, so give
        # the ESP32 a moment to (re)boot and print BOOT on its own.
        self._ok_events = asyncio.Queue(maxsize=64)
        self._pending_frames = {}
        self._pending_signal = asyncio.Event()
        self._serial_lock = asyncio.Lock()
        self._hsk_event = asyncio.Event()
        self._needs_reregister = asyncio.Event()
        self._reader_task = asyncio.create_task(
            self._reader_loop(), name="bottango-serial-reader"
        )

        try:
            await self._handshake()
            await self._register_all_servos(reason="initial")
        except Exception:
            await self.close()
            raise

        self._writer_task = asyncio.create_task(
            self._writer_loop(), name="bottango-serial-writer"
        )
        self._recovery_task = asyncio.create_task(
            self._recovery_loop(), name="bottango-serial-recovery"
        )
        self._connected = True

    async def _register_all_servos(self, *, reason: str) -> None:
        """Send time-sync, clear, and re-register every servo channel.

        Called from :meth:`connect` and from the recovery loop whenever the
        firmware resets mid-session (e.g. brownout from servo inrush current).
        """
        log.info("registering servos (%s)", reason)
        await self._send_and_wait_ok(cmd_time_sync(int(time.monotonic() * 1000)))
        await self._send_and_wait_ok(cmd_deregister_all_effectors())
        await self._send_and_wait_ok(cmd_clear_all_curves())
        for label, cfg in self._channels():
            await self._send_and_wait_ok(
                cmd_register_pin_servo(
                    pin=cfg.pin,
                    min_pwm=cfg.min_pwm,
                    max_pwm=cfg.max_pwm,
                    max_pwm_per_sec=cfg.max_pwm_per_sec,
                    starting_pwm=cfg.resolved_start(),
                )
            )
            self._channel_pins[label] = cfg.pin
            self._channel_invert[label] = bool(cfg.invert)
            log.info(
                "registered %s -> pin %d (pwm %d-%d, start %d%s)",
                label,
                cfg.pin,
                cfg.min_pwm,
                cfg.max_pwm,
                cfg.resolved_start(),
                ", INVERTED" if cfg.invert else "",
            )
        # Force the next frame to re-send all channel values even if they
        # haven't changed, since the firmware just came back with defaults.
        self._last_normalized.clear()

    async def send_frame(self, frame: MotionFrame) -> None:
        if not self._connected or self._pending_signal is None:
            return
        if self._recovering:
            return
        scale = max(0.0, min(1.0, float(self.motion_intensity)))
        values = {
            "jaw": frame.jaw_open * scale,
            "head_lr": 0.5 + (frame.head_lr - 0.5) * scale,
            "head_ud": 0.5 + (frame.head_ud - 0.5) * scale,
            "wing": frame.wing * scale,
        }
        dirty = False
        now = time.monotonic()
        for label, value in values.items():
            # Per-channel send threshold: jaw gets a bigger dead-band because
            # its servo typically runs through extended signal/power wires,
            # and flooding a long line with sub-percent updates causes the
            # servo to read garbage and simply stop moving. Head/wing run
            # over short wires from the same ESP32 and stay responsive at
            # the tight default threshold.
            delta_threshold = (
                self.jaw_min_delta if label == "jaw" else self.min_delta_for_send
            )
            prev = self._last_normalized.get(label)
            if prev is not None and abs(prev - value) < delta_threshold:
                continue
            # Per-channel rate limit: cap jaw serial writes regardless of
            # scheduler rate. The firmware interpolates with ``max_pwm_per_sec``,
            # so rarer-but-cleaner commands produce smoother physical motion
            # on a long-wire servo than a stream of direction reversals.
            #
            # Peak-preserving downsample: during the send-cooldown window we
            # keep a running maximum of the envelope. When the window ends we
            # emit THAT max (not whatever the envelope happens to be at the
            # instant we're allowed to send). This preserves the "fully open"
            # peaks that would otherwise be missed between rate-limited sends
            # — critical for jaw expressiveness.
            if label == "jaw":
                self._jaw_peak_in_window = max(self._jaw_peak_in_window, value)
                if now - self._jaw_last_send_t < self.jaw_min_send_interval_s:
                    self._jaw_pending_value = value
                    continue
                value = self._jaw_peak_in_window
                self._jaw_peak_in_window = 0.0
                self._jaw_last_send_t = now
                self._jaw_pending_value = None
            self._last_normalized[label] = value
            pin = self._channel_pins.get(label)
            if pin is None:
                continue
            effective = 1.0 - value if self._channel_invert.get(label) else value
            compressed = normalized_to_compressed(
                effective, self.compressed_signal_max
            )
            self._pending_frames[label] = cmd_instant_curve(pin, compressed)
            dirty = True
            if label == "jaw":
                self._jaw_sent_count += 1
                self._jaw_last_min = min(self._jaw_last_min, value)
                self._jaw_last_max = max(self._jaw_last_max, value)
                if now - self._jaw_last_log_t > 1.0:
                    log.info(
                        "jaw trace: %d frames/s, value range [%.2f, %.2f], "
                        "last effective=%.2f compressed=%d (invert=%s)",
                        self._jaw_sent_count,
                        self._jaw_last_min,
                        self._jaw_last_max,
                        effective,
                        compressed,
                        self._channel_invert.get(label),
                    )
                    self._jaw_sent_count = 0
                    self._jaw_last_min = 1.0
                    self._jaw_last_max = 0.0
                    self._jaw_last_log_t = now
        if dirty:
            self._pending_signal.set()

    async def jaw_hammer(
        self, *, cycles: int = 10, period_s: float = 0.12
    ) -> None:
        """Aggressively toggle the jaw between extremes to reseat a marginal
        contact on pin 9 (or wake a servo that's stopped responding).

        This is the software equivalent of physically wiggling the jaw
        signal wire — dense, fast PWM edges on the channel are the best
        shot we have at bridging a flaky dupont connection without user
        intervention. Bypasses the usual jaw rate-limit / delta filters so
        every transition actually lands. The head and wing channels aren't
        touched so the bird stays still otherwise.
        """
        if not self._connected:
            return
        pin = self._channel_pins.get("jaw")
        if pin is None:
            log.warning("jaw hammer: jaw pin not registered yet")
            return
        invert = bool(self._channel_invert.get("jaw"))
        log.info(
            "jaw hammer: %d cycles at %.0f ms period on pin %d",
            cycles,
            period_s * 1000,
            pin,
        )
        for i in range(cycles):
            for target in (1.0, 0.0):
                effective = 1.0 - target if invert else target
                payload = cmd_instant_curve(
                    pin,
                    normalized_to_compressed(effective, self.compressed_signal_max),
                )
                self._pending_frames["jaw"] = payload
                self._last_normalized["jaw"] = target
                if self._pending_signal is not None:
                    self._pending_signal.set()
                await asyncio.sleep(period_s)
        # Park at closed.
        park_effective = 1.0 if invert else 0.0
        self._pending_frames["jaw"] = cmd_instant_curve(
            pin,
            normalized_to_compressed(park_effective, self.compressed_signal_max),
        )
        self._last_normalized["jaw"] = 0.0
        if self._pending_signal is not None:
            self._pending_signal.set()
        # Reset throttle bookkeeping so the next speech frame is free to send.
        self._jaw_last_send_t = 0.0
        self._jaw_peak_in_window = 0.0
        log.info("jaw hammer: done")

    async def full_reset(self) -> None:
        """Hardest software reset: deregister all effectors, re-register
        them, then hammer the jaw and wake-sweep all servos.

        Use this as the first recovery step when the jaw goes silent
        mid-session. It's the closest we can get to "physically unplug and
        replug GPIO 9" in pure software.
        """
        if not self._connected:
            return
        log.info("full reset: deregister + re-register + wake")
        try:
            await self._register_all_servos(reason="full_reset")
        except Exception:  # noqa: BLE001
            log.exception("full reset: re-register failed")
            return
        await self.jaw_hammer()
        await self.wake_sweep()

    async def wake_sweep(self, *, hold_s: float = 0.45) -> None:
        """Sweep every servo through min → max → mid to prove it's alive.

        Useful on startup (or after the user reseats a signal wire) because
        a flaky dupont contact often only conducts PWM reliably after the
        servo has physically moved through its full range once. This also
        makes it obvious at-a-glance if any channel is dead before speech
        starts. Writes bypass the per-channel rate/delta filters — we want
        these specific frames to land.
        """
        if not self._connected:
            return
        log.info("wake sweep: exercising all servos")
        sequence = [
            # jaw_open, head_lr, head_ud, wing
            (0.0, 0.5, 0.5, 0.0),
            (1.0, 0.2, 0.2, 0.9),
            (0.0, 0.8, 0.8, 0.0),
            (1.0, 0.5, 0.5, 0.5),
            (0.0, 0.5, 0.5, 0.0),
        ]
        for jaw_open, head_lr, head_ud, wing in sequence:
            await self.send_frame(
                MotionFrame(
                    jaw_open=jaw_open,
                    head_lr=head_lr,
                    head_ud=head_ud,
                    wing=wing,
                )
            )
            await asyncio.sleep(hold_s)
        # Return to rest pose.
        await self.send_frame(
            MotionFrame(jaw_open=0.0, head_lr=0.5, head_ud=0.5, wing=0.0)
        )
        log.info("wake sweep: done")

    async def close(self) -> None:
        if not self._connected and self._serial is None:
            return
        self._connected = False

        tasks = (self._writer_task, self._reader_task, self._recovery_task)
        for task in tasks:
            if task is None:
                continue
            task.cancel()
        for task in tasks:
            if task is None:
                continue
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._writer_task = None
        self._reader_task = None
        self._recovery_task = None

        if self._serial is not None:
            try:
                # Best-effort graceful stop so effectors relax cleanly.
                await asyncio.to_thread(self._serial.write, cmd_stop())  # type: ignore[arg-type]
                await asyncio.to_thread(self._serial.flush)
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.to_thread(self._serial.close)
            except Exception:  # noqa: BLE001
                pass
            self._serial = None

    # ------------------------- internals -------------------------

    async def _handshake(self) -> None:
        random_code = random.randint(100, 99999)
        # Send handshake request up to a few times; if the ESP32 just booted
        # we may have missed its BOOT banner.
        start = time.monotonic()
        attempts = 0
        while (time.monotonic() - start) < self.handshake_timeout_s:
            attempts += 1
            log.debug("sending hRQ,%d (attempt %d)", random_code, attempts)
            await self._send_bytes(cmd_handshake_request(random_code))
            try:
                await asyncio.wait_for(self._hsk_event.wait(), timeout=1.0)  # type: ignore[union-attr]
                break
            except asyncio.TimeoutError:
                continue
        if self._hsk_line is None:
            raise RuntimeError(
                "No handshake response from Bottango firmware on "
                f"{self.port} after {self.handshake_timeout_s:.1f}s."
            )
        parsed = parse_handshake_response(self._hsk_line)
        if not parsed:
            raise RuntimeError(
                f"Unrecognized handshake response: {self._hsk_line!r}"
            )
        if parsed["random_code"] != str(random_code):
            log.warning(
                "Handshake random code mismatch (sent %s, got %s) — continuing",
                random_code,
                parsed["random_code"],
            )
        if not parsed["accepting"]:
            raise RuntimeError(
                "Firmware reports offline-playback mode (accepting=0); "
                "it will not respond to external commands. Re-flash with "
                "the default BottangoArduinoDriver sketch."
            )
        log.info(
            "Bottango firmware version %s acknowledged handshake", parsed["version"]
        )
        # Wait for the OK that follows the handshake response.
        try:
            await asyncio.wait_for(
                self._ok_events.get(), timeout=self.command_timeout_s  # type: ignore[union-attr]
            )
        except asyncio.TimeoutError:
            log.warning("no OK after handshake; continuing anyway")

    async def _send_and_wait_ok(self, payload: bytes) -> None:
        # Drain any OKs that piled up from fire-and-forget motion frames
        # before we wait for the OK matching *this* command.
        if self._ok_events is not None:
            while not self._ok_events.empty():
                try:
                    self._ok_events.get_nowait()
                except asyncio.QueueEmpty:
                    break
        await self._send_bytes(payload)
        try:
            await asyncio.wait_for(
                self._ok_events.get(), timeout=self.command_timeout_s  # type: ignore[union-attr]
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Timed out waiting for OK after {payload!r}"
            ) from exc

    async def _send_bytes(self, payload: bytes) -> None:
        assert self._serial is not None
        log.debug("-> %s", payload.rstrip().decode(errors="replace"))

        def _write() -> None:
            self._serial.write(payload)  # type: ignore[union-attr]

        # Single thread hop per command + no explicit flush: pyserial buffers
        # ~14-byte motion commands into OS serial TX immediately at 115200,
        # and keeping this hot path cheap lets the coalescing writer keep up
        # with 30 Hz × 4 channel motion without lag.
        if self._serial_lock is not None:
            async with self._serial_lock:
                await asyncio.to_thread(_write)
        else:
            await asyncio.to_thread(_write)

    async def _reader_loop(self) -> None:
        assert self._serial is not None
        buffer = bytearray()
        try:
            while True:
                try:
                    chunk = await asyncio.to_thread(self._serial.read, 256)  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(0.05)
                    continue
                if not chunk:
                    await asyncio.sleep(0.01)
                    continue
                buffer.extend(chunk)
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    text = line.decode("ascii", errors="replace").strip()
                    if not text:
                        continue
                    log.debug("<- %s", text)
                    if text == "BOOT":
                        # The ESP32 printed BOOT a second time, i.e. it just
                        # reset (usually a brownout from servo inrush). Kick
                        # the recovery loop to redo handshake + registration.
                        if self._connected and self._needs_reregister is not None:
                            log.warning(
                                "firmware reset detected (BOOT); "
                                "will re-handshake and re-register servos"
                            )
                            self._needs_reregister.set()
                        continue
                    if is_ok(text):
                        if self._ok_events is not None:
                            # Bounded: drop oldest on overflow. During motion
                            # we fire-and-forget so OKs pile up otherwise.
                            if self._ok_events.full():
                                try:
                                    self._ok_events.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                            try:
                                self._ok_events.put_nowait(True)
                            except asyncio.QueueFull:
                                pass
                        continue
                    if text.startswith("btngoHSK"):
                        self._hsk_line = text
                        if self._hsk_event is not None:
                            self._hsk_event.set()
                        continue
                    if text == "HASH_FAIL":
                        log.error("firmware reported HASH_FAIL — aborting")
                        if self._ok_events is not None:
                            self._ok_events.put_nowait(False)
                        continue
                    if text == "TIMEOUT":
                        log.warning("firmware reported TIMEOUT on last command")
                        continue
                    if "errNoServoOnPin" in text:
                        # Either the firmware rebooted without printing BOOT
                        # (rare) or a servo genuinely was never registered.
                        # Treat as a prompt to re-register after a handful of
                        # consecutive misses so we don't thrash on a single
                        # dropped pin.
                        self._missing_servo_count += 1
                        if (
                            self._missing_servo_count >= 3
                            and self._connected
                            and self._needs_reregister is not None
                            and not self._recovering
                        ):
                            log.warning(
                                "firmware keeps reporting errNoServoOnPin; "
                                "triggering re-registration"
                            )
                            self._needs_reregister.set()
                            self._missing_servo_count = 0
                        continue
                    # Any other firmware line counts as "it's alive" — reset
                    # the missing-servo counter so transient misses don't stack.
                    self._missing_servo_count = 0
                    log.info("firmware: %s", text)
        except asyncio.CancelledError:
            return

    async def _writer_loop(self) -> None:
        """Drain coalesced motion frames to the serial port, fire-and-forget.

        Motion commands are sent without waiting for per-command ``OK`` because
        the firmware OK roundtrip (bounded by the serial read-poll timeout)
        caps throughput to ~20 cmd/s — far below the 120 cmd/s generated by a
        30 Hz × 4-channel scheduler. Building up a queue caused several
        seconds of visible lag between audio and motion. Dropping stale
        per-channel values keeps motion tight to the current audio envelope
        even if the firmware is briefly slow. Registration commands still use
        ``_send_and_wait_ok`` where correctness matters.
        """
        assert self._pending_signal is not None
        try:
            while True:
                await self._pending_signal.wait()
                self._pending_signal.clear()
                if self._recovering:
                    self._pending_frames.clear()
                    continue
                snapshot = list(self._pending_frames.items())
                self._pending_frames.clear()
                for _label, payload in snapshot:
                    await self._send_bytes(payload)
        except asyncio.CancelledError:
            return

    async def _recovery_loop(self) -> None:
        """Re-handshake and re-register servos whenever the firmware resets.

        The ESP32 can reboot mid-session if the servo rail browns out on a
        current spike. When that happens, the reader loop sets
        ``_needs_reregister``; we drain any stale pending frames, redo the
        handshake, and re-register every servo so motion can resume without
        requiring the user to restart the Python process.
        """
        assert self._needs_reregister is not None
        try:
            while True:
                await self._needs_reregister.wait()
                self._needs_reregister.clear()
                self._recovering = True
                log.info("recovery: flushing pending frames and re-handshaking")
                try:
                    self._pending_frames.clear()
                    if self._pending_signal is not None:
                        self._pending_signal.clear()
                    while self._ok_events and not self._ok_events.empty():
                        try:
                            self._ok_events.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    if self._hsk_event is not None:
                        self._hsk_event.clear()
                    self._hsk_line = None
                    await asyncio.sleep(0.3)
                    await self._handshake()
                    await self._register_all_servos(reason="recovery")
                    log.info("recovery: complete, motion resuming")
                except Exception as exc:  # noqa: BLE001
                    log.error("recovery failed: %s", exc)
                    await asyncio.sleep(1.0)
                finally:
                    self._recovering = False
        except asyncio.CancelledError:
            return


def _auto_detect_port() -> Optional[str]:
    """Return the first serial port that looks like a likely ESP32 board."""
    if serial is None:
        return None
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None

    def score(info) -> int:
        device = (info.device or "").lower()
        description = (info.description or "").lower()
        manufacturer = (info.manufacturer or "").lower()
        s = 0
        if "usbmodem" in device or "usbserial" in device:
            s += 3
        if "silicon labs" in description or "cp210" in description:
            s += 3
        if "esp" in description or "esp" in manufacturer:
            s += 4
        if "wch" in description:
            s += 2
        if "ftdi" in description:
            s += 2
        if "/dev/cu." in device:
            s += 1
        return s

    ports.sort(key=score, reverse=True)
    best = ports[0]
    if score(best) <= 0:
        return None
    log.info(
        "auto-detected serial port %s (%s / %s)",
        best.device,
        best.description,
        best.manufacturer,
    )
    return best.device

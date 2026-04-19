"""Audio I/O utilities: playback with live envelope streaming and mic capture.

This module is intentionally small. It wraps ``sounddevice`` so the rest of the
app can stay provider-agnostic. If ``sounddevice`` is not available (for
example, on a headless CI machine), the helpers degrade gracefully: playback
becomes a virtual "clock" so envelopes can still be computed from the PCM
buffer for offline testing.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:  # optional at import time; used for playback and capture
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    sd = None  # type: ignore

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    sf = None  # type: ignore

log = logging.getLogger(__name__)


@dataclass
class AudioBuffer:
    """A mono float32 audio buffer with sample rate metadata."""

    samples: np.ndarray  # shape (N,), float32, mono, range [-1, 1]
    sample_rate: int

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self.samples.shape[0]) / float(self.sample_rate)


def decode_wav_bytes(data: bytes) -> AudioBuffer:
    """Decode WAV bytes to a mono float32 AudioBuffer using the stdlib."""
    if sf is not None:
        try:
            bio = io.BytesIO(data)
            samples, sr = sf.read(bio, dtype="float32", always_2d=False)
            if samples.ndim == 2:
                samples = samples.mean(axis=1)
            return AudioBuffer(
                samples=np.ascontiguousarray(samples.astype(np.float32)),
                sample_rate=int(sr),
            )
        except Exception:  # noqa: BLE001 - fall through to stdlib path
            log.debug("soundfile decode failed, falling back to wave", exc_info=True)

    with wave.open(io.BytesIO(data), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(2**31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return AudioBuffer(samples=np.ascontiguousarray(arr.astype(np.float32)), sample_rate=int(sr))


def load_wav_file(path: str) -> AudioBuffer:
    with open(path, "rb") as fh:
        return decode_wav_bytes(fh.read())


def resolve_output_device(device: Optional[str | int]) -> Optional[int]:
    """Resolve a playback-device config value to a concrete sounddevice index.

    ``device`` may be ``None`` (use system default), an int index, or a
    substring of the device name (case-insensitive, e.g. "MacBook Pro Speakers").
    Returns the resolved index, or ``None`` to mean "use default".
    """
    if device is None or sd is None:
        return None
    if isinstance(device, int):
        return device
    text = str(device).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        devices = sd.query_devices()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not query audio devices: %s", exc)
        return None
    needle = text.lower()
    for i, d in enumerate(devices):
        if int(d.get("max_output_channels", 0)) <= 0:
            continue
        if needle in str(d.get("name", "")).lower():
            return i
    log.warning(
        "audio output device %r not found; falling back to system default", text
    )
    return None


async def play_buffer_with_envelope(
    buffer: AudioBuffer,
    *,
    envelope_callback: Callable[[float, float], None],
    frame_ms: float = 20.0,
    block_ms: float = 40.0,
    output_device: Optional[str | int] = None,
) -> None:
    """Play an AudioBuffer while streaming envelope progress to a callback.

    ``envelope_callback`` is called with ``(progress_0_to_1, envelope_rms)``
    at roughly ``frame_ms`` resolution. ``envelope_rms`` is the raw RMS of the
    current window (range [0, 1]); smoothing / jaw mapping happens in the
    caller's EnvelopeFollower so motion can stay in one place.
    """
    if buffer.samples.size == 0:
        return

    frame_samples = max(1, int(buffer.sample_rate * frame_ms / 1000.0))
    total_frames = buffer.samples.shape[0]

    start = time.monotonic()
    playback_started = _start_playback(buffer, output_device=output_device)

    try:
        # Walk the buffer in real time. We rely on sounddevice for actual audio
        # output and use monotonic time to advance the envelope read position,
        # which keeps motion tightly aligned with what the user hears.
        cursor = 0
        while cursor < total_frames:
            now = time.monotonic()
            playback_pos = int((now - start) * buffer.sample_rate)
            target = min(total_frames, max(cursor + frame_samples, playback_pos))
            window = buffer.samples[cursor:target]
            if window.size == 0:
                await asyncio.sleep(frame_ms / 1000.0 / 2)
                continue
            rms = float(np.sqrt(np.mean(np.square(window))))
            progress = target / total_frames
            envelope_callback(progress, rms)
            cursor = target
            elapsed = time.monotonic() - start
            target_time = cursor / buffer.sample_rate
            sleep_for = max(0.0, target_time - elapsed)
            await asyncio.sleep(min(sleep_for, frame_ms / 1000.0))
        envelope_callback(1.0, 0.0)
    finally:
        if playback_started and sd is not None:
            try:
                sd.wait()
            except Exception:  # noqa: BLE001
                pass


def _start_playback(
    buffer: AudioBuffer, *, output_device: Optional[str | int] = None
) -> bool:
    if sd is None:
        log.warning(
            "sounddevice not installed; audio will be simulated silently for motion preview"
        )
        return False
    device_idx = resolve_output_device(output_device)
    try:
        sd.stop()
        kwargs = {}
        if device_idx is not None:
            kwargs["device"] = device_idx
        sd.play(buffer.samples, samplerate=buffer.sample_rate, **kwargs)
        if device_idx is not None:
            try:
                info = sd.query_devices(device_idx)
                log.info("playing audio on '%s'", info.get("name", device_idx))
            except Exception:  # noqa: BLE001
                pass
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("audio playback failed: %s", exc)
        return False


@dataclass
class CapturedAudio:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return float(self.samples.shape[0]) / float(self.sample_rate or 1)


async def record_from_mic(
    *,
    sample_rate: int = 16000,
    max_duration_s: float = 15.0,
    silence_threshold: float = 0.012,
    silence_hangover_s: float = 1.2,
    device: Optional[str | int] = None,
) -> CapturedAudio:
    """Record from the default microphone with simple energy-based VAD.

    Returns once either ``max_duration_s`` elapses or the user has been silent
    for ``silence_hangover_s`` after speaking. This is intentionally dumb but
    works well for a prototype.
    """
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed; microphone capture is unavailable. "
            "Install it with `pip install sounddevice` or use typed-text mode."
        )

    frames: list[np.ndarray] = []
    speaking_seen = False
    silence_for = 0.0
    start = time.monotonic()
    chunk_ms = 50
    chunk_samples = int(sample_rate * chunk_ms / 1000)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    def _callback(indata, frame_count, time_info, status):  # type: ignore[override]
        if status:
            log.debug("mic status: %s", status)
        loop.call_soon_threadsafe(queue.put_nowait, indata.copy().squeeze())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
        device=device,
        callback=_callback,
    )
    with stream:
        while True:
            if (time.monotonic() - start) > max_duration_s:
                break
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            frames.append(chunk)
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            if rms > silence_threshold:
                speaking_seen = True
                silence_for = 0.0
            else:
                silence_for += chunk_ms / 1000.0
                if speaking_seen and silence_for >= silence_hangover_s:
                    break

    if not frames:
        return CapturedAudio(samples=np.zeros(0, dtype=np.float32), sample_rate=sample_rate)
    return CapturedAudio(
        samples=np.concatenate(frames).astype(np.float32),
        sample_rate=sample_rate,
    )


def encode_wav_bytes(buffer: AudioBuffer) -> bytes:
    """Encode an AudioBuffer as 16-bit PCM WAV bytes (for STT upload)."""
    pcm16 = np.clip(buffer.samples, -1.0, 1.0) * 32767.0
    pcm16 = pcm16.astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(buffer.sample_rate)
        wav.writeframes(pcm16.tobytes())
    return bio.getvalue()

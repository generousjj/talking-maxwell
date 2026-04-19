"""OpenAI Realtime API session for low-latency speech-in / speech-out.

This wraps :class:`openai.AsyncOpenAI`'s ``beta.realtime`` client into a
small, self-contained session that the webapp / pipeline can start and
stop on demand. While a session is open, mic audio streams up to OpenAI
at 24 kHz PCM16, the model's audio response streams back, and we feed
both an audio-output stream (for the user to hear) and an envelope
callback (so jaw motion mirrors what the speaker is saying).

Lessons from the previous iteration baked into this design:

* ``start()`` and ``stop()`` are idempotent — calling start twice closes
  the existing session before opening a new one, calling stop when
  nothing is running is a no-op. The webapp leans on this so a user
  double-clicking the toggle button can never wedge us.
* No "STRICT VOICE RULES" prefix is hard-coded into the code path. The
  caller passes ``instructions`` cleanly through the SDK's session
  config; if you want personality tweaks, edit ``config.yaml``.
* Server VAD does turn detection. We don't try to second-guess it from
  the client — that was a major source of dropouts last time.
* The envelope callback uses the same ``EnvelopeFollower`` /
  ``LiveSpeakingContext`` plumbing as the regular ``say()`` path so jaw
  motion stays tight to what the speaker is saying with no second motion
  code path to maintain.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import numpy as np

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # type: ignore

try:
    from openai import AsyncOpenAI  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "openai>=1.40 is required for realtime mode. "
        "Install with `pip install -U openai`."
    ) from exc

log = logging.getLogger(__name__)


REALTIME_SAMPLE_RATE = 24_000
"""OpenAI Realtime's native PCM16 sample rate. Using it for both input
and output dodges any resampling on our side."""

INPUT_BLOCK_MS = 40
"""How often we pump mic audio up to the server. 40 ms keeps the input
audio buffer fresh enough for the server-side VAD to react quickly
without flooding the socket with sub-millisecond chunks."""

OUTPUT_BLOCK_MS = 40
"""Per-write block size for the audio output stream. Matches the input
block size so the playback callback stays steady."""


def _resample_float(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linearly resample a 1-D float audio buffer.

    Quality-good-enough for speech in both directions (24 kHz <-> 48 kHz
    is the common case here). Avoids a scipy dependency. Returns
    a fresh float32 array.
    """
    if src_sr == dst_sr or arr.size == 0:
        return arr.astype(np.float32, copy=False)
    n_in = arr.size
    n_out = max(1, int(round(n_in * dst_sr / src_sr)))
    xp = np.linspace(0.0, 1.0, num=n_in, endpoint=False, dtype=np.float64)
    x = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(x, xp, arr.astype(np.float64)).astype(np.float32)


def _resample_int16_bytes(data: bytes, src_sr: int, dst_sr: int) -> bytes:
    """Resample a raw PCM16 mono buffer (bytes) between two rates.

    Used in the mic callback to convert the device's native rate to
    OpenAI's expected 24 kHz. Linear interp; sufficient for speech.
    """
    if src_sr == dst_sr or not data:
        return data
    arr = np.frombuffer(data, dtype=np.int16)
    out = _resample_float(arr.astype(np.float32), src_sr, dst_sr)
    return np.clip(out, -32768.0, 32767.0).astype(np.int16).tobytes()


# ----------------------------------------------------------------------
# Event-shape helpers
#
# The Realtime API returns events as objects with a ``type`` attribute.
# We pull the small handful we care about into a minimal classifier so
# the rest of the file (and tests) can switch on a stable enum-like
# value instead of poking at OpenAI SDK internals.
# ----------------------------------------------------------------------

EVENT_AUDIO_DELTA = "response.audio.delta"
EVENT_RESPONSE_DONE = "response.done"
EVENT_SPEECH_STARTED = "input_audio_buffer.speech_started"
EVENT_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
EVENT_RESPONSE_CREATED = "response.created"
EVENT_ERROR = "error"
EVENT_USER_TRANSCRIPT = "conversation.item.input_audio_transcription.completed"
EVENT_ASSISTANT_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
EVENT_ASSISTANT_TRANSCRIPT_DONE = "response.audio_transcript.done"


def _event_attr(event: object, name: str) -> Any:
    """Pull ``name`` off either a dict-shaped or SDK-object event."""
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def classify_event(event: object) -> Optional[str]:
    """Return the canonical type of a Realtime event, or ``None``.

    Used by tests so we don't need a live SDK to validate the dispatch.
    Accepts dict-shaped events (raw JSON form) and SDK objects with a
    ``type`` attribute.
    """
    if isinstance(event, dict):
        return event.get("type")
    return getattr(event, "type", None)


def decode_audio_delta(event: object) -> Optional[np.ndarray]:
    """Pull PCM16 samples out of a ``response.audio.delta`` event.

    Returns a float32 ``np.ndarray`` in [-1, 1] (sounddevice's preferred
    format) or ``None`` if there's no audio payload.
    """
    if isinstance(event, dict):
        delta = event.get("delta")
    else:
        delta = getattr(event, "delta", None)
    if not delta:
        return None
    raw = base64.b64decode(delta)
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    if pcm16.size == 0:
        return None
    return pcm16.astype(np.float32) / 32768.0


# ----------------------------------------------------------------------
# RealtimeSession
# ----------------------------------------------------------------------


EnvelopeCallback = Callable[[float], None]
TranscriptCallback = Callable[[str, str], None]
"""Called as ``transcript_callback(role, text)`` whenever a finalized
user or assistant transcript is available. ``role`` is ``"user"`` or
``"maxwell"``. Synchronous so it can be used to push into a deque
without forcing the caller into async."""
"""Called per output audio block with the RMS of the block in [0, 1]."""

StateCallback = Callable[[str], Awaitable[None]]
"""Called with one of: ``"listening"``, ``"thinking"``, ``"speaking"``,
``"idle"`` whenever the conversation moves between phases. Used by the
pipeline to drive the same ``ConversationStateMachine`` as the typed /
live paths so the behavior engine stays in sync."""


@dataclass
class RealtimeSession:
    api_key: str
    model: str = "gpt-realtime"
    voice: str = "ballad"
    instructions: str = ""
    input_device: Optional[str | int] = None
    output_device: Optional[str | int] = None
    envelope_callback: Optional[EnvelopeCallback] = None
    state_callback: Optional[StateCallback] = None
    transcript_callback: Optional[TranscriptCallback] = None

    # VAD configuration. Defaults are tuned for a noisy laptop-mic
    # environment (events, fairs, demo tables): server-side far-field
    # noise reduction + a higher activation threshold + a longer
    # trailing-silence window so background chatter doesn't constantly
    # trigger turn-detection.
    vad_type: str = "server_vad"  # "server_vad" | "semantic_vad"
    vad_threshold: float = 0.7
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 700
    vad_eagerness: str = "low"
    noise_reduction: str = "far_field"  # "off" | "near_field" | "far_field"

    # Half-duplex echo-suppression. OpenAI Realtime ships with no AEC
    # of its own, so when the bird's speakers are anywhere near the
    # laptop mic the server-side VAD just keeps re-triggering on
    # Maxwell's own voice and he runs away talking to himself. The
    # pragmatic fix is to stop streaming mic audio up while playback is
    # in flight, plus a short tail after the last audio chunk to swallow
    # speaker reverb. Set ``half_duplex=False`` only if you're using a
    # headset / AirPods where the mic genuinely doesn't hear playback
    # and you want to preserve barge-in.
    half_duplex: bool = True
    playback_tail_ms: int = 400

    # Smart barge-in: while half-duplex has the mic muted, watch the
    # mic's local RMS. If the user is clearly louder than the speaker
    # leakage for a few consecutive frames, un-mute and cancel
    # Maxwell's in-flight response so they can interrupt naturally.
    # Without this, half-duplex is "wait your turn" which kills the
    # feel of a real-time conversation.
    barge_in_enabled: bool = True
    barge_in_rms_threshold: float = 0.06
    """Absolute mic RMS (0..1, on int16 normalized to float) that has to
    be exceeded for a frame to count as "user is talking". 0.06 is
    around normal speaking volume into a laptop mic."""
    barge_in_above_ambient_factor: float = 5.0
    """Multiplier on top of the recent ambient floor; mic RMS must
    *also* exceed ambient * this factor. Stops bleed-back from the
    speaker (which raises ambient too) from triggering barge-in."""
    barge_in_min_frames: int = 4
    """Number of consecutive loud mic frames required before triggering
    barge-in. With 40 ms input chunks, 4 = 160 ms — long enough to
    skip a single loud syllable from playback, short enough to feel
    snappy when the user really starts talking."""

    # Push-to-talk: when True, server VAD is disabled and the mic only
    # streams to the server while the caller has explicitly held the
    # PTT key (``ptt_down`` -> ``ptt_up``). Releasing PTT commits the
    # buffer and asks the server for a response. Switching this on at
    # runtime is supported via ``set_push_to_talk``; the next session
    # update reconfigures turn detection accordingly. Useful for noisy
    # rooms where any VAD setup would false-trigger.
    push_to_talk: bool = False

    _client: Optional[AsyncOpenAI] = field(default=None, init=False)
    _conn: Optional[object] = field(default=None, init=False)
    _conn_ctx: Optional[object] = field(default=None, init=False)
    _input_stream: Optional["sd.RawInputStream"] = field(default=None, init=False)
    _output_stream: Optional["sd.RawOutputStream"] = field(default=None, init=False)
    _output_queue: Optional[asyncio.Queue] = field(default=None, init=False)
    # Native sample rates of the actually-opened sounddevice streams.
    # Some macOS inputs (built-in mic, AirPods) reject 24 kHz directly
    # via PortAudio (CoreAudio AUHAL returns -10851), so we fall back
    # to the device's native rate and resample on the fly to/from the
    # 24 kHz the OpenAI Realtime API expects.
    _mic_sample_rate: int = field(default=REALTIME_SAMPLE_RATE, init=False)
    _output_sample_rate: int = field(default=REALTIME_SAMPLE_RATE, init=False)
    _tasks: list[asyncio.Task] = field(default_factory=list, init=False)
    _running: bool = field(default=False, init=False)
    _stop_evt: Optional[asyncio.Event] = field(default=None, init=False)

    # Half-duplex bookkeeping: ``_mic_muted`` is set while we expect
    # speaker output (so the mic pump drops chunks). ``_last_audio_t``
    # is the monotonic timestamp of the last audio delta we received,
    # used to decide when the playback "tail" has elapsed.
    _mic_muted: bool = field(default=False, init=False)
    _last_audio_t: float = field(default=0.0, init=False)
    _mic_dropped: int = field(default=0, init=False)

    # Barge-in detector state. ``_ambient_rms`` is an EMA of mic RMS
    # while muted, used as the noise floor. ``_loud_streak`` counts how
    # many consecutive recent frames have exceeded both the absolute
    # threshold and the ambient-multiplier threshold; once it reaches
    # ``barge_in_min_frames`` we trigger barge-in.
    _ambient_rms: float = field(default=0.02, init=False)
    _loud_streak: int = field(default=0, init=False)
    # Set when barge-in pre-empts the current response. The event
    # reader uses it to skip its post-RESPONSE_DONE buffer-clear (which
    # would otherwise wipe the user's just-uploaded barge-in audio).
    _barge_in_active: bool = field(default=False, init=False)
    # Accumulator for assistant audio-transcript deltas; flushed on
    # ``response.audio_transcript.done`` (or response.done as a
    # fallback). Keyed nothing — there's only one assistant response in
    # flight at a time on the realtime endpoint.
    _assistant_transcript_buf: str = field(default="", init=False)
    # Conversation-log ordering: the server often emits the assistant
    # audio transcript (Maxwell's reply) before
    # conversation.item.input_audio_transcription.completed for the
    # *user's* turn that triggered it. Holding the assistant line until
    # we either see the user transcript or hit a small timeout keeps
    # the UI ordered as "you -> Maxwell".
    _pending_assistant_text: str = field(default="", init=False)
    _awaiting_user_transcript: bool = field(default=False, init=False)
    _assistant_flush_task: Optional[asyncio.Task] = field(default=None, init=False)
    _assistant_flush_grace_s: float = 2.5

    # Push-to-talk runtime state. ``_ptt_active`` gates the mic pump
    # when ``push_to_talk`` is on. ``_ptt_uploaded_chunks`` tracks
    # whether at least one frame was sent during the current press;
    # without it ``input_audio_buffer.commit`` errors out (the server
    # rejects empty commits).
    _ptt_active: bool = field(default=False, init=False)
    _ptt_uploaded_chunks: int = field(default=0, init=False)

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Open the WebSocket session, configure it, and start audio I/O.

        Idempotent: if a session is already running, the existing one is
        torn down first and a fresh session opened. The caller never has
        to track running-state itself.
        """
        if self._running:
            log.info("realtime: existing session detected; replacing it")
            await self.stop()

        if sd is None:
            raise RuntimeError(
                "sounddevice not installed; realtime mode needs mic + speaker access. "
                "Install with `pip install sounddevice`."
            )

        self._stop_evt = asyncio.Event()
        self._output_queue = asyncio.Queue(maxsize=128)
        self._client = AsyncOpenAI(api_key=self.api_key)

        log.info("realtime: connecting to %s as voice=%s", self.model, self.voice)
        self._conn_ctx = self._client.beta.realtime.connect(model=self.model)
        self._conn = await self._conn_ctx.__aenter__()

        # Tell the server how we want it configured: PCM16 in/out at
        # 24 kHz, the requested voice, the personality / instruction
        # text, plus VAD + noise-reduction tuned for the deployment
        # environment. Done in a single ``session.update`` so the
        # next event we get back already reflects this config.
        if self.push_to_talk:
            # No automatic turn detection in PTT mode — the client
            # decides when each utterance starts and ends, and we
            # explicitly commit + request a response on key-up.
            turn_detection = None
        elif self.vad_type == "semantic_vad":
            turn_detection = {
                "type": "semantic_vad",
                "eagerness": self.vad_eagerness,
            }
        else:
            turn_detection = {
                "type": "server_vad",
                "threshold": float(self.vad_threshold),
                "prefix_padding_ms": int(self.vad_prefix_padding_ms),
                "silence_duration_ms": int(self.vad_silence_duration_ms),
            }
        session_payload = {
            "modalities": ["audio", "text"],
            "voice": self.voice,
            "instructions": self.instructions,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": turn_detection,
            # Server-side transcription of the user's mic so we can show
            # what they said in the conversation log. Without this,
            # ``conversation.item.input_audio_transcription.completed``
            # events never arrive.
            "input_audio_transcription": {"model": "whisper-1"},
        }
        if self.noise_reduction in ("near_field", "far_field"):
            session_payload["input_audio_noise_reduction"] = {
                "type": self.noise_reduction
            }
        if self.push_to_talk:
            log.info(
                "realtime: turn_detection=disabled (push-to-talk), noise_reduction=%s",
                self.noise_reduction,
            )
        else:
            log.info(
                "realtime: vad=%s thr=%s silence=%sms noise_reduction=%s",
                self.vad_type,
                turn_detection.get("threshold"),
                turn_detection.get("silence_duration_ms"),
                self.noise_reduction,
            )
        await self._conn.session.update(session=session_payload)

        self._open_audio_streams()
        self._running = True

        loop = asyncio.get_running_loop()
        self._tasks = [
            loop.create_task(self._mic_pump_loop(), name="rt-mic-pump"),
            loop.create_task(self._event_reader_loop(), name="rt-events"),
            loop.create_task(self._playback_loop(), name="rt-playback"),
        ]
        # Mic is already live and streaming up to the server, so put the
        # state machine into LISTENING immediately. That makes the
        # behavior engine kick out idle wing flaps + listening jitter so
        # the user sees Maxwell "wake up" the moment the button is hit,
        # without having to wait for the first audio response.
        await self._emit_state("listening")
        log.info("realtime: session active (mic streaming, awaiting speech)")

    async def stop(self) -> None:
        """Tear down the session cleanly. Safe to call repeatedly."""
        if not self._running and not self._tasks:
            return
        log.info("realtime: stopping session")
        self._running = False
        if self._stop_evt is not None:
            self._stop_evt.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks = []

        if self._conn_ctx is not None:
            try:
                await self._conn_ctx.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                log.exception("realtime: error closing websocket")
            finally:
                self._conn_ctx = None
                self._conn = None

        self._close_audio_streams()
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

        await self._emit_state("idle")
        log.info("realtime: stopped")

    # ------------------------- internals -------------------------

    def _device_default_sr(self, device, kind: str) -> int:
        """Best-effort native sample rate for a sounddevice device."""
        try:
            info = sd.query_devices(device, kind)
            sr = int(round(float(info.get("default_samplerate") or 0)))
            if sr > 0:
                return sr
        except Exception:  # noqa: BLE001
            log.debug("realtime: query_devices(%s, %s) failed", device, kind, exc_info=True)
        # Sensible macOS fallback (built-in mic + AirPods both run 48k).
        return 48_000

    def _open_audio_streams(self) -> None:
        loop = asyncio.get_running_loop()
        mic_q: asyncio.Queue = asyncio.Queue(maxsize=64)

        def _mic_cb(indata, frame_count, time_info, status):  # type: ignore[override]
            if status:
                log.debug("realtime mic status: %s", status)
            data = bytes(indata)
            if self._mic_sample_rate != REALTIME_SAMPLE_RATE:
                # Resample int16 PCM block from native to 24 kHz before
                # the OpenAI websocket sees it. Keep this branch tight —
                # _mic_cb runs on PortAudio's RT thread.
                data = _resample_int16_bytes(
                    data, self._mic_sample_rate, REALTIME_SAMPLE_RATE
                )
            try:
                loop.call_soon_threadsafe(mic_q.put_nowait, data)
            except asyncio.QueueFull:
                pass

        # ---------- input ----------
        self._mic_sample_rate = REALTIME_SAMPLE_RATE
        try:
            block_in = int(REALTIME_SAMPLE_RATE * INPUT_BLOCK_MS / 1000)
            self._input_stream = sd.RawInputStream(
                samplerate=REALTIME_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=block_in,
                device=self.input_device,
                callback=_mic_cb,
            )
        except sd.PortAudioError as exc:
            native_sr = self._device_default_sr(self.input_device, "input")
            log.warning(
                "realtime: input @ %d Hz failed (%s); falling back to "
                "native %d Hz with on-the-fly resampling",
                REALTIME_SAMPLE_RATE, exc, native_sr,
            )
            self._mic_sample_rate = native_sr
            block_in = int(native_sr * INPUT_BLOCK_MS / 1000)
            self._input_stream = sd.RawInputStream(
                samplerate=native_sr,
                channels=1,
                dtype="int16",
                blocksize=block_in,
                device=self.input_device,
                callback=_mic_cb,
            )
        self._mic_q = mic_q  # type: ignore[attr-defined]
        self._input_stream.start()

        # ---------- output ----------
        # Output stream: pulled from ``self._output_queue`` by the
        # ``_playback_loop`` task, written to a RawOutputStream in
        # blocking mode (the SDK gives us small chunks, not callbacks).
        self._output_sample_rate = REALTIME_SAMPLE_RATE
        try:
            block_out = int(REALTIME_SAMPLE_RATE * OUTPUT_BLOCK_MS / 1000)
            self._output_stream = sd.RawOutputStream(
                samplerate=REALTIME_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=block_out,
                device=self.output_device,
            )
        except sd.PortAudioError as exc:
            native_sr = self._device_default_sr(self.output_device, "output")
            log.warning(
                "realtime: output @ %d Hz failed (%s); falling back to "
                "native %d Hz with on-the-fly resampling",
                REALTIME_SAMPLE_RATE, exc, native_sr,
            )
            self._output_sample_rate = native_sr
            block_out = int(native_sr * OUTPUT_BLOCK_MS / 1000)
            self._output_stream = sd.RawOutputStream(
                samplerate=native_sr,
                channels=1,
                dtype="int16",
                blocksize=block_out,
                device=self.output_device,
            )
        self._output_stream.start()
        if (
            self._mic_sample_rate != REALTIME_SAMPLE_RATE
            or self._output_sample_rate != REALTIME_SAMPLE_RATE
        ):
            log.info(
                "realtime: audio rates — mic=%d Hz, out=%d Hz, wire=%d Hz",
                self._mic_sample_rate,
                self._output_sample_rate,
                REALTIME_SAMPLE_RATE,
            )

    def _close_audio_streams(self) -> None:
        for stream in (self._input_stream, self._output_stream):
            if stream is None:
                continue
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._input_stream = None
        self._output_stream = None

    async def _mic_pump_loop(self) -> None:
        """Drain the mic queue and forward chunks to the server.

        Two modes:

        1. Mic *unmuted* (normal operation): forward every chunk to the
           server. Server-side VAD does turn detection.

        2. Mic *muted* (Maxwell is currently speaking and half-duplex
           is on): drop chunks instead of forwarding them — this is
           what stops Maxwell from hearing himself loop back. While
           muted, compute the chunk's RMS and run the barge-in
           detector: if the user clearly speaks over the playback,
           un-mute and cancel the in-flight server response so the
           interruption feels natural.
        """
        chunks_sent = 0
        last_log = time.monotonic()
        try:
            while self._running:
                chunk = await self._mic_q.get()  # type: ignore[attr-defined]
                if not self._running or self._conn is None:
                    return
                # Push-to-talk mode: forward chunks only while the
                # button/key is held. Everything else is silently
                # dropped so half-duplex / barge-in / VAD logic is
                # bypassed entirely.
                if self.push_to_talk:
                    if not self._ptt_active:
                        self._mic_dropped += 1
                        continue
                elif self._mic_muted:
                    self._mic_dropped += 1
                    if self.barge_in_enabled:
                        await self._maybe_barge_in(chunk)
                    if self._mic_muted:
                        continue
                b64 = base64.b64encode(chunk).decode("ascii")
                try:
                    await self._conn.input_audio_buffer.append(audio=b64)
                    chunks_sent += 1
                    if self.push_to_talk:
                        self._ptt_uploaded_chunks += 1
                except Exception:  # noqa: BLE001
                    log.exception("realtime: mic upload failed; ending session")
                    return
                now = time.monotonic()
                if now - last_log > 5.0:
                    log.info(
                        "realtime: mic uploading (%d chunks in last 5s, "
                        "%d dropped while muted)",
                        chunks_sent,
                        self._mic_dropped,
                    )
                    chunks_sent = 0
                    self._mic_dropped = 0
                    last_log = now
        except asyncio.CancelledError:
            return

    async def ptt_down(self) -> None:
        """User pressed the push-to-talk key. Open the mic gate and
        cancel any in-flight assistant response so PTT acts like an
        immediate barge-in too. Safe to call when ``push_to_talk`` is
        off — it just no-ops.
        """
        if not self.push_to_talk:
            return
        self._ptt_active = True
        self._ptt_uploaded_chunks = 0
        log.info("realtime: PTT down (mic open)")
        if self._conn is not None:
            # If Maxwell is currently speaking, cut him off so the user
            # can talk over him without waiting for him to finish.
            try:
                await self._conn.response.cancel()
            except Exception:  # noqa: BLE001
                log.debug("realtime: response.cancel on PTT down failed", exc_info=True)
            try:
                await self._conn.input_audio_buffer.clear()
            except Exception:  # noqa: BLE001
                log.debug(
                    "realtime: input buffer clear on PTT down failed",
                    exc_info=True,
                )
            # Drain anything still queued for the speaker so playback
            # actually stops on the device, not just on the server.
            if self._output_queue is not None:
                while not self._output_queue.empty():
                    try:
                        self._output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
        await self._emit_state("listening")

    async def ptt_up(self) -> None:
        """User released the push-to-talk key. Commit whatever audio
        was uploaded during the press and ask the server to respond.
        No-op when PTT is off."""
        if not self.push_to_talk:
            return
        if not self._ptt_active:
            return
        self._ptt_active = False
        sent = self._ptt_uploaded_chunks
        self._ptt_uploaded_chunks = 0
        log.info("realtime: PTT up (sent %d chunks)", sent)
        if self._conn is None:
            return
        if sent == 0:
            # The server rejects empty commits; silently bail.
            log.info("realtime: PTT had no audio, skipping commit")
            return
        try:
            await self._conn.input_audio_buffer.commit()
        except Exception:  # noqa: BLE001
            log.exception("realtime: input_audio_buffer.commit failed")
            return
        try:
            await self._conn.response.create()
        except Exception:  # noqa: BLE001
            log.exception("realtime: response.create after PTT failed")
            return
        await self._emit_state("thinking")

    async def set_push_to_talk(self, enabled: bool) -> bool:
        """Toggle PTT on the live session. Returns ``True`` if a real
        change was applied. The session is reconfigured with
        ``turn_detection: null`` (PTT) or the previously-configured
        VAD (auto), without dropping the WebSocket."""
        enabled = bool(enabled)
        if enabled == self.push_to_talk:
            return False
        self.push_to_talk = enabled
        # Reset transient PTT state so a stale held-key flag doesn't
        # leak across the toggle.
        self._ptt_active = False
        self._ptt_uploaded_chunks = 0
        if self._conn is None:
            return True
        if enabled:
            turn_detection = None
        elif self.vad_type == "semantic_vad":
            turn_detection = {
                "type": "semantic_vad",
                "eagerness": self.vad_eagerness,
            }
        else:
            turn_detection = {
                "type": "server_vad",
                "threshold": float(self.vad_threshold),
                "prefix_padding_ms": int(self.vad_prefix_padding_ms),
                "silence_duration_ms": int(self.vad_silence_duration_ms),
            }
        try:
            await self._conn.session.update(
                session={"turn_detection": turn_detection}
            )
            log.info(
                "realtime: PTT %s (turn_detection=%s)",
                "enabled" if enabled else "disabled",
                "off" if turn_detection is None else turn_detection.get("type"),
            )
        except Exception:  # noqa: BLE001
            log.exception("realtime: session.update for PTT toggle failed")
        return True

    async def _maybe_barge_in(self, chunk: bytes) -> None:
        """Check whether this mic chunk indicates the user is barging
        in over Maxwell's playback. If so, un-mute and cancel the
        in-flight server response.

        We treat a frame as "loud" when its RMS exceeds *both* the
        absolute floor (``barge_in_rms_threshold``) *and* the recent
        ambient floor times ``barge_in_above_ambient_factor``. Quiet
        frames feed an EMA of the ambient noise floor (which captures
        the speaker bleed-back into the mic) so the multiplier
        threshold adapts to whatever volume Maxwell is playing at.
        ``barge_in_min_frames`` consecutive loud frames are required
        to trigger, which protects against a single loud syllable from
        playback briefly tripping the detector.
        """
        if not chunk:
            return
        samples = np.frombuffer(chunk, dtype=np.int16)
        if samples.size == 0:
            return
        rms = float(
            np.sqrt(np.mean(np.square(samples.astype(np.float32) / 32768.0)))
        )
        is_loud = (
            rms >= self.barge_in_rms_threshold
            and rms >= self._ambient_rms * self.barge_in_above_ambient_factor
        )
        if is_loud:
            self._loud_streak += 1
        else:
            self._loud_streak = 0
            # EMA only on quiet frames so the speaker's own playback
            # raises the ambient floor (and the multiplier requirement
            # along with it) without ever counting itself as barge-in.
            self._ambient_rms = 0.85 * self._ambient_rms + 0.15 * rms

        if self._loud_streak < self.barge_in_min_frames:
            return

        log.info(
            "realtime: BARGE-IN detected (rms=%.3f, ambient=%.3f, "
            "streak=%d) -> cancelling response",
            rms,
            self._ambient_rms,
            self._loud_streak,
        )
        # Stop dropping mic frames immediately so the in-progress user
        # speech actually reaches the server.
        self._mic_muted = False
        self._loud_streak = 0
        self._barge_in_active = True
        # Tell the server to stop generating its current response and
        # drain the playback queue so the speaker stops mid-syllable
        # instead of trailing on for another second.
        if self._conn is not None:
            try:
                await self._conn.response.cancel()
            except Exception:  # noqa: BLE001
                log.debug("realtime: response.cancel failed", exc_info=True)
            try:
                await self._conn.input_audio_buffer.clear()
            except Exception:  # noqa: BLE001
                log.debug(
                    "realtime: input_audio_buffer.clear failed",
                    exc_info=True,
                )
        if self._output_queue is not None:
            while not self._output_queue.empty():
                try:
                    self._output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        await self._emit_state("listening")

    async def _event_reader_loop(self) -> None:
        """Consume server events and route them to playback / state callbacks.

        Heavy logging here is intentional: when realtime mode "doesn't
        do anything" we need to know whether the WebSocket is silent,
        whether audio deltas are landing, whether state transitions are
        firing, etc. — without it we're flying blind.
        """
        if self._conn is None:
            return
        speaking = False
        audio_chunks = 0
        last_log = time.monotonic()
        first_event = True
        try:
            async for event in self._conn:
                etype = classify_event(event)
                if first_event:
                    log.info("realtime: first event from server type=%s", etype)
                    first_event = False
                if etype == EVENT_AUDIO_DELTA:
                    samples = decode_audio_delta(event)
                    if samples is not None:
                        audio_chunks += 1
                        self._last_audio_t = time.monotonic()
                        if not speaking:
                            speaking = True
                            # Reset barge-in tracking for the new
                            # response (ambient floor will re-learn
                            # this response's playback level).
                            self._loud_streak = 0
                            self._barge_in_active = False
                            # Engage half-duplex *before* announcing
                            # SPEAKING so the very first mic frame after
                            # we start playback is already dropped.
                            if self.half_duplex:
                                self._mic_muted = True
                                log.info(
                                    "realtime: mic muted (half-duplex, "
                                    "playback starting)"
                                )
                            log.info("realtime: assistant started speaking")
                            await self._emit_state("speaking")
                        if self._output_queue is not None:
                            try:
                                self._output_queue.put_nowait(samples)
                            except asyncio.QueueFull:
                                try:
                                    self._output_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                                self._output_queue.put_nowait(samples)
                elif etype == EVENT_RESPONSE_DONE:
                    if speaking:
                        log.info(
                            "realtime: response done from server (%d audio "
                            "chunks); waiting for playback to drain",
                            audio_chunks,
                        )
                        audio_chunks = 0
                        speaking = False
                        # The server is finished generating, but the
                        # output queue can still have hundreds of ms of
                        # buffered audio. If we flip the state machine
                        # back to LISTENING right now, the jaw goes
                        # silent mid-sentence. Wait for the queue to
                        # actually drain before emitting "listening".
                        if self._output_queue is not None:
                            while self._running and not self._output_queue.empty():
                                await asyncio.sleep(0.04)
                            # And give the OutputStream a beat to play
                            # the last buffered chunk to the speakers.
                            await asyncio.sleep(0.08)
                        # Tail wait for room reverb to die before we
                        # un-mute the mic. Without this, the trailing
                        # echo of Maxwell's last syllable gets caught
                        # by the server's VAD and triggers another
                        # response loop. Skip if barge-in already
                        # un-muted us — the user is talking now and
                        # clearing the input buffer would wipe their
                        # in-flight audio.
                        if self.half_duplex and not self._barge_in_active:
                            tail_s = max(0.0, self.playback_tail_ms / 1000.0)
                            await asyncio.sleep(tail_s)
                            if self._conn is not None:
                                try:
                                    await self._conn.input_audio_buffer.clear()
                                except Exception:  # noqa: BLE001
                                    log.debug(
                                        "realtime: input buffer clear failed",
                                        exc_info=True,
                                    )
                            self._mic_muted = False
                            log.info("realtime: mic unmuted (tail elapsed)")
                        elif self._barge_in_active:
                            log.info(
                                "realtime: skipping post-response cleanup "
                                "(barge-in handled the unmute)"
                            )
                        # Reset barge-in flag for the next turn.
                        self._barge_in_active = False
                    await self._emit_state("listening")
                elif etype == EVENT_SPEECH_STARTED:
                    if self._mic_muted:
                        # We dropped all mic chunks while muted, so the
                        # server can only be hearing its own playback
                        # bleeding back in. Ignore.
                        log.debug(
                            "realtime: ignoring speech_started while mic muted"
                        )
                        continue
                    log.info("realtime: user speech started")
                    await self._emit_state("listening")
                elif etype == EVENT_SPEECH_STOPPED:
                    if self._mic_muted:
                        continue
                    log.info("realtime: user speech stopped -> thinking")
                    # The user just finished an utterance; expect
                    # input_audio_transcription.completed for it before
                    # we surface Maxwell's reply.
                    self._awaiting_user_transcript = True
                    await self._emit_state("thinking")
                elif etype == EVENT_RESPONSE_CREATED:
                    # Same gate as speech_stopped: handles the PTT path
                    # where the server may not synthesize speech_stopped
                    # but always emits response.created at turn start.
                    self._awaiting_user_transcript = True
                    await self._emit_state("thinking")
                elif etype == EVENT_ASSISTANT_TRANSCRIPT_DELTA:
                    delta = _event_attr(event, "delta") or ""
                    if delta:
                        self._assistant_transcript_buf += delta
                elif etype == EVENT_ASSISTANT_TRANSCRIPT_DONE:
                    final = (
                        _event_attr(event, "transcript")
                        or self._assistant_transcript_buf
                    ).strip()
                    self._assistant_transcript_buf = ""
                    if final:
                        await self._deliver_assistant_transcript(final)
                elif etype == EVENT_USER_TRANSCRIPT:
                    text = (_event_attr(event, "transcript") or "").strip()
                    if text and self.transcript_callback is not None:
                        try:
                            self.transcript_callback("user", text)
                        except Exception:  # noqa: BLE001
                            log.debug(
                                "realtime: transcript callback (user) failed",
                                exc_info=True,
                            )
                    self._awaiting_user_transcript = False
                    # Now that the user line is on the wire, flush any
                    # assistant reply that arrived first.
                    await self._flush_pending_assistant()
                elif etype == EVENT_ERROR:
                    err = getattr(event, "error", None) or (
                        event.get("error") if isinstance(event, dict) else None
                    )
                    log.error("realtime: server error %s", err)
                else:
                    # Periodic heartbeat so we can see the socket is alive
                    # even when the user is just sitting silently.
                    now = time.monotonic()
                    if now - last_log > 5.0:
                        log.info("realtime: socket alive, last event=%s", etype)
                        last_log = now
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            log.exception("realtime: event reader exited unexpectedly")
            self._running = False

    async def _playback_loop(self) -> None:
        """Drain ``_output_queue`` to the audio output stream + envelope.

        Critical detail: per-audio-block RMS is way too coarse for the
        jaw — OpenAI sends deltas of 100-300 ms each, and the
        envelope follower's release coefficient decays to ~0 between
        them. Instead we walk every block in 20 ms windows (matching
        the existing ``play_buffer_with_envelope`` cadence) so the
        follower sees syllable-rate energy and the jaw actually tracks
        speech.
        """
        if self._output_queue is None or self._output_stream is None:
            return
        envelope_window_samples = max(
            1, int(REALTIME_SAMPLE_RATE * 20 / 1000)
        )
        try:
            while self._running:
                try:
                    samples = await asyncio.wait_for(
                        self._output_queue.get(), timeout=0.2
                    )
                except asyncio.TimeoutError:
                    continue
                if self._output_sample_rate != REALTIME_SAMPLE_RATE:
                    # Upsample 24 kHz float -> device native rate
                    # (typically 48 kHz on macOS) before handing to
                    # PortAudio. Linear interp is plenty for speech and
                    # cheaper than scipy.
                    samples_to_write = _resample_float(
                        samples, REALTIME_SAMPLE_RATE, self._output_sample_rate
                    )
                else:
                    samples_to_write = samples
                pcm16 = np.clip(samples_to_write * 32768.0, -32768, 32767).astype(np.int16)
                try:
                    await asyncio.to_thread(
                        self._output_stream.write, pcm16.tobytes()
                    )
                except Exception:  # noqa: BLE001
                    log.exception("realtime: audio playback write failed")
                if self.envelope_callback is not None and samples.size > 0:
                    for start in range(0, samples.size, envelope_window_samples):
                        window = samples[start : start + envelope_window_samples]
                        if window.size == 0:
                            continue
                        rms = float(np.sqrt(np.mean(np.square(window))))
                        try:
                            self.envelope_callback(rms)
                        except Exception:  # noqa: BLE001
                            log.exception("realtime: envelope callback raised")
        except asyncio.CancelledError:
            return

    async def _emit_state(self, name: str) -> None:
        if self.state_callback is None:
            return
        try:
            await self.state_callback(name)
        except Exception:  # noqa: BLE001
            log.exception("realtime: state callback raised")

    async def _deliver_assistant_transcript(self, text: str) -> None:
        """Emit Maxwell's transcript or hold it briefly if we're still
        waiting on the user's transcript for the same turn.
        """
        if self.transcript_callback is None:
            return
        if not self._awaiting_user_transcript:
            self._safe_transcript("maxwell", text)
            return
        # Buffer + start a grace timer so we don't hold forever if the
        # user transcription event never comes (e.g. server-side ASR is
        # disabled or fails).
        self._pending_assistant_text = text
        if self._assistant_flush_task is not None:
            self._assistant_flush_task.cancel()

        async def _grace() -> None:
            try:
                await asyncio.sleep(self._assistant_flush_grace_s)
            except asyncio.CancelledError:
                return
            if self._pending_assistant_text:
                log.debug(
                    "realtime: user transcript didn't arrive in %.1fs, "
                    "flushing held assistant line",
                    self._assistant_flush_grace_s,
                )
                self._awaiting_user_transcript = False
                await self._flush_pending_assistant()

        self._assistant_flush_task = asyncio.create_task(
            _grace(), name="rt-assistant-flush"
        )

    async def _flush_pending_assistant(self) -> None:
        text = self._pending_assistant_text
        if not text:
            return
        self._pending_assistant_text = ""
        if self._assistant_flush_task is not None:
            self._assistant_flush_task.cancel()
            self._assistant_flush_task = None
        self._safe_transcript("maxwell", text)

    def _safe_transcript(self, role: str, text: str) -> None:
        if self.transcript_callback is None:
            return
        try:
            self.transcript_callback(role, text)
        except Exception:  # noqa: BLE001
            log.debug(
                "realtime: transcript callback (%s) failed", role, exc_info=True
            )

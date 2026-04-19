"""Conversation pipeline: typed-text and live-mic loops.

This module wires providers, the state machine, the envelope follower, and the
motion scheduler together. Other modules should depend on this one; it should
not depend on ``cli.py``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from conversation.audio import (
    AudioBuffer,
    CapturedAudio,
    play_buffer_with_envelope,
    record_from_mic,
)
from conversation.llm import LLMProvider
from conversation.stt import STTProvider
from conversation.tts import TTSProvider
from motion.behavior_engine import BehaviorEngine, analyze_text, estimate_phrase_boundaries
from motion.envelope import EnvelopeFollower
from motion.models import (
    BehaviorGains,
    ConversationState,
    JawCalibration,
    SpeakingContext,
)
from motion.scheduler import MotionScheduler
from motion.state_machine import ConversationStateMachine
from transport.base import MotionBackend

log = logging.getLogger(__name__)


@dataclass
class LiveSpeakingContext:
    """Mutable container read each motion tick while speaking."""

    envelope_follower: EnvelopeFollower
    text: str = ""
    progress: float = 0.0
    rms: float = 0.0
    phrase_boundary: bool = False
    emphasis: float = 0.0
    question_like: bool = False
    excited: bool = False
    _phrase_boundaries: list[float] = field(default_factory=list)
    _consumed: int = 0
    _audio_duration_s: float = 0.0

    def set_utterance(self, text: str, duration_s: float) -> None:
        self.text = text
        self.progress = 0.0
        self.rms = 0.0
        self.emphasis = 0.0
        self._audio_duration_s = duration_s
        self._phrase_boundaries = estimate_phrase_boundaries(text, duration_s)
        self._consumed = 0
        analysis = analyze_text(text)
        self.question_like = analysis["question_like"]
        self.excited = analysis["excited"]
        self.envelope_follower.reset()
        self._behavior_smoothed = 0.0
        self._latest_envelope = 0.0
        self.phrase_boundary = True  # phrase-start nod

    def update_from_audio(self, progress: float, rms: float) -> None:
        self.progress = progress
        self.rms = rms
        # Jaw output uses the user-tuned calibration (gain, floor,
        # ceiling) so the physical mouth opening matches the bird.
        self.envelope_follower.process_rms(rms)
        # Behavior envelope drives wing flaps, head bobs and emphasis
        # in BehaviorEngine._speaking. It MUST be independent of the
        # jaw calibration, because the jaw is intentionally set to a
        # low gain (1.6) to match the physical servo — that low gain
        # would otherwise keep the behavior envelope below the wing /
        # bob thresholds, leaving Maxwell almost still while talking.
        # Use a fixed strong gain (tuned to land in 0.4-1.0 for normal
        # conversational TTS) and the same attack/release smoothing so
        # the value tracks syllables instead of jerking per sample.
        cal = self.envelope_follower.calibration
        target = min(1.0, max(0.0, rms) * 6.0)
        prev = getattr(self, "_behavior_smoothed", 0.0)
        if target > prev:
            coeff = max(0.0, min(1.0, cal.attack))
        else:
            coeff = max(0.0, min(1.0, cal.release))
        self._behavior_smoothed = prev + coeff * (target - prev)
        self._latest_envelope = self._behavior_smoothed
        # Emphasis: brief instantaneous spike on loud syllables. Same
        # rms*4 mapping as before, independent of the smoothed envelope.
        self.emphasis = min(1.0, rms * 4.0)

    def snapshot(self, now: float) -> SpeakingContext:
        boundary = False
        if self._phrase_boundaries and self._consumed < len(self._phrase_boundaries):
            next_boundary_s = self._phrase_boundaries[self._consumed]
            elapsed = self.progress * self._audio_duration_s
            if elapsed >= next_boundary_s:
                boundary = True
                self._consumed += 1
        if self.phrase_boundary:
            boundary = True
            self.phrase_boundary = False
        return SpeakingContext(
            envelope=getattr(self, "_latest_envelope", 0.0),
            text=self.text,
            progress=self.progress,
            phrase_boundary=boundary,
            emphasis=self.emphasis,
            question_like=self.question_like,
            excited=self.excited,
        )


@dataclass
class ConversationPipeline:
    stt: STTProvider
    llm: LLMProvider
    tts: TTSProvider
    backend: MotionBackend
    state_machine: ConversationStateMachine
    jaw_calibration: JawCalibration
    behavior_gains: BehaviorGains
    rate_hz: float = 30.0
    personality: str = ""
    audio_frame_ms: float = 20.0
    audio_input_sample_rate: int = 16000
    playback_device: Optional[str | int] = None
    mic_max_s: float = 15.0
    mic_silence_threshold: float = 0.012
    mic_silence_hangover_s: float = 1.2

    _speaking_ctx: Optional[LiveSpeakingContext] = None
    _scheduler: Optional[MotionScheduler] = None
    _rt_session: Optional[object] = None
    _rt_lock: Optional[asyncio.Lock] = None

    async def __aenter__(self) -> "ConversationPipeline":
        behavior = BehaviorEngine(gains=self.behavior_gains)
        self._scheduler = MotionScheduler(
            behavior=behavior,
            backend=self.backend,
            state_machine=self.state_machine,
            rate_hz=self.rate_hz,
            speaking_context_provider=self._speaking_snapshot,
        )
        await self._scheduler.start()
        await self.state_machine.idle()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop_realtime()
        if self._scheduler is not None:
            await self._scheduler.stop()

    def _speaking_snapshot(self, now: float) -> Optional[SpeakingContext]:
        if self._speaking_ctx is None:
            return None
        return self._speaking_ctx.snapshot(now)

    async def say(self, text: str) -> None:
        """Synthesize `text`, play it, and drive motion while it plays."""
        if not text.strip():
            return
        log.info("synthesizing: %s", text)
        audio = await self.tts.synthesize(text)
        await self._speak_audio(audio, text)

    async def speak_wav_file(self, path: str, *, text: str = "(replay)") -> None:
        """Replay a WAV file through the motion pipeline (for demo/offline modes)."""
        from conversation.audio import load_wav_file

        audio = load_wav_file(path)
        await self._speak_audio(audio, text)

    async def _speak_audio(self, audio: AudioBuffer, text: str) -> None:
        follower = EnvelopeFollower(
            sample_rate=audio.sample_rate,
            calibration=self.jaw_calibration,
            frame_ms=self.audio_frame_ms,
        )
        self._speaking_ctx = LiveSpeakingContext(envelope_follower=follower)
        self._speaking_ctx.set_utterance(text=text, duration_s=audio.duration_s)
        await self.state_machine.speaking()
        try:
            await play_buffer_with_envelope(
                audio,
                envelope_callback=self._speaking_ctx.update_from_audio,
                frame_ms=self.audio_frame_ms,
                output_device=self.playback_device,
            )
        finally:
            self._speaking_ctx = None
            await self.state_machine.idle()

    async def handle_typed_turn(self, user_text: str) -> str:
        await self.state_machine.thinking()
        reply = await self.llm.reply(user_text, personality=self.personality)
        await self.say(reply)
        return reply

    # ------------------------------------------------------------
    # Realtime API mode
    #
    # Opens a single OpenAI Realtime websocket session that streams
    # mic audio up and assistant audio down. Same envelope-follower /
    # state-machine plumbing as the typed-text path, so jaw motion
    # stays in lockstep with what the speaker is actually saying with
    # no second motion code path.
    # ------------------------------------------------------------

    async def start_realtime(
        self,
        *,
        api_key: str,
        model: str = "gpt-realtime",
        voice: str = "ballad",
        instructions: str = "",
        input_device: Optional[str | int] = None,
        vad_type: str = "server_vad",
        vad_threshold: float = 0.7,
        vad_prefix_padding_ms: int = 300,
        vad_silence_duration_ms: int = 700,
        vad_eagerness: str = "low",
        noise_reduction: str = "far_field",
        half_duplex: bool = True,
        playback_tail_ms: int = 400,
        barge_in_enabled: bool = True,
        barge_in_rms_threshold: float = 0.06,
        barge_in_above_ambient_factor: float = 5.0,
        barge_in_min_frames: int = 4,
        push_to_talk: bool = False,
        transcript_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Start (or restart) a Realtime API session. Idempotent."""
        from conversation.realtime import REALTIME_SAMPLE_RATE, RealtimeSession

        if self._rt_lock is None:
            self._rt_lock = asyncio.Lock()
        async with self._rt_lock:
            if self._rt_session is not None and getattr(self._rt_session, "is_running", False):
                log.info("realtime: replacing existing session")
                try:
                    await self._rt_session.stop()
                except Exception:  # noqa: BLE001
                    log.exception("realtime: error stopping previous session")
                self._rt_session = None

            follower = EnvelopeFollower(
                sample_rate=REALTIME_SAMPLE_RATE,
                calibration=self.jaw_calibration,
                frame_ms=self.audio_frame_ms,
            )
            ctx = LiveSpeakingContext(envelope_follower=follower)
            ctx.set_utterance(text="(realtime)", duration_s=60.0)
            self._speaking_ctx = ctx

            env_count = {"n": 0, "last": 0.0}

            def envelope_cb(rms: float) -> None:
                ctx.update_from_audio(progress=0.5, rms=rms)
                env_count["n"] += 1
                now = time.monotonic()
                if now - env_count["last"] > 2.0:
                    log.info(
                        "realtime: envelope driving jaw (%d frames, rms=%.3f)",
                        env_count["n"],
                        rms,
                    )
                    env_count["n"] = 0
                    env_count["last"] = now

            async def state_cb(name: str) -> None:
                log.info("realtime: state -> %s", name)
                if name == "listening":
                    await self.state_machine.listening()
                elif name == "thinking":
                    await self.state_machine.thinking()
                elif name == "speaking":
                    # Re-arm the speaking context for this fresh
                    # response: resets the envelope follower (so we
                    # don't carry decay state from a previous turn) and
                    # arms the phrase-start nod. Without this, the jaw
                    # only animates on the first response of the
                    # session, then sits idle on subsequent replies.
                    ctx.set_utterance(text="(realtime)", duration_s=60.0)
                    env_count["n"] = 0
                    env_count["last"] = 0.0
                    await self.state_machine.speaking()
                else:
                    await self.state_machine.idle()

            session = RealtimeSession(
                api_key=api_key,
                model=model,
                voice=voice,
                instructions=instructions or self.personality,
                input_device=input_device,
                output_device=self.playback_device,
                envelope_callback=envelope_cb,
                state_callback=state_cb,
                vad_type=vad_type,
                vad_threshold=vad_threshold,
                vad_prefix_padding_ms=vad_prefix_padding_ms,
                vad_silence_duration_ms=vad_silence_duration_ms,
                vad_eagerness=vad_eagerness,
                noise_reduction=noise_reduction,
                half_duplex=half_duplex,
                playback_tail_ms=playback_tail_ms,
                barge_in_enabled=barge_in_enabled,
                barge_in_rms_threshold=barge_in_rms_threshold,
                barge_in_above_ambient_factor=barge_in_above_ambient_factor,
                barge_in_min_frames=barge_in_min_frames,
                push_to_talk=push_to_talk,
                transcript_callback=transcript_callback,
            )
            await session.start()
            self._rt_session = session

            # Visible "I'm awake" — flap the wing once so the user has
            # immediate confirmation that realtime mode launched and the
            # serial backend is reachable, without waiting for the
            # assistant's first audio response.
            if hasattr(self.backend, "send_frame"):
                from motion.models import MotionFrame
                try:
                    await self.backend.send_frame(
                        MotionFrame(jaw_open=0.0, head_lr=0.5, head_ud=0.5, wing=1.0)
                    )
                    await asyncio.sleep(0.25)
                    await self.backend.send_frame(
                        MotionFrame(jaw_open=0.0, head_lr=0.5, head_ud=0.5, wing=0.0)
                    )
                except Exception:  # noqa: BLE001
                    log.exception("realtime: wake flap failed (non-fatal)")

    async def stop_realtime(self) -> None:
        """Tear down the Realtime session if one is open. Idempotent."""
        if self._rt_lock is None:
            self._rt_lock = asyncio.Lock()
        async with self._rt_lock:
            if self._rt_session is None:
                return
            try:
                await self._rt_session.stop()
            except Exception:  # noqa: BLE001
                log.exception("realtime: error during stop")
            self._rt_session = None
            self._speaking_ctx = None
            await self.state_machine.idle()

    @property
    def realtime_running(self) -> bool:
        return self._rt_session is not None and getattr(
            self._rt_session, "is_running", False
        )

    async def realtime_set_push_to_talk(self, enabled: bool) -> bool:
        """Toggle PTT on the live Realtime session.
        Returns ``True`` if a change was applied. ``False`` if there
        was no session or the value was already correct."""
        if self._rt_session is None:
            return False
        try:
            return await self._rt_session.set_push_to_talk(enabled)
        except Exception:  # noqa: BLE001
            log.exception("realtime: set_push_to_talk failed")
            return False

    async def realtime_ptt_down(self) -> bool:
        if self._rt_session is None:
            return False
        try:
            await self._rt_session.ptt_down()
            return True
        except Exception:  # noqa: BLE001
            log.exception("realtime: ptt_down failed")
            return False

    async def realtime_ptt_up(self) -> bool:
        if self._rt_session is None:
            return False
        try:
            await self._rt_session.ptt_up()
            return True
        except Exception:  # noqa: BLE001
            log.exception("realtime: ptt_up failed")
            return False

    async def handle_live_turn(self) -> tuple[str, str]:
        await self.state_machine.listening()
        captured = await record_from_mic(
            sample_rate=self.audio_input_sample_rate,
            max_duration_s=self.mic_max_s,
            silence_threshold=self.mic_silence_threshold,
            silence_hangover_s=self.mic_silence_hangover_s,
        )
        if captured.samples.size == 0:
            await self.state_machine.idle()
            return "", ""
        await self.state_machine.thinking()
        user_text = await self.stt.transcribe(
            AudioBuffer(samples=captured.samples, sample_rate=captured.sample_rate)
        )
        if not user_text.strip():
            await self.state_machine.idle()
            return "", ""
        log.info("heard: %s", user_text)
        reply = await self.llm.reply(user_text, personality=self.personality)
        await self.say(reply)
        return user_text, reply

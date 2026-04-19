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
from typing import Optional

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
        self.phrase_boundary = True  # phrase-start nod

    def update_from_audio(self, progress: float, rms: float) -> None:
        self.progress = progress
        self.rms = rms
        jaw = self.envelope_follower.process_rms(rms)
        # jaw mapped to [floor..ceiling]; derive a 0..1 "envelope" for the
        # behavior engine so it can blend with other motion cleanly.
        floor = self.envelope_follower.calibration.floor
        ceil = self.envelope_follower.calibration.ceiling
        if ceil - floor > 1e-6:
            self._latest_envelope = max(0.0, min(1.0, (jaw - floor) / (ceil - floor)))
        else:
            self._latest_envelope = jaw
        # Mark emphasis when RMS spikes significantly.
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

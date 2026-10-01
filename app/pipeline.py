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
    GazeContext,
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
    vision_config: Optional[object] = None  # app.config.VisionConfig; kept loose to avoid an import cycle

    _speaking_ctx: Optional[LiveSpeakingContext] = None
    _scheduler: Optional[MotionScheduler] = None
    _rt_session: Optional[object] = None
    _rt_lock: Optional[asyncio.Lock] = None
    _gaze_ctx: Optional[GazeContext] = None
    _face_tracker: Optional[object] = None
    _vision_lock: Optional[asyncio.Lock] = None
    _last_scene_at: float = 0.0
    _face_memory: Optional[object] = None
    _recognition: Optional[object] = None
    _recognition_ctx: Optional[object] = None

    async def __aenter__(self) -> "ConversationPipeline":
        behavior = BehaviorEngine(gains=self.behavior_gains)
        # Shared gaze target: written by the face-tracking task, read once per
        # motion tick by the behavior engine. Always present (centered / zero
        # confidence) so the scheduler can read it unconditionally; it only moves
        # the head once the tracker starts writing detections into it.
        self._gaze_ctx = GazeContext()
        self._scheduler = MotionScheduler(
            behavior=behavior,
            backend=self.backend,
            state_machine=self.state_machine,
            rate_hz=self.rate_hz,
            speaking_context_provider=self._speaking_snapshot,
            gaze_provider=self._gaze_snapshot,
        )
        await self._scheduler.start()
        await self.state_machine.idle()
        cfg = self.vision_config
        if cfg is not None and getattr(cfg, "enabled", False):
            try:
                await self.start_face_tracking()
            except Exception:  # noqa: BLE001 - vision is optional; never block startup
                log.exception("vision: face tracking failed to start (continuing)")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop_realtime()
        await self.stop_face_tracking()
        if self._scheduler is not None:
            await self._scheduler.stop()

    def _speaking_snapshot(self, now: float) -> Optional[SpeakingContext]:
        if self._speaking_ctx is None:
            return None
        return self._speaking_ctx.snapshot(now)

    def _gaze_snapshot(self, now: float) -> Optional[GazeContext]:
        return self._gaze_ctx

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

            tools, tool_hint = self._build_realtime_tools()
            base_instructions = instructions or self.personality
            if tool_hint:
                base_instructions = (
                    (base_instructions + "\n\n") if base_instructions else ""
                ) + tool_hint

            session = RealtimeSession(
                api_key=api_key,
                model=model,
                voice=voice,
                instructions=base_instructions,
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
                tools=tools or None,
                tool_handler=self._handle_realtime_tool if tools else None,
            )
            await session.start()
            self._rt_session = session
            # If a face is already present when the session opens, seed the identity
            # line and greet them right away (don't wait for them to speak first).
            if self._recognition_ctx is not None:
                line = self._identity_line(self._recognition_ctx)
                if line:
                    try:
                        await session.set_context_line(line)
                    except Exception:  # noqa: BLE001
                        log.exception("realtime: initial identity injection failed")
                try:
                    await self._maybe_greet(self._recognition_ctx)
                except Exception:  # noqa: BLE001
                    log.exception("realtime: initial greeting failed")

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

    # ------------------------------------------------------------
    # Vision: face tracking (fast, local) + scene understanding (on-demand)
    # ------------------------------------------------------------

    async def start_face_tracking(self) -> bool:
        """Start (or no-op if already running) the webcam face-tracking loop.

        Builds the camera + MediaPipe detector from ``vision_config`` and spins up
        a :class:`vision.face_tracker.FaceTracker` writing into the shared
        ``_gaze_ctx`` the motion scheduler already reads. Idempotent.
        """
        if self._vision_lock is None:
            self._vision_lock = asyncio.Lock()
        async with self._vision_lock:
            if self._face_tracker is not None and getattr(
                self._face_tracker, "is_running", False
            ):
                return False
            from vision.camera import OpenCVCameraSource
            from vision.face_detector import build_face_detector
            from vision.face_tracker import FaceTracker

            cfg = self.vision_config
            camera = OpenCVCameraSource(index=getattr(cfg, "camera_index", 0))
            detector = build_face_detector(getattr(cfg, "detector", "auto"))
            tracker = FaceTracker(
                camera=camera,
                detector=detector,
                gaze=self._gaze_ctx,
                fps=getattr(cfg, "tracking_fps", 12.0),
                gain_lr=getattr(cfg, "gaze_gain_lr", 1.4),
                gain_ud=getattr(cfg, "gaze_gain_ud", 1.2),
                invert_lr=getattr(cfg, "invert_lr", False),
                invert_ud=getattr(cfg, "invert_ud", False),
                deadzone=getattr(cfg, "deadzone", 0.05),
                lost_face_timeout_s=getattr(cfg, "lost_face_timeout_s", 1.5),
            )
            await tracker.start()
            self._face_tracker = tracker
            log.info("vision: face tracking started")
            if getattr(cfg, "recognition_enabled", False):
                await self._start_recognition(cfg, tracker)
            return True

    async def _start_recognition(self, cfg, tracker) -> None:
        """Build + start the face-recognition task alongside tracking.

        Guarded: a missing/broken InsightFace just logs and skips recognition so
        face tracking + scene understanding keep working.
        """
        try:
            from vision.face_memory import FaceMemory
            from vision.face_recognizer import build_face_recognizer
            from vision.recognition import RecognitionContext, RecognitionTracker

            recognizer = build_face_recognizer(getattr(cfg, "insightface_model", "buffalo_l"))
            persist = (
                getattr(cfg, "memory_path", None)
                if getattr(cfg, "memory_persist", False)
                else None
            )
            self._face_memory = FaceMemory(
                persist_path=persist,
                max_per_person=getattr(cfg, "max_embeddings_per_person", 20),
            )
            self._recognition_ctx = RecognitionContext()
            recognition = RecognitionTracker(
                face_tracker=tracker,
                recognizer=recognizer,
                memory=self._face_memory,
                context=self._recognition_ctx,
                fps=getattr(cfg, "recognition_fps", 3.0),
                threshold=getattr(cfg, "recognition_threshold", 0.40),
                margin=getattr(cfg, "recognition_margin", 0.05),
                votes=getattr(cfg, "recognition_votes", 8),
                min_sharpness=getattr(cfg, "min_sharpness", 60.0),
                enroll_sample_interval_s=getattr(cfg, "enroll_sample_interval_s", 1.0),
                stranger_match_threshold=getattr(cfg, "stranger_match_threshold", 0.45),
                on_identity_change=self._on_identity_change,
            )
            await recognition.start()
            self._recognition = recognition
            log.info("vision: face recognition started")
        except Exception:  # noqa: BLE001 - recognition is optional
            log.exception("vision: face recognition unavailable (continuing without it)")
            self._recognition = None

    async def _on_identity_change(self, ctx) -> None:
        """Push the recognized identity into the live Realtime conversation and,
        when someone new or remembered appears, make Maxwell greet them first."""
        session = self._rt_session
        if session is None or not getattr(session, "is_running", False):
            return
        line = self._identity_line(ctx)
        try:
            await session.set_context_line(line)
        except Exception:  # noqa: BLE001
            log.exception("vision: failed to push identity to realtime session")
        await self._maybe_greet(ctx)

    async def _maybe_greet(self, ctx) -> None:
        """Speak first when a face appears — greet by name / welcome a stranger.

        Gated by a per-identity **last-seen cooldown**: if this face was on screen
        within ``greeting_cooldown_s`` it's the same ongoing encounter (detection
        just flickered), so we stay quiet rather than re-greeting someone we're
        already with. Only a genuine return after a long absence — or a first
        sighting — greets. Also gated by the session being idle so a greeting never
        talks over an in-progress turn.
        """
        cfg = self.vision_config
        if cfg is None or not getattr(cfg, "greet_on_sight", True):
            return
        session = self._rt_session
        if session is None or not getattr(session, "is_running", False):
            return
        key, instruction = self._greeting_for(ctx)
        if key is None:
            return
        cooldown = getattr(cfg, "greeting_cooldown_s", 600.0)
        rec = self._recognition
        last_seen = rec.last_seen_at(ctx) if rec is not None else None
        if last_seen is not None and (time.monotonic() - last_seen) < cooldown:
            # Seen recently — same encounter, detection just blipped. Don't re-greet.
            return
        try:
            await session.trigger_greeting(instruction)
        except Exception:  # noqa: BLE001
            log.exception("vision: greeting trigger failed")

    @staticmethod
    def _greeting_for(ctx) -> tuple[Optional[str], str]:
        """(cooldown_key, instruction) for greeting this identity, or (None, '')."""
        if ctx is None:
            return None, ""
        if ctx.name:
            return ctx.name, (
                f"{ctx.name} has just come into view. Greet them warmly by name like "
                "an old friend — one short, cheerful sentence."
            )
        if ctx.is_unknown:
            return "__unknown__", (
                "Someone new has just appeared in front of you. Warmly say hello, "
                "introduce yourself as Maxwell, and ask their name — one or two short "
                "sentences."
            )
        return None, ""

    @staticmethod
    def _identity_line(ctx) -> str:
        """Natural-language 'who am I looking at' line for the model."""
        if ctx is None:
            return ""
        if ctx.name:
            return (
                f"You are looking at {ctx.name}, someone you've met before. "
                "Greet them warmly by name."
            )
        if ctx.is_unknown:
            return (
                "You are looking at someone you don't recognize yet. If they tell "
                "you their name, call remember_person to remember their face."
            )
        return ""

    async def stop_face_tracking(self) -> None:
        """Stop the face-tracking loop (and recognition) if running. Idempotent."""
        if self._vision_lock is None:
            self._vision_lock = asyncio.Lock()
        async with self._vision_lock:
            if self._recognition is not None:
                try:
                    await self._recognition.stop()
                except Exception:  # noqa: BLE001
                    log.exception("vision: error stopping recognition")
                self._recognition = None
            if self._face_tracker is None:
                return
            try:
                await self._face_tracker.stop()
            except Exception:  # noqa: BLE001
                log.exception("vision: error stopping face tracker")
            self._face_tracker = None
        if self._gaze_ctx is not None:
            self._gaze_ctx.confidence = 0.0

    @property
    def face_tracking_running(self) -> bool:
        return self._face_tracker is not None and getattr(
            self._face_tracker, "is_running", False
        )

    def vision_status(self) -> dict:
        """Snapshot for the operator UI: tracking state + who Maxwell recognizes."""
        gaze = self._gaze_ctx
        seeing_face = bool(gaze is not None and gaze.confidence > 0.05)
        status = {
            "tracking": self.face_tracking_running,
            "seeing_face": seeing_face,
            "target_lr": round(gaze.target_lr, 3) if gaze else 0.5,
            "target_ud": round(gaze.target_ud, 3) if gaze else 0.5,
            "confidence": round(gaze.confidence, 3) if gaze else 0.0,
        }
        rc = self._recognition_ctx
        mem = self._face_memory
        status["recognition"] = {
            "running": self._recognition is not None
            and getattr(self._recognition, "is_running", False),
            "name": getattr(rc, "name", None) if rc else None,
            "is_unknown": bool(getattr(rc, "is_unknown", False)) if rc else False,
            "confidence": round(getattr(rc, "confidence", 0.0), 3) if rc else 0.0,
            "known": mem.summary() if mem is not None else {},
        }
        cfg = self.vision_config
        # Feature switches, so the operator UI can grey out what's turned off.
        status["scene_enabled"] = bool(getattr(cfg, "scene_enabled", False)) if cfg else False
        status["recognition_enabled"] = (
            bool(getattr(cfg, "recognition_enabled", False)) if cfg else False
        )
        return status

    def forget_face(self, name: Optional[str] = None, *, everyone: bool = False) -> dict:
        """Delete a stored person (or everyone). For the operator UI / voice tool."""
        mem = self._face_memory
        if mem is None:
            return {"ok": False, "error": "recognition not running"}
        if everyone:
            n = mem.forget_all()
            return {"ok": True, "removed": n}
        if not name:
            return {"ok": False, "error": "no name given"}
        removed = mem.forget(name)
        return {"ok": removed, "removed": 1 if removed else 0, "name": name}

    async def describe_scene(self, prompt: Optional[str] = None) -> str:
        """Grab the current camera frame and describe it via the vision LLM.

        Reuses the tracker's most recent frame (no second camera reader). If
        tracking isn't running, grabs a single frame directly. Throttled by
        ``scene_min_interval_s`` so rapid re-triggers don't stack paid calls — a
        too-soon call returns a short in-character deferral instead of hitting the
        API. Speaking the result is the caller's job (e.g. ``pipeline.say``).
        """
        cfg = self.vision_config
        if cfg is None or not getattr(cfg, "scene_enabled", False):
            # Paid image analysis is switched off — never touch the API.
            return "My eyes are just for following faces today, so I can't describe things."
        min_interval = getattr(cfg, "scene_min_interval_s", 3.0) if cfg else 3.0
        now = time.monotonic()
        if self._last_scene_at and (now - self._last_scene_at) < min_interval:
            return "Give me a second — I'm still looking!"

        frame = None
        if self._face_tracker is not None:
            frame = self._face_tracker.latest_frame()
        if frame is None:
            frame = await self._grab_single_frame()
        if frame is None:
            return "I can't see anything right now — my camera's not on."

        self._last_scene_at = now
        from vision.scene import build_vision_provider

        provider = build_vision_provider(
            getattr(cfg, "scene_provider", "openai") if cfg else "openai",
            model=getattr(cfg, "scene_model", "gpt-4o-mini") if cfg else "gpt-4o-mini",
            max_output_tokens=getattr(cfg, "scene_max_tokens", 120) if cfg else 120,
        )
        scene_prompt = prompt or (
            getattr(cfg, "scene_prompt", "") if cfg else ""
        ) or "Describe what you see in one short sentence."
        return await provider.describe(frame, prompt=scene_prompt)

    def _build_realtime_tools(self) -> tuple[list[dict], str]:
        """Assemble the Realtime tool schemas + an instruction hint for them.

        ``look_and_describe`` (paid image analysis) is offered only when
        ``scene_enabled`` is on; the ``remember_person`` / ``forget_person``
        face-memory tools only when ``recognition_enabled`` is on. With both off the
        session gets no tools at all. Returns ``(schemas, hint_text)``.
        """
        if self.vision_config is None:
            return [], ""
        tools: list[dict] = []
        hint = ""
        if getattr(self.vision_config, "scene_enabled", False):
            tools.append(
                {
                    "type": "function",
                    "name": "look_and_describe",
                    "description": (
                        "Look through your camera and describe what you currently see. "
                        "Call this whenever the user asks what you see, asks you to look "
                        "at something, or shows you an object."
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }
            )
            hint = (
                "You have a camera and can see. When someone asks what you see, asks "
                "you to look, or shows you something, call look_and_describe and react "
                "to what it returns."
            )
        if getattr(self.vision_config, "recognition_enabled", False):
            tools.extend(
                [
                    {
                        "type": "function",
                        "name": "remember_person",
                        "description": (
                            "Remember the face of the person you're currently looking "
                            "at under a name. Call this when someone you don't "
                            "recognize tells you their name so you can greet them next "
                            "time."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The person's name.",
                                }
                            },
                            "required": ["name"],
                        },
                    },
                    {
                        "type": "function",
                        "name": "forget_person",
                        "description": (
                            "Forget a person you've remembered. Call this if someone "
                            "asks you to forget them."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the person to forget.",
                                }
                            },
                            "required": ["name"],
                        },
                    },
                ]
            )
            hint += (
                " You can remember faces. The 'Current view' note tells you who you're "
                "looking at. If it's someone you've met, greet them by name. If it's "
                "someone new and they tell you their name, call remember_person to "
                "remember them. If someone asks to be forgotten, call forget_person."
            )
        return tools, hint.strip()

    async def _handle_realtime_tool(self, name: str, args: dict) -> str:
        """Dispatch a Realtime tool call to the right vision/memory action."""
        if name == "look_and_describe":
            try:
                return await self.describe_scene()
            except Exception:  # noqa: BLE001
                log.exception("realtime look failed")
                return "I tried to look but couldn't see anything just now."
        if name == "remember_person":
            person = (args or {}).get("name", "").strip()
            if not person:
                return "I didn't catch the name — what should I call you?"
            if self._recognition is None:
                return "My face memory isn't running right now."
            count = self._recognition.flush_pending(person)
            if count > 0:
                return f"Great, I'll remember you, {person}!"
            return (
                f"I'd love to remember you, {person}, but I can't see your face "
                "clearly yet — can you look at me for a moment?"
            )
        if name == "forget_person":
            person = (args or {}).get("name", "").strip()
            result = self.forget_face(person)
            if result.get("ok"):
                return f"Okay, I've forgotten {person}."
            return f"I don't think I had {person} remembered."
        log.warning("realtime: unknown tool %s", name)
        return "I'm not sure how to do that."

    async def _grab_single_frame(self):
        """Open the camera, grab one frame, close it — used when tracking is off."""
        from vision.camera import OpenCVCameraSource

        cfg = self.vision_config
        camera = OpenCVCameraSource(index=getattr(cfg, "camera_index", 0) if cfg else 0)
        loop = asyncio.get_event_loop()

        def _grab():
            camera.open()
            try:
                # Discard a couple of warm-up frames; the first read is often black.
                frame = None
                for _ in range(4):
                    frame = camera.read()
                return frame
            finally:
                camera.close()

        try:
            return await loop.run_in_executor(None, _grab)
        except Exception:  # noqa: BLE001
            log.exception("vision: single-frame grab failed")
            return None

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

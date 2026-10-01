"""Decoupled face-recognition loop.

Identity does *not* need 30 fps — head tracking does. This task runs at a low rate
(~3 Hz) on the frame the tracker already grabbed (``FaceTracker.latest_frame``),
embeds the largest face, matches it against :class:`FaceMemory`, and commits an
identity only after **temporal voting** over several frames so a single bad frame
can't make Maxwell blurt the wrong name.

While the committed identity is *unknown*, it buffers sharp, confident embeddings
into a ``pending`` set; when the conversation yields a name (the ``remember_person``
tool calls :meth:`flush_pending`), that buffer is enrolled under the name. Sampling
is gated on a Laplacian sharpness floor so the moving head's motion-blurred frames
don't pollute the stored set.

Mirrors :class:`vision.face_tracker.FaceTracker`: one ``asyncio.Task`` with a
``start``/``stop`` lifecycle; the blocking embed runs in a thread executor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, List, Optional

import numpy as np

log = logging.getLogger(__name__)

IdentityChangeCallback = Callable[["RecognitionContext"], Awaitable[None]]


@dataclass
class RecognitionContext:
    """Shared, mutable identity state — the recognition analog of GazeContext.

    Written by the task, read by the pipeline to inject "who am I looking at" into
    the conversation. ``name`` is the committed identity (``None`` = nobody / not yet
    decided); ``is_unknown`` is True when a face is present but not recognized.
    """

    name: Optional[str] = None
    confidence: float = 0.0
    is_unknown: bool = False
    updated_at: float = 0.0
    pending_count: int = 0
    # Internal cooldown key for the committed identity: a real name for known
    # people, or a transient per-stranger id (e.g. "\x00stranger:2") so each
    # distinct unrecognized person gets their own greeting cooldown. Not shown to
    # the model — the chat only ever sees ``name`` / ``is_unknown``.
    identity_key: Optional[str] = None


@dataclass
class RecognitionTracker:
    """Runs recognition at a low rate and maintains the committed identity."""

    face_tracker: object  # provides latest_frame() -> Optional[np.ndarray]
    recognizer: object  # FaceRecognizer
    memory: object  # FaceMemory
    context: RecognitionContext
    fps: float = 3.0
    threshold: float = 0.40
    margin: float = 0.05
    votes: int = 8
    min_sharpness: float = 60.0
    min_det_score: float = 0.5
    enroll_sample_interval_s: float = 1.0
    max_pending: int = 15
    stranger_match_threshold: float = 0.45
    on_identity_change: Optional[IdentityChangeCallback] = None

    _task: Optional[asyncio.Task] = field(default=None, init=False)
    _stopped: bool = field(default=False, init=False)
    _window: Deque[Optional[str]] = field(default=None, init=False)  # type: ignore[assignment]
    _pending: List[np.ndarray] = field(default_factory=list, init=False)
    _last_sample_t: float = field(default=0.0, init=False)
    _committed: Optional[str] = field(default=None, init=False)
    # monotonic timestamp each identity was last actually detected on screen.
    # Refreshed every frame a face is present, so a person who never leaves stays
    # "recently seen" even when detection flickers — the greeting logic reads this
    # so we never re-greet someone we're currently with.
    _last_seen: dict = field(default_factory=dict, init=False)
    # Transient in-session stranger identities: id -> embedding set. Lets us tell
    # distinct unrecognized people apart (each gets its own greeting + cooldown)
    # and re-identify a stranger who steps out and comes back. Session-only.
    _strangers: dict = field(default_factory=dict, init=False)
    _stranger_seq: int = field(default=0, init=False)

    _STRANGER_PREFIX = "\x00stranger:"

    @classmethod
    def _is_stranger(cls, value) -> bool:
        return isinstance(value, str) and value.startswith(cls._STRANGER_PREFIX)

    def _resolve_stranger(self, vec: np.ndarray) -> str:
        """Return the transient id of the stranger this embedding belongs to.

        Matches against the in-session stranger set (max-over-set cosine). Prefers
        the currently-committed stranger when it still matches (hysteresis, so a
        single person doesn't fragment into several ids frame-to-frame); otherwise
        the nearest stranger above threshold; otherwise mints a brand-new id.
        """
        vec = np.asarray(vec, dtype=np.float32)

        def _sim(sid: str) -> float:
            vecs = self._strangers.get(sid) or []
            return max((float(np.dot(vec, v)) for v in vecs), default=-1.0)

        # Incumbent first (hysteresis): keep the current stranger latched if it
        # still matches, so angle/lighting jitter doesn't spawn duplicate ids.
        if self._is_stranger(self._committed) and _sim(self._committed) >= self.stranger_match_threshold:
            best_id = self._committed
        else:
            best_id, best = None, self.stranger_match_threshold
            for sid in self._strangers:
                s = _sim(sid)
                if s >= best:
                    best, best_id = s, sid
        if best_id is None:
            self._stranger_seq += 1
            best_id = f"{self._STRANGER_PREFIX}{self._stranger_seq}"
            self._strangers[best_id] = []
        bucket = self._strangers[best_id]
        bucket.append(vec)
        if len(bucket) > self.max_pending:
            del bucket[: len(bucket) - self.max_pending]
        return best_id

    def last_seen_at(self, ctx) -> Optional[float]:
        """Monotonic time the identity in ``ctx`` was last on screen, or None.

        The greeting policy uses this: if a face was seen within the cooldown it's
        the same ongoing encounter (detection just blipped), so don't re-greet.
        Keyed on the identity's cooldown key — the real name for known people, or
        the per-stranger transient id for unrecognized people.
        """
        key = getattr(ctx, "identity_key", None) if ctx is not None else None
        if key is None:
            return None
        return self._last_seen.get(key)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopped = False
        self._window = deque(maxlen=max(3, self.votes * 2))
        self._task = asyncio.create_task(self._run(), name="face-recognition")
        log.info("recognition started (%.1f Hz)", self.fps)

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        log.info("recognition stopped")

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def flush_pending(self, name: str) -> int:
        """Enroll the buffered unknown-person embeddings under ``name``.

        Called by the ``remember_person`` tool. Returns how many embeddings were
        stored (0 if we haven't gathered a good look yet). Clears the buffer and
        immediately commits the identity so Maxwell stops treating them as a
        stranger.
        """
        if not self._pending:
            return 0
        count = self.memory.enroll(name, list(self._pending))
        # This stranger now has a real name — drop their transient identity.
        if self._is_stranger(self._committed):
            self._strangers.pop(self._committed, None)
        self._pending.clear()
        now = time.monotonic()
        self._committed = name
        self.context.name = name
        self.context.is_unknown = False
        self.context.identity_key = name
        self.context.confidence = 1.0
        self.context.pending_count = 0
        self.context.updated_at = now
        # They're right here and just introduced themselves — mark seen so the
        # greeting logic doesn't immediately greet them as a fresh arrival.
        self._last_seen[name] = now
        return count

    def has_pending(self) -> bool:
        return bool(self._pending)

    async def _run(self) -> None:
        period = 1.0 / max(0.5, self.fps)
        loop = asyncio.get_event_loop()
        next_tick = time.monotonic()
        while not self._stopped:
            now = time.monotonic()
            frame = None
            getter = getattr(self.face_tracker, "latest_frame", None)
            if getter is not None:
                frame = getter()
            emb = None
            if frame is not None:
                try:
                    emb = await loop.run_in_executor(
                        None, self.recognizer.embed_largest, frame
                    )
                except Exception:  # noqa: BLE001 - a bad frame must not kill the loop
                    log.exception("recognition embed failed")
            await self._process(emb, now)

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for < 0:
                next_tick = time.monotonic()
                sleep_for = 0
            try:
                await asyncio.sleep(sleep_for)
            except asyncio.CancelledError:
                return

    async def _process(self, emb, now: float) -> None:
        # Vote for this frame: a matched name (known person), a per-stranger
        # transient id (face present but not a known name — resolved against the
        # in-session stranger set so distinct people get distinct ids), or None
        # (no face at all — doesn't count toward a commit).
        if emb is None:
            vote: Optional[str] = None
        else:
            match = self.memory.identify(
                emb.vector, threshold=self.threshold, margin=self.margin
            )
            vote = match.name if match is not None else self._resolve_stranger(emb.vector)

        self._window.append(vote)

        # Committed identity = a value with a clear plurality in the window. Only
        # real votes (name / stranger id) count; None (no face) is ignored so brief
        # drop-outs don't reset the identity.
        counts = Counter(v for v in self._window if v is not None)
        committed = self._committed
        confidence = self.context.confidence
        if counts:
            top_value, top_count = counts.most_common(1)[0]
            if top_count >= self.votes:
                committed = top_value
                confidence = 0.0 if self._is_stranger(top_value) else min(
                    1.0, top_count / max(1, len(self._window))
                )

        is_unknown = self._is_stranger(committed)
        name = None if (committed is None or is_unknown) else committed
        changed = committed != self._committed

        # Buffer embeddings while unknown so a volunteered name can enroll them.
        # Reset the buffer when the stranger changes so we never mix two people.
        if changed and is_unknown:
            self._pending.clear()
        if is_unknown and emb is not None:
            if (
                emb.sharpness >= self.min_sharpness
                and emb.det_score >= self.min_det_score
                and (now - self._last_sample_t) >= self.enroll_sample_interval_s
            ):
                self._pending.append(emb.vector)
                if len(self._pending) > self.max_pending:
                    del self._pending[: len(self._pending) - self.max_pending]
                self._last_sample_t = now
        elif not is_unknown:
            # Recognized (or nobody) — no point hoarding stranger samples.
            self._pending.clear()

        self._committed = committed
        self.context.name = name
        self.context.is_unknown = is_unknown
        self.context.identity_key = committed
        self.context.confidence = confidence
        self.context.pending_count = len(self._pending)
        self.context.updated_at = now

        # Fire the change callback BEFORE refreshing last-seen, so the greeting
        # policy reads the timestamp from *before* this appearance (a genuine
        # return reads "long ago"; a detection blip reads "seconds ago").
        if changed and self.on_identity_change is not None:
            try:
                await self.on_identity_change(self.context)
            except Exception:  # noqa: BLE001
                log.exception("recognition: identity-change callback failed")

        # Refresh last-seen for whoever is actually on screen right now, keyed by
        # the committed identity (name or per-stranger id). Gated on a real
        # detection this frame so that once a person leaves and the frame is empty
        # their timestamp stops advancing and the cooldown can elapse.
        if emb is not None and committed is not None:
            self._last_seen[committed] = now

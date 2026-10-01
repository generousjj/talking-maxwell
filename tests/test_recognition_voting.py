"""RecognitionTracker temporal voting + pending-enrollment behavior."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

import numpy as np

from vision.face_memory import Match
from vision.recognition import RecognitionContext, RecognitionTracker


@dataclass
class _Emb:
    vector: np.ndarray
    det_score: float = 0.99
    sharpness: float = 500.0


class _StubMemory:
    """identify() returns a preset Match (or None); records enrollments."""

    def __init__(self, result):
        self.result = result
        self.enrolled = []

    def identify(self, vec, *, threshold, margin):
        return self.result

    def enroll(self, name, vecs):
        self.enrolled.append((name, list(vecs)))
        return len(vecs)


def _tracker(memory, votes=3):
    t = RecognitionTracker(
        face_tracker=None,
        recognizer=None,
        memory=memory,
        context=RecognitionContext(),
        votes=votes,
        enroll_sample_interval_s=0.0,  # sample every eligible frame in tests
    )
    t._window = deque(maxlen=votes * 2)  # normally set in start()
    return t


def _run(coro):
    return asyncio.run(coro)


def test_commits_only_after_enough_agreeing_votes():
    mem = _StubMemory(Match("Sarah", 0.7))
    t = _tracker(mem, votes=3)
    emb = _Emb(np.zeros(512, dtype=np.float32))

    async def go():
        # Two frames: not enough votes yet.
        await t._process(emb, now=0.0)
        await t._process(emb, now=0.1)
        assert t.context.name is None
        # Third agreeing frame commits.
        await t._process(emb, now=0.2)
        assert t.context.name == "Sarah"
        assert t.context.is_unknown is False

    _run(go())


def test_single_stray_frame_does_not_flip_identity():
    mem = _StubMemory(Match("Sarah", 0.7))
    t = _tracker(mem, votes=3)
    emb = _Emb(np.zeros(512, dtype=np.float32))

    async def go():
        for i in range(4):
            await t._process(emb, now=i * 0.1)
        assert t.context.name == "Sarah"
        # One stray "no face" frame shouldn't unseat the committed identity.
        await t._process(None, now=0.5)
        assert t.context.name == "Sarah"

    _run(go())


def test_unknown_buffers_pending_and_flush_enrolls():
    mem = _StubMemory(None)  # never recognizes -> unknown
    t = _tracker(mem, votes=2)
    emb = _Emb(np.ones(512, dtype=np.float32))

    async def go():
        for i in range(4):
            await t._process(emb, now=i * 1.0)
        assert t.context.is_unknown is True
        assert t.has_pending()
        count = t.flush_pending("Josh")
        assert count > 0
        assert mem.enrolled and mem.enrolled[0][0] == "Josh"
        # After flushing, identity is committed and pending cleared.
        assert t.context.name == "Josh"
        assert not t.has_pending()

    _run(go())


def test_last_seen_refreshes_while_present_and_stalls_when_gone():
    mem = _StubMemory(Match("Sarah", 0.7))
    t = _tracker(mem, votes=2)
    emb = _Emb(np.zeros(512, dtype=np.float32))
    ctx = RecognitionContext(name="Sarah", identity_key="Sarah")

    async def go():
        # Commit Sarah, and last-seen advances with each detected frame.
        for i in range(3):
            await t._process(emb, now=float(i))
        seen_a = t.last_seen_at(ctx)
        assert seen_a is not None
        await t._process(emb, now=10.0)
        seen_b = t.last_seen_at(ctx)
        assert seen_b == 10.0 and seen_b > seen_a
        # No face for a few frames -> last-seen must NOT advance (so the cooldown
        # can eventually elapse for a genuine departure).
        await t._process(None, now=20.0)
        await t._process(None, now=21.0)
        assert t.last_seen_at(ctx) == 10.0

    _run(go())


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def test_distinct_strangers_get_distinct_ids_and_buckets():
    # identify() always returns None -> everyone is a stranger, told apart by
    # embedding. Two near-orthogonal faces must become two different ids.
    mem = _StubMemory(None)
    t = _tracker(mem, votes=2)
    rng = np.random.default_rng(0)
    a = _Emb(_unit(rng.standard_normal(512)))
    b = _Emb(_unit(rng.standard_normal(512)))

    async def go():
        for i in range(3):
            await t._process(a, now=float(i))
        id_a = t.context.identity_key
        assert t.context.is_unknown and RecognitionTracker._is_stranger(id_a)
        # Person B replaces A.
        for i in range(4):
            await t._process(b, now=10.0 + i)
        id_b = t.context.identity_key
        assert RecognitionTracker._is_stranger(id_b)
        assert id_b != id_a
        assert len(t._strangers) == 2  # two distinct people tracked

    _run(go())


def test_same_stranger_reidentified_keeps_one_id():
    mem = _StubMemory(None)
    t = _tracker(mem, votes=2)
    rng = np.random.default_rng(1)
    base = _unit(rng.standard_normal(512))
    a = _Emb(base)

    async def go():
        for i in range(3):
            await t._process(a, now=float(i))
        id1 = t.context.identity_key
        # They step out (no face), then the same person returns.
        await t._process(None, now=5.0)
        await t._process(None, now=6.0)
        for i in range(3):
            await t._process(_Emb(_unit(base + 0.01 * rng.standard_normal(512))), now=10.0 + i)
        id2 = t.context.identity_key
        assert id2 == id1  # re-identified as the same stranger
        assert len(t._strangers) == 1  # not spawned a duplicate

    _run(go())


def test_blurry_frames_are_not_buffered():
    mem = _StubMemory(None)
    t = _tracker(mem, votes=2)
    t.min_sharpness = 100.0
    blurry = _Emb(np.ones(512, dtype=np.float32), sharpness=10.0)

    async def go():
        for i in range(4):
            await t._process(blurry, now=i * 1.0)
        # Unknown, but every frame was too blurry to store.
        assert t.context.is_unknown is True
        assert not t.has_pending()

    _run(go())

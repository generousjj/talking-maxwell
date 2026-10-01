"""FaceMemory: identity matching (rejection band, max-over-set) + persistence."""

from __future__ import annotations

import numpy as np

from vision.face_memory import FaceMemory


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(dim))


def test_identify_returns_best_match_above_threshold():
    mem = FaceMemory()
    sarah = _vec(1)
    mem.enroll("Sarah", [sarah])
    mem.enroll("Tom", [_vec(2)])
    # Query almost identical to Sarah.
    q = _unit(sarah + 0.01 * _vec(99))
    match = mem.identify(q, threshold=0.4, margin=0.05)
    assert match is not None
    assert match.name == "Sarah"
    assert match.score > 0.9


def test_identify_unknown_below_threshold():
    mem = FaceMemory()
    mem.enroll("Sarah", [_vec(1)])
    # A totally unrelated face.
    assert mem.identify(_vec(500), threshold=0.4, margin=0.05) is None


def test_identify_rejection_band_when_top_two_are_close():
    # Two people whose stored vectors are near each other; a query roughly between
    # them yields top-1 and top-2 within the margin → unknown (never a wrong name).
    mem = FaceMemory()
    base = _vec(10)
    a = _unit(base + 0.02 * _vec(11))
    b = _unit(base + 0.02 * _vec(12))
    mem.enroll("Anna", [a])
    mem.enroll("Bella", [b])
    q = _unit(a + b)  # nearly equidistant
    assert mem.identify(q, threshold=0.2, margin=0.05) is None


def test_identify_uses_max_over_set_not_centroid():
    # A person stored from two very different views. A query matching ONE view
    # should be recognized via max-over-set, even though the centroid of the two
    # views is far from the query (a centroid approach would miss it).
    mem = FaceMemory()
    view1 = _vec(20)
    view2 = _vec(21)  # nearly orthogonal to view1
    mem.enroll("Kai", [view1, view2])
    q = _unit(view1 + 0.01 * _vec(22))
    match = mem.identify(q, threshold=0.4, margin=0.05)
    assert match is not None and match.name == "Kai"
    # Score reflects the best view, not the (much lower) centroid similarity.
    centroid = _unit(view1 + view2)
    assert match.score > float(np.dot(q, centroid)) + 0.2


def test_enroll_caps_per_person():
    mem = FaceMemory(max_per_person=3)
    for i in range(10):
        mem.enroll("Sam", [_vec(100 + i)])
    assert mem.summary()["Sam"] == 3


def test_forget_and_forget_all():
    mem = FaceMemory()
    mem.enroll("Sarah", [_vec(1)])
    mem.enroll("Tom", [_vec(2)])
    assert mem.forget("sarah") is True  # case-insensitive
    assert "Sarah" not in mem.names()
    assert mem.forget("nobody") is False
    assert mem.forget_all() == 1
    assert mem.names() == []


def test_persistence_round_trip(tmp_path):
    path = str(tmp_path / "mem.json")
    mem = FaceMemory(persist_path=path)
    mem.enroll("Sarah", [_vec(1), _vec(3)])
    mem.enroll("Tom", [_vec(2)])

    # A fresh instance loads the same people/counts.
    reloaded = FaceMemory(persist_path=path)
    assert reloaded.summary() == {"Sarah": 2, "Tom": 1}
    # And a loaded embedding still identifies.
    q = _unit(_vec(1) + 0.01 * _vec(77))
    assert reloaded.identify(q, threshold=0.4, margin=0.05).name == "Sarah"

    # Forgetting rewrites disk.
    reloaded.forget("Tom")
    again = FaceMemory(persist_path=path)
    assert "Tom" not in again.names()
    again.forget_all()
    assert FaceMemory(persist_path=path).names() == []

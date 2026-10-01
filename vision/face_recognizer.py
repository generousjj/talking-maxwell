"""Face embedding via InsightFace (ArcFace / buffalo_l).

Face *recognition* is a different problem from face *tracking*: the tracker
(``face_tracker``/``face_detector``) tells us *where* a face is to drive the servos;
this module tells us *whose* face it is by mapping a face to a 512-d embedding where
cosine distance means identity.

We use InsightFace's ``buffalo_l`` bundle (SCRFD detector + ArcFace recogniser via
ONNX Runtime). Critically it does its **own** detection and 5-point alignment on the
frame — we do *not* hand it the tracker's box — because skipping ArcFace's alignment
quietly tanks accuracy (the Haar/MediaPipe boxes don't map onto the alignment
template). ``insightface`` is imported lazily so this module imports cleanly without
it; the recognition subsystem then just stays off (like the detector fallback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceEmbedding:
    """One detected+embedded face.

    ``vector`` is the L2-normalized 512-d ArcFace embedding (so cosine similarity is
    a plain dot product). ``bbox`` is ``(x1, y1, x2, y2)`` in pixels; ``det_score``
    is the detector's confidence; ``sharpness`` is the Laplacian variance of the face
    crop (higher = crisper), used to reject motion-blurred frames before enrolling.
    """

    vector: np.ndarray
    bbox: tuple[float, float, float, float]
    det_score: float
    sharpness: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def laplacian_sharpness(frame: np.ndarray, bbox) -> float:
    """Variance of the Laplacian over the face crop — a cheap blur metric.

    A moving head produces motion-blurred frames; storing those pollutes a person's
    embedding set, so the recognition task gates enrollment on this being above a
    configured floor. Returns 0.0 if opencv is unavailable or the crop is empty.
    """
    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover - import guard
        return 0.0
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FaceRecognizer:
    """InsightFace embedder. Detects, aligns, and embeds faces in a frame."""

    def __init__(self, *, model: str = "buffalo_l", det_size: int = 640) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "insightface is required for face recognition. "
                "Install it via `pip install insightface onnxruntime`."
            ) from exc
        # CPU is plenty at our ~3 Hz recognition rate and avoids GPU setup.
        self._app = FaceAnalysis(name=model, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=(det_size, det_size))
        log.info("face recognizer ready (insightface %s)", model)

    def embed_faces(self, frame: np.ndarray) -> List[FaceEmbedding]:
        """Detect + embed every face in a BGR frame."""
        faces = self._app.get(frame)
        out: List[FaceEmbedding] = []
        for f in faces:
            vec = getattr(f, "normed_embedding", None)
            if vec is None:
                continue
            bbox = tuple(float(v) for v in f.bbox)
            out.append(
                FaceEmbedding(
                    vector=np.asarray(vec, dtype=np.float32),
                    bbox=bbox,  # type: ignore[arg-type]
                    det_score=float(getattr(f, "det_score", 0.0)),
                    sharpness=laplacian_sharpness(frame, bbox),
                )
            )
        return out

    def embed_largest(self, frame: np.ndarray) -> Optional[FaceEmbedding]:
        """Embed only the largest (nearest) face — the one-on-one booth case.

        Multi-person disambiguation (per-track identity) is intentionally out of
        scope; picking the biggest face matches how the tracker chooses its target.
        """
        faces = self.embed_faces(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)

    def close(self) -> None:  # symmetry with other vision components
        self._app = None


def build_face_recognizer(model: str = "buffalo_l", *, det_size: int = 640) -> FaceRecognizer:
    """Factory mirroring the other ``build_x`` helpers in this package."""
    return FaceRecognizer(model=model, det_size=det_size)

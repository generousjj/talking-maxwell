"""Session face memory: name → embeddings, with identity matching.

Maps a face embedding to a name (or "unknown"). Backed by RAM; optional disk
persistence when a ``persist_path`` is given (default off — session-only, nothing
biometric written to disk).

Two design points from the recognition guidance:

* **Match against the max similarity across a person's embedding set, not a
  centroid.** Averaging blurs genuinely different viewing conditions (angle,
  lighting) together; the max keeps each stored view usable.
* **Bias toward "unknown".** ``identify`` returns ``None`` not only when the best
  score is low, but also when the top two candidates are within a small margin of
  each other. Nearest-neighbour without a rejection band always returns *somebody* —
  and confidently calling a person the wrong name is far worse than "sorry, remind
  me?".
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Match:
    name: str
    score: float


class FaceMemory:
    """In-memory name→embeddings store with optional JSON persistence.

    Thread-safe: the recognition task writes/reads from an executor thread while
    tool handlers on the event loop enroll/forget, so all access takes ``_lock``.
    Persistence (when enabled) is plain JSON — no pickle — so the file is inert and
    inspectable; it lives under a gitignored ``data/`` dir since it's biometric.
    """

    def __init__(
        self,
        *,
        persist_path: Optional[str] = None,
        max_per_person: int = 20,
    ) -> None:
        self.persist_path = persist_path
        self.max_per_person = max_per_person
        self._store: Dict[str, List[np.ndarray]] = {}
        self._lock = threading.Lock()
        if self.persist_path:
            self._load()

    # ---- identity ----

    def identify(
        self, vec: np.ndarray, *, threshold: float = 0.40, margin: float = 0.05
    ) -> Optional[Match]:
        """Return the best-matching name, or ``None`` for unknown.

        ``vec`` and stored vectors are L2-normalized, so cosine similarity is a dot
        product. Unknown when the top score is below ``threshold`` OR the top two
        names are within ``margin`` of each other (the rejection band).
        """
        vec = np.asarray(vec, dtype=np.float32)
        with self._lock:
            scored = [
                (name, max(float(np.dot(vec, v)) for v in vecs))
                for name, vecs in self._store.items()
                if vecs
            ]
        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        top_name, top_score = scored[0]
        if top_score < threshold:
            return None
        if len(scored) >= 2 and (top_score - scored[1][1]) < margin:
            return None
        return Match(name=top_name, score=top_score)

    # ---- mutation ----

    def enroll(self, name: str, vecs: List[np.ndarray]) -> int:
        """Add embeddings for ``name`` (created if new). Returns total stored count.

        Caps at ``max_per_person`` by dropping the oldest — keeping a spread of
        recent views across angles/lighting matters more than any single one.
        """
        name = (name or "").strip()
        if not name or not vecs:
            return 0
        with self._lock:
            bucket = self._store.setdefault(name, [])
            bucket.extend(np.asarray(v, dtype=np.float32) for v in vecs)
            if len(bucket) > self.max_per_person:
                del bucket[: len(bucket) - self.max_per_person]
            count = len(bucket)
            self._save_locked()
        log.info("face memory: enrolled %d embedding(s) for %r (total %d)",
                 len(vecs), name, count)
        return count

    def forget(self, name: str) -> bool:
        """Remove one person (case-insensitive). Returns True if anyone was removed."""
        target = (name or "").strip().lower()
        with self._lock:
            for key in list(self._store):
                if key.lower() == target:
                    del self._store[key]
                    self._save_locked()
                    log.info("face memory: forgot %r", key)
                    return True
        return False

    def forget_all(self) -> int:
        """Wipe everyone. Returns how many people were removed."""
        with self._lock:
            n = len(self._store)
            self._store.clear()
            self._save_locked()
        log.info("face memory: forgot all (%d people)", n)
        return n

    # ---- introspection ----

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._store.keys())

    def summary(self) -> Dict[str, int]:
        """name → number of stored embeddings, for the operator UI."""
        with self._lock:
            return {name: len(vecs) for name, vecs in sorted(self._store.items())}

    # ---- persistence (JSON; caller-locked variants) ----

    def _save_locked(self) -> None:
        if not self.persist_path:
            return
        try:
            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "people": {
                    name: [v.tolist() for v in vecs]
                    for name, vecs in self._store.items()
                },
            }
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
        except Exception:  # noqa: BLE001 - persistence must never crash recognition
            log.exception("face memory: save failed")

    def _load(self) -> None:
        path = Path(self.persist_path) if self.persist_path else None
        if not path or not path.exists():
            return
        try:
            payload = json.loads(path.read_text())
            people = payload.get("people", {})
            with self._lock:
                self._store = {
                    name: [np.asarray(v, dtype=np.float32) for v in vecs]
                    for name, vecs in people.items()
                }
            log.info("face memory: loaded %d people from %s", len(self._store), path)
        except Exception:  # noqa: BLE001 - a corrupt file must not block startup
            log.exception("face memory: load failed; starting empty")
            self._store = {}

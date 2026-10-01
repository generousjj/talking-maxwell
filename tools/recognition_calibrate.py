"""Calibrate the face-recognition threshold for your camera + lighting.

The `recognition_threshold` (0.40) and `recognition_margin` (0.05) defaults are
starting points, not constants. This tool prints the raw cosine-similarity scores so
you can see where the same-person and different-person distributions actually sit for
your setup, and pick a threshold that separates them cleanly.

Usage:

    python tools/recognition_calibrate.py            # live: enroll then compare
    python tools/recognition_calibrate.py --seconds 5

Flow: it captures a reference face for a few seconds (multiple embeddings), then
streams live similarity of the current face against that reference set. Show your own
face first (scores should be high, ~0.5-0.8), then have a different person step in
(scores should drop, ~0.0-0.3). The gap between those two bands is your margin; put
the threshold in the middle.

Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from vision.camera import OpenCVCameraSource
from vision.face_recognizer import build_face_recognizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=4.0, help="reference capture window")
    args = parser.parse_args()

    print("Loading InsightFace (first run downloads the model)...")
    recognizer = build_face_recognizer()
    camera = OpenCVCameraSource(index=args.camera)
    camera.open()

    try:
        print(f"\n=== Capturing REFERENCE face for {args.seconds:.0f}s — look at the camera ===")
        ref: list[np.ndarray] = []
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            frame = camera.read()
            if frame is None:
                continue
            emb = recognizer.embed_largest(frame)
            if emb is not None and emb.sharpness > 40:
                ref.append(emb.vector)
                print(f"  captured ref embedding ({len(ref)})  sharpness={emb.sharpness:.0f}")
            time.sleep(0.2)
        if not ref:
            print("No face captured. Check lighting / camera and retry.")
            return 1

        print(f"\nGot {len(ref)} reference embeddings.")
        print("=== Now streaming live similarity. Same person = high, different = low. ===")
        print("Show your own face, then have someone else step in. Ctrl-C to stop.\n")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            emb = recognizer.embed_largest(frame)
            if emb is None:
                print("  (no face)")
            else:
                sims = [float(np.dot(emb.vector, r)) for r in ref]
                best = max(sims)
                mean = sum(sims) / len(sims)
                bar = "#" * int(max(0.0, best) * 40)
                print(f"  max={best:.3f}  mean={mean:.3f}  det={emb.det_score:.2f}  |{bar}")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        camera.close()
        recognizer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

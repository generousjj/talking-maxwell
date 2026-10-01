"""Standalone webcam + face-tracking preview — no Maxwell hardware needed.

Run this first to confirm the camera and face detector work on your machine and
to tune the gaze mapping before wiring anything into the bird:

    python tools/vision_preview.py                # default webcam (index 0)
    python tools/vision_preview.py --camera 1      # a different camera
    python tools/vision_preview.py --mock          # no webcam; moving test dot

It opens a window showing the camera feed with every detected face boxed, the
chosen primary face highlighted, and the computed head-aim ``(head_lr, head_ud)``
drawn as a crosshair + printed to the terminal. Press ``q`` or ``Esc`` to quit.

The gaze values here are exactly what the ``FaceTracker`` feeds Maxwell's head
servos, so if the crosshair tracks your face the way you'd want his beak to point,
the mapping (gain / invert / deadzone) is right.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.camera import MockCameraSource, OpenCVCameraSource
from vision.face_detector import build_face_detector
from vision.face_tracker import map_face_to_gaze, select_primary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--mock", action="store_true", help="use the mock moving-dot source")
    parser.add_argument("--gain-lr", type=float, default=1.4)
    parser.add_argument("--gain-ud", type=float, default=1.2)
    parser.add_argument("--invert-lr", action="store_true")
    parser.add_argument("--invert-ud", action="store_true")
    parser.add_argument("--deadzone", type=float, default=0.05)
    args = parser.parse_args()

    try:
        import cv2  # type: ignore
    except Exception:
        print("opencv-python is required: pip install opencv-python", file=sys.stderr)
        return 1

    camera = MockCameraSource() if args.mock else OpenCVCameraSource(index=args.camera)

    if args.mock:
        # The mock source draws a bright blob, not a real face — a real detector
        # won't fire on it, so fall back to a trivial "brightest region" stand-in
        # just so the preview shows motion end-to-end without a webcam.
        detector = _BlobDetector()
    else:
        detector = build_face_detector("auto")

    camera.open()
    prev_center = None
    last_print = 0.0
    print("Preview running. Press 'q' or Esc in the window to quit.")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            h, w = frame.shape[:2]
            faces = detector.detect(frame)
            primary = select_primary(faces, prev_center, hysteresis=0.25)
            if primary is not None:
                prev_center = (primary.cx, primary.cy)

            for f in faces:
                x = int((f.cx - f.width / 2) * w)
                y = int((f.cy - f.height / 2) * h)
                bw = int(f.width * w)
                bh = int(f.height * h)
                is_primary = primary is not None and f is primary
                color = (0, 255, 0) if is_primary else (120, 120, 120)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            if primary is not None:
                lr, ud = map_face_to_gaze(
                    primary,
                    gain_lr=args.gain_lr,
                    gain_ud=args.gain_ud,
                    invert_lr=args.invert_lr,
                    invert_ud=args.invert_ud,
                    deadzone=args.deadzone,
                )
                gx, gy = int(lr * w), int(ud * h)
                cv2.drawMarker(frame, (gx, gy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
                cv2.putText(
                    frame,
                    f"head_lr={lr:.2f}  head_ud={ud:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                now = time.monotonic()
                if now - last_print > 0.25:
                    print(f"faces={len(faces)}  head_lr={lr:.3f}  head_ud={ud:.3f}")
                    last_print = now
            else:
                cv2.putText(
                    frame,
                    "no face",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (180, 180, 180),
                    2,
                )

            cv2.imshow("Maxwell vision preview", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.close()
        detector.close()
        cv2.destroyAllWindows()
    return 0


class _BlobDetector:
    """Preview-only stand-in for --mock: finds the brightest blob's centroid.

    Only used so `--mock` shows an end-to-end moving target without a webcam or a
    real face. The real pipeline always uses the MediaPipe detector.
    """

    def detect(self, frame):
        import numpy as np

        from vision.face_detector import FaceBox

        gray = frame.mean(axis=2)
        if gray.max() < 40:
            return []
        ys, xs = np.where(gray > gray.max() * 0.6)
        if xs.size == 0:
            return []
        h, w = gray.shape
        cx = float(xs.mean()) / w
        cy = float(ys.mean()) / h
        bw = float(xs.max() - xs.min()) / w
        bh = float(ys.max() - ys.min()) / h
        return [FaceBox(cx=cx, cy=cy, width=bw, height=bh, score=1.0)]

    def close(self):
        pass


if __name__ == "__main__":
    raise SystemExit(main())

"""Vision subsystem: webcam capture, face tracking, and scene understanding.

Two independent pipelines share this package:

* A fast, free, **local** face-tracking loop (``camera`` + ``face_detector`` +
  ``face_tracker``) that turns Maxwell's head toward the nearest face at ~12 Hz.
* A slow, paid, **on-demand** scene-understanding call (``scene``) that sends a
  single frame to a vision-capable LLM so Maxwell can talk about what he's shown.

Heavy third-party imports (``cv2``, ``mediapipe``) are guarded inside the modules
that need them, so importing this package never fails just because a camera or
those wheels aren't installed — mirroring how ``conversation.audio`` guards
``sounddevice``.
"""

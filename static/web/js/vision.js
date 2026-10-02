// Browser-side face tracking. Client-only port of vision/face_tracker.py:
// grab webcam frames, detect the nearest face with the native Shape
// Detection API (`window.FaceDetector`), and maintain a shared gaze
// context ({target_lr, target_ud, confidence}) that the motion
// scheduler feeds into BehaviorEngine._apply_gaze each tick — exactly
// where app/pipeline.py feeds the Python GazeContext. No server/Vercel
// changes: getUserMedia + detection + the existing Web Serial motion
// path all run in the booth browser.
//
// Deliberately uses only the native FaceDetector (no CDN/WASM deps).
// That's Chrome/Edge-only and can be gated behind
// chrome://flags/#enable-experimental-web-platform-features on some
// builds; `isSupported()` lets the UI fail soft with a clear message.

function clamp01(x) { if (x < 0) return 0; if (x > 1) return 1; return x; }

// Port of vision.face_tracker.map_face_to_gaze. A face centered in frame
// yields (0.5, 0.5); offsets scale by gain and flip per axis via invert.
function mapFaceToGaze(face, { gainLr, gainUd, invertLr, invertUd, deadzone }) {
  let ex = face.cx - 0.5;
  let ey = face.cy - 0.5;
  if (Math.abs(ex) < deadzone) ex = 0;
  if (Math.abs(ey) < deadzone) ey = 0;
  const sx = invertLr ? -1 : 1;
  const sy = invertUd ? -1 : 1;
  return {
    lr: clamp01(0.5 + sx * gainLr * ex),
    ud: clamp01(0.5 + sy * gainUd * ey),
  };
}

// Port of vision.face_tracker.select_primary: keep tracking the face
// closest to the previous center (temporal stability); a different,
// bigger face only steals focus if it clearly beats the held one.
function selectPrimary(faces, prevCenter, hysteresis = 0.25) {
  if (!faces.length) return null;
  let largest = faces[0];
  for (const f of faces) if (f.area > largest.area) largest = f;
  if (!prevCenter) return largest;
  let nearest = faces[0];
  let bestD = Infinity;
  for (const f of faces) {
    const d = (f.cx - prevCenter[0]) ** 2 + (f.cy - prevCenter[1]) ** 2;
    if (d < bestD) { bestD = d; nearest = f; }
  }
  if (nearest === largest) return largest;
  if (largest.area > nearest.area * (1 + Math.max(0, hysteresis))) return largest;
  return nearest;
}

export function faceDetectionSupported() {
  return typeof window !== "undefined" && "FaceDetector" in window;
}

export class BrowserFaceTracker {
  // Defaults mirror the config.yaml `vision:` block + FaceTracker
  // dataclass defaults so browser behavior matches the Python booth app.
  constructor({
    video,
    log = () => {},
    onStatus = () => {},
    fps = 12,
    gainLr = 1.4,
    gainUd = 1.2,
    invertLr = true,   // config.yaml vision.invert_lr: true
    invertUd = false,
    deadzone = 0.05,
    lostFaceTimeoutS = 1.5,
    targetSmoothing = 0.35,
    confidenceAttack = 0.4,
  } = {}) {
    this.video = video;
    this.log = log;
    this.onStatus = onStatus;
    this.fps = fps;
    this.gainLr = gainLr;
    this.gainUd = gainUd;
    this.invertLr = invertLr;
    this.invertUd = invertUd;
    this.deadzone = deadzone;
    this.lostFaceTimeoutS = lostFaceTimeoutS;
    this.targetSmoothing = targetSmoothing;
    this.confidenceAttack = confidenceAttack;

    this.gaze = { target_lr: 0.5, target_ud: 0.5, confidence: 0 };
    this._stream = null;
    this._detector = null;
    this._timer = null;
    this._running = false;
    this._prevCenter = null;
  }

  isRunning() { return this._running; }

  // Snapshot handed to the motion scheduler's gazeProvider each tick.
  snapshot() {
    return {
      target_lr: this.gaze.target_lr,
      target_ud: this.gaze.target_ud,
      confidence: this._running ? this.gaze.confidence : 0,
    };
  }

  setInvert({ lr, ud } = {}) {
    if (typeof lr === "boolean") this.invertLr = lr;
    if (typeof ud === "boolean") this.invertUd = ud;
  }

  async start() {
    if (this._running) return;
    if (!faceDetectionSupported()) {
      throw new Error(
        "This browser has no native FaceDetector. Use Chrome/Edge (you may need " +
        "chrome://flags/#enable-experimental-web-platform-features)."
      );
    }
    this.onStatus({ state: "starting" });
    this._stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    if (this.video) {
      this.video.srcObject = this._stream;
      this.video.muted = true;
      this.video.playsInline = true;
      try { await this.video.play(); } catch (_) {}
    }
    // fastMode trades a little accuracy for the frame rate we want.
    this._detector = new window.FaceDetector({ maxDetectedFaces: 5, fastMode: true });
    this._running = true;
    this._prevCenter = null;
    this.gaze = { target_lr: 0.5, target_ud: 0.5, confidence: 0 };
    this.log(`vision: face tracking started (~${this.fps} Hz)`);
    this.onStatus({ state: "on", seeingFace: false });
    this._loop();
  }

  async stop() {
    this._running = false;
    if (this._timer) { clearTimeout(this._timer); this._timer = null; }
    if (this._stream) {
      for (const t of this._stream.getTracks()) { try { t.stop(); } catch (_) {} }
      this._stream = null;
    }
    if (this.video) { try { this.video.srcObject = null; } catch (_) {} }
    this._detector = null;
    // Drop confidence so the head eases back to procedural motion.
    this.gaze.confidence = 0;
    this._prevCenter = null;
    this.log("vision: face tracking stopped");
    this.onStatus({ state: "off" });
  }

  _loop() {
    const periodMs = 1000 / Math.max(1, this.fps);
    const tick = async () => {
      if (!this._running) return;
      const t0 = performance.now();
      try {
        await this._detectOnce(periodMs / 1000);
      } catch (e) {
        // A bad frame / transient detector error must not kill the loop.
        this.log(`vision: detect tick failed (${e.message || e})`);
      }
      if (!this._running) return;
      const elapsed = performance.now() - t0;
      this._timer = setTimeout(tick, Math.max(0, periodMs - elapsed));
    };
    tick();
  }

  async _detectOnce(periodS) {
    const v = this.video;
    const vw = v && v.videoWidth;
    const vh = v && v.videoHeight;
    let primary = null;
    if (this._detector && vw && vh) {
      const raw = await this._detector.detect(v);
      const faces = raw.map((d) => {
        const b = d.boundingBox;
        const area = (b.width * b.height) / (vw * vh);
        return { cx: (b.x + b.width / 2) / vw, cy: (b.y + b.height / 2) / vh, area };
      });
      primary = selectPrimary(faces, this._prevCenter, 0.25);
      if (primary) this._prevCenter = [primary.cx, primary.cy];
    }

    if (primary) {
      const { lr, ud } = mapFaceToGaze(primary, {
        gainLr: this.gainLr, gainUd: this.gainUd,
        invertLr: this.invertLr, invertUd: this.invertUd, deadzone: this.deadzone,
      });
      const s = clamp01(this.targetSmoothing);
      this.gaze.target_lr += (lr - this.gaze.target_lr) * s;
      this.gaze.target_ud += (ud - this.gaze.target_ud) * s;
      const a = clamp01(this.confidenceAttack);
      this.gaze.confidence += (1 - this.gaze.confidence) * a;
      this.onStatus({ state: "on", seeingFace: true });
    } else {
      if (this.lostFaceTimeoutS > 0) {
        this.gaze.confidence *= Math.exp(-periodS / this.lostFaceTimeoutS);
      } else {
        this.gaze.confidence = 0;
      }
      if (this.gaze.confidence < 1e-3) {
        this.gaze.confidence = 0;
        this._prevCenter = null;
      }
      this.onStatus({ state: "on", seeingFace: false });
    }
  }
}

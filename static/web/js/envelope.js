// Port of motion/envelope.py:EnvelopeFollower (process_rms path) to JS.
// See that file for the canonical documentation; the math here must
// stay in lockstep so browser mode feels like local mode.

export const DEFAULT_JAW_CALIBRATION = {
  floor: 0.08,
  ceiling: 0.90,
  noiseFloor: 0.010,
  attack: 0.55,
  release: 0.25,
  peakHoldMs: 60,
  gain: 1.6,
};

export class EnvelopeFollower {
  constructor(calibration = {}) {
    this.calibration = { ...DEFAULT_JAW_CALIBRATION, ...calibration };
    this._smoothed = 0;
    this._peak = 0;
    this._peakHoldUntil = 0;
  }

  reset() { this._smoothed = 0; this._peak = 0; this._peakHoldUntil = 0; }

  setCalibration(patch) { this.calibration = { ...this.calibration, ...patch }; }

  // rms is the normalized RMS of this audio frame in [0,1].
  processRms(rms) {
    const cal = this.calibration;
    let r = Math.max(0, rms);
    if (r < cal.noiseFloor) r = 0;
    const target = Math.min(1, r * cal.gain);
    const coeff = target > this._smoothed ? cal.attack : cal.release;
    this._smoothed += Math.max(0, Math.min(1, coeff)) * (target - this._smoothed);
    const now = performance.now();
    if (this._smoothed >= this._peak || now >= this._peakHoldUntil) {
      this._peak = this._smoothed;
      if (cal.peakHoldMs > 0 && this._smoothed >= this._peak) {
        this._peakHoldUntil = now + cal.peakHoldMs;
      }
    }
    const value = Math.max(this._smoothed, this._peak);
    return this._mapToJaw(value);
  }

  _mapToJaw(smoothed) {
    const cal = this.calibration;
    if (smoothed <= 0) return cal.floor;
    const jaw = cal.floor + smoothed * (cal.ceiling - cal.floor);
    if (jaw < cal.floor) return cal.floor;
    if (jaw > cal.ceiling) return cal.ceiling;
    return jaw;
  }

  // Latest smoothed envelope (without the jaw-range mapping). Useful
  // for driving behavior animations (wings, head bobs) at full swing
  // while still letting the jaw stay inside its calibrated range.
  get behaviorEnvelope() { return Math.min(1, this._smoothed); }
}

// Compute RMS of a Float32Array chunk, returning a value in [0, 1].
export function rmsOf(float32) {
  if (!float32 || float32.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < float32.length; i++) {
    const s = float32[i];
    sum += s * s;
  }
  return Math.sqrt(sum / float32.length);
}

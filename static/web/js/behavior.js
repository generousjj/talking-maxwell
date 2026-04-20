// MVP behavior engine for the browser mode.
//
// Ports just enough of motion/behavior_engine.py to keep Maxwell lively:
//
//   - IDLE / LISTENING / THINKING: continuous raised-cosine wing flap
//     with ~4.3s period, plus two independent slow head sines on
//     non-harmonic periods (yaw ~7.9s, pitch ~11.3s) so he never
//     looks frozen.
//   - SPEAKING: jaw driven by the envelope follower; wings also
//     modulated by the behavior envelope; slower head sines on top
//     for a conversational head-bob.
//
// The text-driven niceties from the Python engine (phrase-boundary
// nods, question_like tilt, emphasis spikes, excited word detection)
// are not ported — they depend on having the full utterance text up
// front, which we don't reliably get in a WebRTC stream. This is the
// documented parity gap.

const TAU = Math.PI * 2;

function clamp01(x) { if (x < 0) return 0; if (x > 1) return 1; return x; }

function raisedCosine(phase) {
  // phase in [0,1); returns a 0..1..0 pulse.
  return 0.5 - 0.5 * Math.cos(TAU * phase);
}

export const DEFAULT_GAINS = {
  headLrDrift: 0.22,
  headUdDrift: 0.15,
  wingStrength: 0.85,
  idleWingPeriodS: 4.3,
  yawPeriodS: 7.9,
  pitchPeriodS: 11.3,
  speakingBobStrength: 0.09,
  speakingWingStrength: 0.55,
};

export class BehaviorEngine {
  constructor({ envelope, gains = {}, now = () => performance.now() / 1000 } = {}) {
    this.envelope = envelope;           // EnvelopeFollower instance
    this.gains = { ...DEFAULT_GAINS, ...gains };
    this._now = now;
    this.state = "idle";
    this._stateSince = this._now();
  }

  setState(s) {
    if (s === this.state) return;
    this.state = s;
    this._stateSince = this._now();
  }

  updateGains(patch) { this.gains = { ...this.gains, ...patch }; }

  // Returns { jaw, head_lr, head_ud, wing }, each in [0,1].
  tick() {
    const t = this._now();
    const env = this.envelope ? this.envelope.behaviorEnvelope : 0;

    if (this.state === "speaking") {
      // Jaw: pull the calibrated jaw value from the follower.
      const jaw = this.envelope ? this.envelope._mapToJaw(this.envelope._smoothed) : 0;
      const headLr = 0.5 + this._yawSine(t) + env * 0.02;
      const headUd = 0.5 + this._pitchSine(t) * 0.7 + env * this.gains.speakingBobStrength;
      const wing = clamp01(env * this.gains.speakingWingStrength + 0.05 * this._flapPulse(t, 1.6));
      return { jaw: clamp01(jaw), head_lr: clamp01(headLr), head_ud: clamp01(headUd), wing };
    }

    // idle / listening / thinking all share the continuous-motion look
    const wing = clamp01(this._flapPulse(t, this.gains.idleWingPeriodS) * this.gains.wingStrength);
    const headLr = clamp01(0.5 + this._yawSine(t));
    const headUd = clamp01(0.5 + this._pitchSine(t));
    return { jaw: 0, head_lr: headLr, head_ud: headUd, wing };
  }

  _yawSine(t) {
    const p = this.gains.yawPeriodS;
    return Math.sin(TAU * (t / p)) * this.gains.headLrDrift;
  }
  _pitchSine(t) {
    const p = this.gains.pitchPeriodS;
    // non-zero phase so it desyncs from yaw
    return Math.sin(TAU * (t / p) + 1.23) * this.gains.headUdDrift;
  }
  _flapPulse(t, periodS) {
    // Raised cosine pulse for each cycle.
    const phase = (t % periodS) / periodS;
    return raisedCosine(phase);
  }
}

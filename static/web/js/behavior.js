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

// Defaults mirror motion.behavior in config.yaml. Keep these in lockstep
// with DEFAULT_GAINS in app/motion_config.py — that's the Python-side
// source of truth that actually reads the yaml at runtime.
export const DEFAULT_GAINS = {
  headLrDrift:          0.32,   // head_lr_drift
  headUdDrift:          0.26,   // head_ud_drift
  wingStrength:         0.69,   // waiting_wing_strength (idle flap)
  speakingWingStrength: 1.0,    // wing_strength (speaking flap)
  speakingBobStrength:  0.22,   // envelope_head_bob
  idleNodStrength:      0.30,   // idle_nod_strength   (head_ud sine amp)
  idleTiltStrength:     0.20,   // idle_tilt_strength  (head_lr sine amp)
  idleNodPeriodS:       3.7,    // idle_nod_period_s
  idleTiltPeriodS:      5.1,    // idle_tilt_period_s
  idleWingPeriodS:      4.3,
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
      // Speaking head is idle sines + envelope-driven bob on pitch,
      // plus a small envelope-driven yaw sway. Amplitudes use the
      // full head_lr_drift / head_ud_drift range from config.yaml so
      // the head motion is visibly larger during speech than idle.
      const yaw = this._sineAmp(t, this.gains.idleTiltPeriodS, this.gains.headLrDrift);
      const pitch = this._sineAmp(t, this.gains.idleNodPeriodS, this.gains.headUdDrift, 1.23);
      const headLr = 0.5 + yaw + env * 0.05;
      const headUd = 0.5 + pitch + env * this.gains.speakingBobStrength;
      // Wing: envelope-driven with a slow flap underpulse so it never
      // flat-lines during long silences in a turn.
      const wing = clamp01(
        env * this.gains.speakingWingStrength
        + 0.15 * this._flapPulse(t, this.gains.idleWingPeriodS) * this.gains.wingStrength
      );
      return { jaw: clamp01(jaw), head_lr: clamp01(headLr), head_ud: clamp01(headUd), wing };
    }

    // idle / listening / thinking share the continuous-motion look:
    // head yaws on one sine, pitches on another (non-harmonic periods),
    // wings flap on a third. Nothing ever sits still.
    const yaw = this._sineAmp(t, this.gains.idleTiltPeriodS, this.gains.idleTiltStrength);
    const pitch = this._sineAmp(t, this.gains.idleNodPeriodS, this.gains.idleNodStrength, 1.23);
    const wing = clamp01(
      this._flapPulse(t, this.gains.idleWingPeriodS) * this.gains.wingStrength
    );
    return {
      jaw: 0,
      head_lr: clamp01(0.5 + yaw),
      head_ud: clamp01(0.5 + pitch),
      wing,
    };
  }

  _sineAmp(t, periodS, amp, phase = 0) {
    if (!(periodS > 0)) return 0;
    return Math.sin(TAU * (t / periodS) + phase) * amp;
  }
  _flapPulse(t, periodS) {
    // Raised cosine pulse for each cycle.
    const phase = (t % periodS) / periodS;
    return raisedCosine(phase);
  }
}

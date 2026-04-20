// Browser port of motion/behavior_engine.py. Aims for algorithmic
// parity with the Python engine — same state-machine branches
// (_idle / _listening / _thinking / _speaking), same drift/smoothing
// time constants, same phrase-boundary nod + emphasis-bump + wing-
// cooldown heuristics, same output-side head lowpass.
//
// Two things intentionally differ:
//   1. RNG. Python uses `random.Random` (Mersenne Twister); we use a
//      seeded mulberry32 when `seed` is an integer and Math.random
//      otherwise. That matches Python's "seed=null means unseeded"
//      contract and gives reproducible JS runs for the same integer
//      seed, at the cost of not being bit-identical to Python's MT.
//   2. `SpeakingContext` is fed in by the motion scheduler (via a
//      provider function), the same way Python does it — see
//      `live_speaking_context.js` for the JS port of Python's
//      `LiveSpeakingContext` in app/pipeline.py.
//
// All gains come from the server via /api/web/motion-config (which
// reads config.yaml), so tuning the operator build automatically
// tunes the browser build. Defaults below are the last-known-good
// values and must stay in sync with app/motion_config.DEFAULT_GAINS.

const TAU = Math.PI * 2;

function clamp01(x) { if (x < 0) return 0; if (x > 1) return 1; return x; }

// ---- RNG ----
// mulberry32 when seeded (deterministic), Math.random otherwise.
function makeRng(seed) {
  if (seed === null || seed === undefined) return () => Math.random();
  let a = (seed | 0) >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Defaults mirror motion.BehaviorGains + config.yaml overrides. Must
// stay identical to app/motion_config.DEFAULT_GAINS — the server
// endpoint wins at runtime; this just exists so a first-paint tick
// before the config arrives still behaves correctly.
export const DEFAULT_GAINS = {
  headLrDrift:         0.32,
  headUdDrift:         0.26,
  nodStrength:         0.45,
  emphasisStrength:    0.26,
  questionTilt:        0.32,
  wingStrength:        1.0,     // speaking flap (Python wing_strength)
  wingCooldownS:       2.0,
  waitingWingStrength: 0.69,    // idle flap
  waitingWingPeriodS:  2.16,
  envelopeHeadBob:     0.22,
  speakingDriftRate:   0.8,
  idleNodStrength:     0.30,
  idleTiltStrength:    0.20,
  idleNodPeriodS:      3.7,
  idleTiltPeriodS:     5.1,
  headSmoothingTauS:   0.08,
  seed:                null,

  // Legacy alias names still accepted for backward-compat with older
  // motion-config JSON. If present, they're merged into the canonical
  // names in the constructor.
  speakingWingStrength: null,
  speakingBobStrength:  null,
  idleWingPeriodS:      null,
};

// Conversation-state constants. Match motion.models.ConversationState.
export const STATE = Object.freeze({
  IDLE: "idle",
  LISTENING: "listening",
  THINKING: "thinking",
  SPEAKING: "speaking",
});

export class BehaviorEngine {
  constructor({ envelope, gains = {}, now = () => performance.now() / 1000 } = {}) {
    this.envelope = envelope;
    this.gains = this._mergeGains({ ...DEFAULT_GAINS, ...gains });
    this._now = now;
    this._rng = makeRng(this.gains.seed);
    this.state = STATE.IDLE;
    this._stateSince = this._now();

    this._last_wing_time = 0;
    this._yaw_drift = 0.5;
    this._yaw_target = 0.5;
    this._pitch_drift = 0.5;
    this._pitch_target = 0.5;
    this._last_tick = 0;
    this._nod_until = 0;
    this._nod_strength = 0;
    this._tilt_until = 0;
    this._tilt_direction = 1;
    this._head_lr_out = 0.5;
    this._head_ud_out = 0.5;
  }

  reset() {
    this._rng = makeRng(this.gains.seed);
    this._last_wing_time = 0;
    this._yaw_drift = this._yaw_target = 0.5;
    this._pitch_drift = this._pitch_target = 0.5;
    this._last_tick = 0;
    this._nod_until = this._tilt_until = 0;
    this._nod_strength = 0;
    this._tilt_direction = 1;
    this._head_lr_out = this._head_ud_out = 0.5;
  }

  setState(s) {
    if (s === this.state) return;
    this.state = s;
    this._stateSince = this._now();
  }

  updateGains(patch) {
    const merged = this._mergeGains({ ...this.gains, ...patch });
    const seedChanged = merged.seed !== this.gains.seed;
    this.gains = merged;
    if (seedChanged) this._rng = makeRng(this.gains.seed);
  }

  // Merge legacy JS-only alias keys into the canonical Python names
  // when the canonical ones are missing. Keeps old server responses
  // working during rollout.
  _mergeGains(g) {
    const out = { ...g };
    if (out.speakingWingStrength != null) out.wingStrength = out.speakingWingStrength;
    if (out.speakingBobStrength  != null) out.envelopeHeadBob = out.speakingBobStrength;
    if (out.idleWingPeriodS      != null) out.waitingWingPeriodS = out.idleWingPeriodS;
    if (out.waitingWingStrength  == null) out.waitingWingStrength = 0.69;
    return out;
  }

  // Main tick. Signature matches Python: tick(speakingContext) with
  // the engine holding `state` internally (vs. Python passing state
  // in; functionally equivalent). Returns {jaw, head_lr, head_ud, wing}.
  tick(speakingContext = null) {
    const now = this._now();
    const dt = this._last_tick
      ? Math.max(0, Math.min(0.5, now - this._last_tick))
      : 1 / 30;
    this._last_tick = now;
    this._update_drift(dt, this.state);

    let raw;
    if (this.state === STATE.SPEAKING) {
      raw = this._speaking(now, dt, speakingContext || DEFAULT_SPEAKING_CONTEXT);
    } else if (this.state === STATE.LISTENING) {
      raw = this._listening(now, dt);
    } else if (this.state === STATE.THINKING) {
      raw = this._thinking(now, dt);
    } else {
      raw = this._idle(now, dt);
    }

    // Output-side head lowpass (mirrors Python head_smoothing_tau_s).
    const tau = Math.max(0, this.gains.headSmoothingTauS || 0);
    if (tau > 0) {
      const follow = 1 - Math.exp(-dt / tau);
      this._head_lr_out += (raw.head_lr - this._head_lr_out) * follow;
      this._head_ud_out += (raw.head_ud - this._head_ud_out) * follow;
      return {
        jaw: clamp01(raw.jaw),
        head_lr: clamp01(this._head_lr_out),
        head_ud: clamp01(this._head_ud_out),
        wing: clamp01(raw.wing),
      };
    }
    this._head_lr_out = raw.head_lr;
    this._head_ud_out = raw.head_ud;
    return {
      jaw: clamp01(raw.jaw),
      head_lr: clamp01(raw.head_lr),
      head_ud: clamp01(raw.head_ud),
      wing: clamp01(raw.wing),
    };
  }

  // ---- drift (matches Python _update_drift) ----
  _update_drift(dt, state) {
    if (state !== STATE.SPEAKING) {
      this._yaw_target = 0.5;
      this._pitch_target = 0.5;
      const follow = 1 - Math.exp(-dt / 0.35);
      this._yaw_drift   += (0.5 - this._yaw_drift)   * follow;
      this._pitch_drift += (0.5 - this._pitch_drift) * follow;
      return;
    }
    const changeRate = this.gains.speakingDriftRate;
    if (this._rng() < changeRate * dt) {
      const m = this.gains.headLrDrift;
      this._yaw_target = clamp01(0.5 + this._uniform(-m, m));
    }
    if (this._rng() < changeRate * dt * 0.7) {
      const m = this.gains.headUdDrift;
      this._pitch_target = clamp01(0.5 + this._uniform(-m, m));
    }
    const follow = 1 - Math.exp(-dt / 0.45);
    this._yaw_drift   += (this._yaw_target   - this._yaw_drift)   * follow;
    this._pitch_drift += (this._pitch_target - this._pitch_drift) * follow;
  }

  // ---- waiting wing (idle/listening/thinking flap) ----
  _waiting_wing(now) {
    const period = Math.max(0.2, this.gains.waitingWingPeriodS);
    const phase = (now % period) / period;
    const shape = 0.5 * (1 - Math.cos(TAU * phase));
    return this.gains.waitingWingStrength * shape;
  }

  // ---- idle head offsets (slow non-harmonic sines) ----
  _idle_head_offsets(now) {
    const nodPeriod  = Math.max(0.5, this.gains.idleNodPeriodS);
    const tiltPeriod = Math.max(0.5, this.gains.idleTiltPeriodS);
    const head_ud_off = -this.gains.idleNodStrength  * Math.sin(TAU * now / nodPeriod);
    const head_lr_off =  this.gains.idleTiltStrength * Math.sin(TAU * now / tiltPeriod);
    return [head_lr_off, head_ud_off];
  }

  _idle(now, _dt) {
    const [hlr, hud] = this._idle_head_offsets(now);
    return {
      jaw: 0,
      head_lr: clamp01(this._yaw_drift + hlr),
      head_ud: clamp01(this._pitch_drift + hud),
      wing:    clamp01(this._waiting_wing(now)),
    };
  }
  _listening(now, dt) { return this._idle(now, dt); }
  _thinking(now, dt)  { return this._idle(now, dt); }

  // ---- speaking ----
  _speaking(now, _dt, ctx) {
    if (ctx.phrase_boundary) {
      this._nod_until = now + 0.4;
      this._nod_strength = this.gains.nodStrength;
      if (ctx.question_like && this._rng() < 0.6) {
        this._tilt_until = now + 0.7;
        this._tilt_direction = this._rng() < 0.5 ? 1 : -1;
      }
    }

    let emphasis_bump = 0;
    if (ctx.emphasis > 0.45) {
      emphasis_bump = this.gains.emphasisStrength * ctx.emphasis;
    }

    let nod = 0;
    if (now < this._nod_until) {
      const remaining = Math.max(0, this._nod_until - now);
      nod = -this._nod_strength * Math.sin(Math.PI * (1 - remaining / 0.4));
    }

    let tilt = 0;
    if (now < this._tilt_until) {
      const remaining = Math.max(0, this._tilt_until - now);
      tilt = this.gains.questionTilt
        * this._tilt_direction
        * Math.sin(Math.PI * (1 - remaining / 0.7));
    }

    const envelope_bob = -this.gains.envelopeHeadBob * ctx.envelope;

    let wing = 0;
    const can_wing =
      ctx.envelope > 0.45
      && (now - this._last_wing_time) > this.gains.wingCooldownS;
    if (can_wing) {
      const excited_boost = ctx.excited ? 0.35 : 0;
      if (this._rng() < 0.20 + excited_boost) {
        wing = this.gains.wingStrength;
        this._last_wing_time = now;
      }
    }

    return {
      jaw: clamp01(ctx.envelope + emphasis_bump),
      head_lr: clamp01(this._yaw_drift + tilt),
      head_ud: clamp01(this._pitch_drift + envelope_bob - emphasis_bump * 0.35 + nod),
      wing: clamp01(wing),
    };
  }

  _uniform(a, b) { return a + this._rng() * (b - a); }
}

// Default empty speaking context — matches Python SpeakingContext().
const DEFAULT_SPEAKING_CONTEXT = Object.freeze({
  envelope: 0,
  text: "",
  progress: 0,
  phrase_boundary: false,
  emphasis: 0,
  question_like: false,
  excited: false,
});

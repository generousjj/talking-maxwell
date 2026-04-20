// Browser port of app.pipeline.LiveSpeakingContext.
//
// The Python server-side pipeline feeds PCM frames from the TTS audio
// into this class; the class maintains its own "behavior envelope"
// (a higher-gain, attack/release-smoothed envelope separate from the
// jaw envelope — because the jaw runs a very low gain to match the
// physical servo, which would otherwise leave the wing/head thresholds
// unreachable) plus a per-frame "emphasis" value (unsmoothed rms*4).
// It also stages phrase-boundary events (from text heuristics or
// explicit TTS timing metadata) and exposes a `.snapshot(now)` that
// the motion scheduler feeds to BehaviorEngine._speaking.
//
// Browser-specific differences:
//   - The audio source is the remote realtime track (playback through
//     an <audio> element + AnalyserNode). `updateFromRms(rms)` is
//     called at ~50 Hz by realtime.js.
//   - Without reliable phrase-boundary timing for realtime streams,
//     we detect boundaries heuristically: a voiced onset after >=
//     0.5 s of silence counts as a new phrase.
//   - question_like / excited are fed from assistant transcript text
//     via `setUtterance`.

const _EXCITED_WORDS = new Set([
  "wow", "hello", "hi", "hey", "yay", "awesome",
  "amazing", "great", "whoa", "squawk", "polly", "cracker",
]);

export function analyzeText(text) {
  const stripped = (text || "").trim();
  const lowered = stripped.toLowerCase();
  const question = stripped.endsWith("?");
  const exclamation = stripped.endsWith("!");
  const words = lowered.match(/[a-z']+/g) || [];
  const excited = exclamation || words.some((w) => _EXCITED_WORDS.has(w));
  return { question_like: question, excited, words };
}

// Behavior-envelope gain. Matches app/pipeline.py (rms * 6.0). Kept
// independent from jaw gain so the wing / head_bob thresholds in
// BehaviorEngine stay reachable during conversational loudness.
const BEHAVIOR_ENVELOPE_GAIN = 6.0;
// Matches app/pipeline.py: emphasis = rms * 4, unsmoothed.
const EMPHASIS_GAIN = 4.0;
// Seconds of silence that count as a phrase boundary on the next onset.
const PHRASE_SILENCE_S = 0.5;
// "Voiced" threshold for onset detection on the jaw-mapped envelope.
const VOICED_THRESHOLD = 0.18;

export class LiveSpeakingContext {
  // `envelopeFollower` is the same EnvelopeFollower that drives the
  // jaw — we read its .calibration for attack/release but keep our
  // own smoothed value so the jaw gain doesn't suppress behavior.
  constructor({ envelopeFollower, now = () => performance.now() / 1000 } = {}) {
    this.envelopeFollower = envelopeFollower;
    this._now = now;
    this._behavior_smoothed = 0;
    this._latest_envelope = 0;
    this.emphasis = 0;
    this.text = "";
    this.progress = 0;
    this.question_like = false;
    this.excited = false;
    this.phrase_boundary = false;  // explicit (set by caller)
    this._phrase_boundaries = [];  // timestamps in seconds
    this._consumed = 0;
    this._audio_duration_s = 0;
    this._speakStart = 0;
    // Onset detection state (for realtime streams with no TTS timing).
    this._lastVoicedAt = 0;
  }

  setUtterance({ text = "", durationS = 0, boundaries = null } = {}) {
    this.text = text || "";
    const meta = analyzeText(this.text);
    this.question_like = meta.question_like;
    this.excited = meta.excited;
    this._audio_duration_s = Math.max(0, durationS);
    this._phrase_boundaries = Array.isArray(boundaries) ? [...boundaries] : [];
    this._consumed = 0;
    this._speakStart = this._now();
  }

  // Called when a new assistant reply streams in during realtime.
  // Lets us mid-turn update the "which words are being spoken" so
  // question_like/excited reflect the latest phrase.
  appendText(chunk) {
    if (!chunk) return;
    this.text = (this.text || "") + chunk;
    const meta = analyzeText(this.text);
    this.question_like = meta.question_like;
    this.excited = meta.excited;
  }

  // Called at audio tick rate from realtime.js with the latest RMS
  // value in [0, 1] (same math as the jaw envelope follower).
  updateFromRms(rms) {
    const cal = this.envelopeFollower.calibration;
    const r = Math.max(0, rms || 0);
    const target = Math.min(1, r * BEHAVIOR_ENVELOPE_GAIN);
    const coeff = target > this._behavior_smoothed
      ? Math.max(0, Math.min(1, cal.attack))
      : Math.max(0, Math.min(1, cal.release));
    this._behavior_smoothed += coeff * (target - this._behavior_smoothed);
    this._latest_envelope = this._behavior_smoothed;
    this.emphasis = Math.min(1, r * EMPHASIS_GAIN);

    // Heuristic phrase-boundary: voiced after >=0.5 s of silence.
    const now = this._now();
    if (this._latest_envelope >= VOICED_THRESHOLD) {
      const sinceLast = now - this._lastVoicedAt;
      if (this._lastVoicedAt > 0 && sinceLast >= PHRASE_SILENCE_S) {
        this.phrase_boundary = true;
      }
      this._lastVoicedAt = now;
    }
  }

  // Must produce a plain object matching Python SpeakingContext —
  // consumed once per behavior-engine tick. Also drains the
  // phrase_boundary flag(s) so each boundary fires exactly once.
  snapshot(now) {
    let boundary = false;
    if (this._phrase_boundaries.length && this._consumed < this._phrase_boundaries.length) {
      const nextAt = this._phrase_boundaries[this._consumed];
      const elapsed = this._audio_duration_s > 0
        ? this.progress * this._audio_duration_s
        : (now - this._speakStart);
      if (elapsed >= nextAt) {
        boundary = true;
        this._consumed += 1;
      }
    }
    if (this.phrase_boundary) {
      boundary = true;
      this.phrase_boundary = false;
    }
    return {
      envelope: this._latest_envelope,
      text: this.text,
      progress: this.progress,
      phrase_boundary: boundary,
      emphasis: this.emphasis,
      question_like: this.question_like,
      excited: this.excited,
    };
  }
}

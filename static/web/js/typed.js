// Typed fallback: the browser sends a text prompt to the backend,
// the backend runs LLM + TTS with its server-side key, and returns
// base64-encoded MP3 audio. We decode the audio, play it, and run
// the envelope follower off the decoded samples so the jaw still
// moves in sync. No raw API key ever touches the browser.

import { apiJson } from "./auth.js";

function base64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export class TypedSession {
  constructor({ envelope, behavior, speakingContext = null, log = () => {}, onState = () => {}, onTranscript = () => {} }) {
    this.envelope = envelope;
    this.behavior = behavior;
    // Optional LiveSpeakingContext. When provided we feed it the TTS
    // RMS so the browser behavior engine's jaw/head/wing (which read
    // ctx.envelope, not the jaw follower) actually move during typed
    // speech, matching realtime mode. Callers that don't pass one keep
    // the previous behavior.
    this.speakingContext = speakingContext;
    this.log = log;
    this.onState = onState;
    this.onTranscript = onTranscript;
    this.audioCtx = null;
    this._speaking = false;
    this._outputDeviceId = "";
  }

  async send(text, { voice = "ballad", outputDeviceId = "" } = {}) {
    if (!text || this._speaking) return;
    this._outputDeviceId = outputDeviceId || "";
    this.onTranscript({ role: "user", text, final: true });
    this.onState("thinking");
    this.behavior && this.behavior.setState("thinking");

    // Pre-warm the AudioContext while we're still inside the click/
    // keypress user-gesture window. The `await apiJson(...)` below
    // breaks the gesture chain, so if we leave context creation for
    // _speakMp3 it lands in the "suspended" state — audio never
    // actually plays AND the analyser reads silence, which is what
    // was making the jaw sit at 0 the entire reply.
    try {
      if (!this.audioCtx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (AC) this.audioCtx = new AC();
      }
      if (this.audioCtx && this.audioCtx.state === "suspended") {
        await this.audioCtx.resume().catch(() => {});
      }
    } catch (e) {
      this.log(`audio ctx setup: ${e.message || e}`);
    }

    let resp;
    try {
      resp = await apiJson("/api/web/typed", {
        method: "POST",
        body: { text, voice },
      });
    } catch (e) {
      this.log(`typed failed: ${e.message || e}`);
      this.onState("idle");
      this.behavior && this.behavior.setState("idle");
      return;
    }
    this.onTranscript({ role: "assistant", text: resp.text, final: true });
    // Prime phrase/emphasis heuristics (question_like, excited) from the
    // reply text so nods/tilts fire while he talks.
    if (this.speakingContext) this.speakingContext.setUtterance({ text: resp.text });
    await this._speakMp3(resp.audio_b64);
  }

  async _speakMp3(b64) {
    const bytes = base64ToBytes(b64);
    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const ctx = this.audioCtx;
    // Belt-and-braces resume in case send()'s pre-warm path didn't
    // run (e.g. very first send queued before the AudioContext class
    // was available).
    if (ctx.state === "suspended") {
      try { await ctx.resume(); } catch (_) {}
    }
    // Route to the chosen speaker when the browser supports per-context
    // output routing (Chrome 110+). Best-effort; falls back to default.
    if (this._outputDeviceId && typeof ctx.setSinkId === "function") {
      try { await ctx.setSinkId(this._outputDeviceId); } catch (_) {}
    }
    this.log(`typed: audio ctx state=${ctx.state}`);
    const buf = await ctx.decodeAudioData(bytes.buffer.slice(0));

    const src = ctx.createBufferSource();
    src.buffer = buf;
    const an = ctx.createAnalyser();
    an.fftSize = 1024;
    src.connect(an);
    an.connect(ctx.destination);

    this._speaking = true;
    this.onState("speaking");
    this.behavior && this.behavior.setState("speaking");
    this.envelope && this.envelope.reset();

    const poll = () => {
      if (!this._speaking) return;
      const tmp = new Float32Array(an.fftSize);
      an.getFloatTimeDomainData(tmp);
      let sum = 0;
      for (let i = 0; i < tmp.length; i++) sum += tmp[i] * tmp[i];
      const rms = Math.sqrt(sum / tmp.length);
      this.envelope && this.envelope.processRms(rms);
      this.speakingContext && this.speakingContext.updateFromRms(rms);
      requestAnimationFrame(poll);
    };
    poll();

    src.addEventListener("ended", () => {
      this._speaking = false;
      this.envelope && this.envelope.reset();
      this.speakingContext && this.speakingContext.updateFromRms(0);
      this.onState("idle");
      this.behavior && this.behavior.setState("idle");
    });
    src.start();
  }
}

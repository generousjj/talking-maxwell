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
  constructor({ envelope, behavior, log = () => {}, onState = () => {}, onTranscript = () => {} }) {
    this.envelope = envelope;
    this.behavior = behavior;
    this.log = log;
    this.onState = onState;
    this.onTranscript = onTranscript;
    this.audioCtx = null;
    this._speaking = false;
  }

  async send(text, { voice = "ballad" } = {}) {
    if (!text || this._speaking) return;
    this.onTranscript({ role: "user", text, final: true });
    this.onState("thinking");
    this.behavior && this.behavior.setState("thinking");

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
    await this._speakMp3(resp.audio_b64);
  }

  async _speakMp3(b64) {
    const bytes = base64ToBytes(b64);
    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const ctx = this.audioCtx;
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
      requestAnimationFrame(poll);
    };
    poll();

    src.addEventListener("ended", () => {
      this._speaking = false;
      this.envelope && this.envelope.reset();
      this.onState("idle");
      this.behavior && this.behavior.setState("idle");
    });
    src.start();
  }
}

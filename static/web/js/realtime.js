// OpenAI Realtime via WebRTC, using an ephemeral client-secret minted
// by POST /api/web/realtime/session on our backend. The browser holds
// only the short-lived `ek_...` token; the raw OPENAI_API_KEY stays
// on the server.
//
// Flow:
//   1. POST /api/web/realtime/session -> { client_secret, model, voice }
//   2. getUserMedia({ audio: true })
//   3. new RTCPeerConnection(); addTrack(micTrack); addTransceiver("audio", recvonly)
//   4. dataChannel = pc.createDataChannel("oai-events")
//   5. offer = await pc.createOffer(); await pc.setLocalDescription(offer)
//   6. POST https://api.openai.com/v1/realtime?model=... (SDP, Bearer ek_...)
//   7. answer = await resp.text(); await pc.setRemoteDescription({type:"answer",sdp:answer})
//   8. Remote audio track -> <audio> element for playback
//   9. Same track -> AudioContext -> AnalyserNode -> RMS -> envelope follower
//
// Interrupts: data-channel `response.cancel` on user speech start.

import { apiJson } from "./auth.js";
import { rmsOf } from "./envelope.js";

export class RealtimeSession {
  constructor({
    envelope,
    behavior,
    speakingContext = null,
    log = () => {},
    onState = () => {},
    onTranscript = () => {},
  }) {
    this.envelope = envelope;
    this.behavior = behavior;
    // Optional LiveSpeakingContext. When provided, every envelope tick
    // pumps the same RMS into it so BehaviorEngine sees behavior
    // envelope + emphasis + phrase_boundary + question_like/excited —
    // matching the Python operator pipeline's behavior inputs.
    this.speakingContext = speakingContext;
    this.log = log;
    this.onState = onState;
    this.onTranscript = onTranscript;

    this.pc = null;
    this.dc = null;
    this.micStream = null;
    this.remoteStream = null;
    this.audioEl = null;
    this.audioCtx = null;
    this.analyser = null;
    this._analysisBuf = null;
    this._analysisTimer = null;
    this._running = false;
    this._pttMode = false;
    this._micEnabled = true;
    this._pendingUser = "";
    this._pendingAssistant = "";
  }

  isRunning() { return this._running; }

  async start({ voice = "ballad", pttMode = false } = {}) {
    if (this._running) return;
    this._running = true;
    this._pttMode = pttMode;
    this.onState("connecting");

    const sess = await apiJson("/api/web/realtime/session", {
      method: "POST",
      body: { voice },
    });
    const token = sess.client_secret;
    const model = sess.model;

    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 24000,
        noiseSuppression: true,
        echoCancellation: true,
        autoGainControl: true,
      },
    });

    const pc = new RTCPeerConnection();
    this.pc = pc;

    // Recv-only audio transceiver — Maxwell's voice back from OpenAI.
    pc.addTransceiver("audio", { direction: "recvonly" });

    // Send mic.
    const micTrack = this.micStream.getAudioTracks()[0];
    pc.addTrack(micTrack, this.micStream);
    if (pttMode) micTrack.enabled = false;
    this._micEnabled = micTrack.enabled;

    // Data channel for realtime events (transcripts, turn control).
    const dc = pc.createDataChannel("oai-events");
    this.dc = dc;
    dc.addEventListener("open", () => this._onDcOpen());
    dc.addEventListener("message", (ev) => this._onDcMessage(ev));

    pc.addEventListener("track", (ev) => this._onRemoteTrack(ev));
    pc.addEventListener("connectionstatechange", () => {
      this.log(`webrtc state: ${pc.connectionState}`);
      if (["failed", "disconnected", "closed"].includes(pc.connectionState)) {
        this.onState("disconnected");
        this._running = false;
      }
    });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const sdpResp = await fetch(
      `https://api.openai.com/v1/realtime?model=${encodeURIComponent(model)}`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/sdp",
          "OpenAI-Beta": "realtime=v1",
        },
        body: offer.sdp,
      }
    );
    if (!sdpResp.ok) {
      const txt = await sdpResp.text();
      this._running = false;
      this.onState("error");
      throw new Error(`WebRTC SDP exchange failed: ${sdpResp.status} ${txt.slice(0, 200)}`);
    }
    const answer = { type: "answer", sdp: await sdpResp.text() };
    await pc.setRemoteDescription(answer);

    this.onState("connected");
    this.log("realtime connected");
  }

  async stop() {
    this._running = false;
    this._pendingAssistant = "";
    this._pendingUser = "";
    try { if (this.dc && this.dc.readyState === "open") this.dc.close(); } catch (_) {}
    this.dc = null;
    try { if (this.pc) this.pc.close(); } catch (_) {}
    this.pc = null;
    if (this.micStream) {
      for (const t of this.micStream.getTracks()) t.stop();
      this.micStream = null;
    }
    if (this._analysisTimer) { clearInterval(this._analysisTimer); this._analysisTimer = null; }
    if (this.audioCtx) { try { await this.audioCtx.close(); } catch (_) {} this.audioCtx = null; }
    if (this.audioEl) {
      try { this.audioEl.pause(); this.audioEl.srcObject = null; } catch (_) {}
      if (this.audioEl.parentNode) this.audioEl.parentNode.removeChild(this.audioEl);
      this.audioEl = null;
    }
    this.analyser = null;
    this.onState("idle");
  }

  pttDown() {
    this._pttMode = true;
    if (!this.micStream) return;
    for (const t of this.micStream.getAudioTracks()) t.enabled = true;
    this._micEnabled = true;
    // Clear server buffer and cancel any in-flight response on barge-in.
    this._dcSend({ type: "input_audio_buffer.clear" });
    this._dcSend({ type: "response.cancel" });
  }

  pttUp() {
    if (!this.micStream) return;
    this._dcSend({ type: "input_audio_buffer.commit" });
    this._dcSend({ type: "response.create" });
    for (const t of this.micStream.getAudioTracks()) t.enabled = false;
    this._micEnabled = false;
  }

  // --- internals ---

  _onDcOpen() {
    this.log("data channel open");
    const turn_detection = this._pttMode
      ? null
      : { type: "server_vad", threshold: 0.65, prefix_padding_ms: 300, silence_duration_ms: 650 };
    this._dcSend({
      type: "session.update",
      session: {
        voice: undefined,      // keep server default from our ephemeral session
        turn_detection,
        input_audio_transcription: { model: "whisper-1" },
      },
    });
  }

  _dcSend(obj) {
    if (!this.dc || this.dc.readyState !== "open") return;
    try { this.dc.send(JSON.stringify(obj)); } catch (e) { this.log(`dc send fail: ${e.message || e}`); }
  }

  _onDcMessage(ev) {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (_) { return; }
    const t = msg.type || "";
    if (t === "response.audio_transcript.delta") {
      this._pendingAssistant += msg.delta || "";
      // Feed assistant text into the speaking context so question_like
      // / excited reflect the utterance in flight (same heuristic as
      // the Python pipeline's analyze_text).
      if (this.speakingContext && msg.delta) {
        this.speakingContext.appendText(msg.delta);
      }
    } else if (t === "response.audio_transcript.done") {
      const text = (msg.transcript || this._pendingAssistant || "").trim();
      if (text) this.onTranscript({ role: "assistant", text, final: true });
      if (this.speakingContext) {
        this.speakingContext.setUtterance({ text });
      }
      this._pendingAssistant = "";
    } else if (t === "conversation.item.input_audio_transcription.completed") {
      const text = (msg.transcript || "").trim();
      if (text) this.onTranscript({ role: "user", text, final: true });
    } else if (t === "input_audio_buffer.speech_started") {
      this.behavior && this.behavior.setState("listening");
      this.onState("listening");
    } else if (t === "response.created") {
      this.behavior && this.behavior.setState("thinking");
      this.onState("thinking");
    } else if (t === "response.audio.delta" || t === "output_audio_buffer.started") {
      this.behavior && this.behavior.setState("speaking");
      this.onState("speaking");
    } else if (t === "response.done" || t === "output_audio_buffer.stopped") {
      this.behavior && this.behavior.setState("listening");
      this.onState("listening");
    } else if (t === "error") {
      this.log(`openai error: ${JSON.stringify(msg.error || msg)}`);
    }
  }

  _onRemoteTrack(ev) {
    const [stream] = ev.streams;
    this.remoteStream = stream;

    // Play through <audio> so the browser mixes it to the speakers.
    if (!this.audioEl) {
      const el = document.createElement("audio");
      el.autoplay = true;
      el.playsInline = true;
      document.body.appendChild(el);
      this.audioEl = el;
    }
    this.audioEl.srcObject = stream;

    // Tap the same stream for envelope analysis (jaw motion).
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.audioCtx = ctx;
    const src = ctx.createMediaStreamSource(stream);
    const an = ctx.createAnalyser();
    an.fftSize = 1024;
    src.connect(an);
    this.analyser = an;
    this._analysisBuf = new Float32Array(an.fftSize);
    // 50 Hz envelope updates is plenty for a servo; the behavior tick
    // runs at 30 Hz and picks up whatever the follower has smoothed.
    this._analysisTimer = setInterval(() => this._pumpEnvelope(), 20);
  }

  _pumpEnvelope() {
    if (!this.analyser || !this.envelope) return;
    this.analyser.getFloatTimeDomainData(this._analysisBuf);
    const rms = rmsOf(this._analysisBuf);
    this.envelope.processRms(rms);
    // Same RMS feeds the separate (higher-gain) behavior envelope.
    if (this.speakingContext) this.speakingContext.updateFromRms(rms);
  }
}

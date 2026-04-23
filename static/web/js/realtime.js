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
    // Last wall-clock timestamp the remote audio crossed a "voiced"
    // threshold. Used by the silence watchdog to safely transition
    // SPEAKING -> LISTENING after playback actually drains, in case
    // `output_audio_buffer.stopped` isn't emitted by the WebRTC peer.
    this._lastLoudAt = 0;
    this._speakingSince = 0;
    // PTT bookkeeping: track when the user started holding and whether
    // an assistant response is currently in flight, so we don't commit
    // empty buffers or stack `response.create` on top of an unfinished
    // turn. Without these guards, rapid taps spam:
    //   input_audio_buffer_commit_empty (0.00 ms buffered)
    //   conversation_already_has_active_response
    this._pttDownAt = 0;
    this._responseActive = false;
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
    this._responseActive = false;
    this._pttDownAt = 0;
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
    this._pttDownAt = Date.now();
    for (const t of this.micStream.getAudioTracks()) t.enabled = true;
    this._micEnabled = true;
    this._dcSend({ type: "input_audio_buffer.clear" });
    // Only cancel if a response is actually in progress — sending
    // `response.cancel` with nothing to cancel is harmless, but the
    // server still echoes back a "nothing to cancel" error on some
    // builds. Tracked via _responseActive which is flipped by
    // response.created / response.done messages.
    if (this._responseActive) {
      this._dcSend({ type: "response.cancel" });
      this._responseActive = false;
    }
  }

  pttUp() {
    if (!this.micStream) return;
    for (const t of this.micStream.getAudioTracks()) t.enabled = false;
    this._micEnabled = false;
    // If the hold was too short (< 150 ms) we almost certainly don't
    // have 100 ms of audio buffered server-side yet. Committing in
    // that case triggers `input_audio_buffer_commit_empty`. Just clear
    // the buffer and bail — no new response for this tap.
    const heldMs = this._pttDownAt ? (Date.now() - this._pttDownAt) : 0;
    this._pttDownAt = 0;
    if (heldMs < 150) {
      this._dcSend({ type: "input_audio_buffer.clear" });
      return;
    }
    // Don't stack another response on top of one that's already in
    // flight — the server will reject it with
    // `conversation_already_has_active_response`.
    if (this._responseActive) return;
    this._dcSend({ type: "input_audio_buffer.commit" });
    this._dcSend({ type: "response.create" });
    this._responseActive = true;
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
      this._responseActive = true;
      this.behavior && this.behavior.setState("thinking");
      this.onState("thinking");
    } else if (t === "response.done" || t === "response.cancelled" || t === "response.failed") {
      this._responseActive = false;
    } else if (t === "response.audio.delta" || t === "output_audio_buffer.started") {
      // Enter SPEAKING as soon as audio starts flowing. We stay here
      // until either the WebRTC output buffer actually drains OR the
      // envelope-silence watchdog catches sustained silence — NOT
      // on `response.done`, which only means the model finished
      // generating, not that playback finished. The old trigger was
      // dropping Maxwell back into idle ~1 s after speech started
      // and breathing right over the rest of the audio.
      //
      // Only stamp the speaking-start timer on the edge (state flip),
      // not on every audio delta — otherwise speakingMs would never
      // exceed the watchdog floor and Maxwell would get stuck SPEAKING.
      const wasSpeaking = this.behavior && this.behavior.state === "speaking";
      this.behavior && this.behavior.setState("speaking");
      this.onState("speaking");
      if (!wasSpeaking) this._speakingSince = performance.now();
      this._lastLoudAt = performance.now();
    } else if (t === "output_audio_buffer.stopped" || t === "output_audio_buffer.cleared") {
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

    // ---- SPEAKING -> LISTENING silence watchdog ----
    // OpenAI sometimes drops `output_audio_buffer.stopped` on WebRTC
    // peers (or the event name/timing varies). Without this watchdog
    // Maxwell could stay stuck SPEAKING forever after the last reply.
    // With it, we watch the post-jaw smoothed envelope and flip to
    // LISTENING only after >=1.1 s of continuous quiet AND >=1.5 s
    // since SPEAKING started (so a short gap right after the first
    // audio chunk doesn't immediately kick us out).
    const now = performance.now();
    // Jaw-path _smoothed is already attack/release-smoothed and sits
    // in [0, 1] — a reliable "is audio playing" signal.
    const live = this.envelope._smoothed > 0.05;
    if (live) this._lastLoudAt = now;
    if (this.behavior && this.behavior.state === "speaking") {
      const quietMs = now - this._lastLoudAt;
      const speakingMs = now - this._speakingSince;
      if (quietMs > 1100 && speakingMs > 1500) {
        this.behavior.setState("listening");
        this.onState("listening");
      }
    }
  }
}

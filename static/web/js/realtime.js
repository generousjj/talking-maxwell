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

  async start({ voice = "ballad", pttMode = false, micDeviceId = "", outputDeviceId = "" } = {}) {
    if (this._running) return;
    this._running = true;
    this._pttMode = pttMode;
    this._outputDeviceId = outputDeviceId || "";
    this.onState("connecting");

    const sess = await apiJson("/api/web/realtime/session", {
      method: "POST",
      body: { voice },
    });
    const token = sess.client_secret;
    const model = sess.model;

    // Pre-create the AudioContext now, while we're still inside the
    // click's user-gesture window. If we wait until the remote track
    // arrives (a few hundred ms later), Chrome's autoplay policy
    // forces the new context into "suspended" state and the analyser
    // reads silence for the first ~200 ms of Maxwell's reply —
    // exactly the "he just breathes at first" symptom.
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC && !this.audioCtx) {
        this.audioCtx = new AC();
        if (this.audioCtx.state === "suspended") {
          // Still inside the gesture, so this resolves immediately.
          await this.audioCtx.resume().catch(() => {});
        }
      }
    } catch (e) {
      this.log(`audio ctx setup: ${e.message || e}`);
    }

    const audioConstraints = {
      channelCount: 1,
      sampleRate: 24000,
      noiseSuppression: true,
      echoCancellation: true,
      autoGainControl: true,
    };
    if (micDeviceId) audioConstraints.deviceId = { exact: micDeviceId };
    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: audioConstraints,
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

    // GA Realtime API (post May 2026): SDP exchange moved from
    // `/v1/realtime?model=...` to `/v1/realtime/calls?model=...` and
    // the OpenAI-Beta header is no longer accepted.
    const sdpResp = await fetch(
      `https://api.openai.com/v1/realtime/calls?model=${encodeURIComponent(model)}`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/sdp",
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

  // Swap the audio output device on the fly (user picked a new
  // speaker mid-session). Stores the choice so reconnects still use
  // it. Returns a promise that resolves true on success.
  async setOutputDevice(deviceId) {
    this._outputDeviceId = deviceId || "";
    if (!this.audioEl || typeof this.audioEl.setSinkId !== "function") return false;
    try {
      await this.audioEl.setSinkId(this._outputDeviceId);
      return true;
    } catch (e) {
      this.log(`setSinkId failed: ${e.message || e}`);
      return false;
    }
  }

  pttDown() {
    this._pttMode = true;
    if (!this.micStream) return;
    this._pttDownAt = Date.now();
    for (const t of this.micStream.getAudioTracks()) t.enabled = true;
    this._micEnabled = true;
    this._dcSend({ type: "input_audio_buffer.clear" });

    // ---- barge-in ----
    // If Maxwell is mid-response (either still generating OR already
    // playing audio out of the local jitter buffer), treat this press
    // as an interruption. We need to stop three things independently:
    //   1) the server's response generator       -> response.cancel
    //   2) the server's output audio queue       -> output_audio_buffer.clear
    //   3) whatever is already buffered locally  -> mute the <audio>
    // #3 matters because WebRTC keeps a jitter buffer of already-
    // delivered frames; without this the user hears ~200 ms of
    // Maxwell blaring on top of their own voice after they press.
    const wasSpeaking = this.behavior && this.behavior.state === "speaking";
    if (this._responseActive) {
      this._dcSend({ type: "response.cancel" });
      this._responseActive = false;
    }
    if (wasSpeaking) {
      this._dcSend({ type: "output_audio_buffer.clear" });
      if (this.audioEl) {
        try { this.audioEl.muted = true; } catch (_) {}
      }
      // Flip the behavior engine out of speaking so the jaw stops
      // chomping along with the now-muted audio.
      if (this.behavior) {
        this.behavior.setState("listening");
        this.onState("listening");
      }
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
    // GA Realtime: session.update requires session.type and nests
    // turn_detection/transcription under session.audio.input.
    const turn_detection = this._pttMode
      ? null
      : { type: "server_vad", threshold: 0.65, prefix_padding_ms: 300, silence_duration_ms: 650 };
    this._dcSend({
      type: "session.update",
      session: {
        type: "realtime",
        audio: {
          input: {
            transcription: { model: "whisper-1" },
            turn_detection,
          },
        },
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
    // GA event names use "output_audio_transcript" / "output_audio".
    // Pre-GA used "audio_transcript" / "audio". Handle both so a
    // single binary serves both pre-GA and GA accounts.
    if (t === "response.audio_transcript.delta"
        || t === "response.output_audio_transcript.delta") {
      this._pendingAssistant += msg.delta || "";
      // Feed assistant text into the speaking context so question_like
      // / excited reflect the utterance in flight (same heuristic as
      // the Python pipeline's analyze_text).
      if (this.speakingContext && msg.delta) {
        this.speakingContext.appendText(msg.delta);
      }
    } else if (t === "response.audio_transcript.done"
            || t === "response.output_audio_transcript.done") {
      const text = (msg.transcript || this._pendingAssistant || "").trim();
      if (text) this.onTranscript({ role: "assistant", text, final: true });
      if (this.speakingContext) {
        this.speakingContext.setUtterance({ text });
      }
      this._pendingAssistant = "";
    } else if (t === "conversation.item.input_audio_transcription.completed"
            || t === "conversation.item.input_audio_transcription_completed"
            || t === "conversation.item.input_audio_transcription.done") {
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
    } else if (t === "response.audio.delta"
            || t === "response.output_audio.delta"
            || t === "output_audio_buffer.started") {
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
      // Clear the barge-in mute so this reply is audible. (pttDown
      // muted the <audio> element to kill whatever was still in the
      // jitter buffer from the previous turn.)
      if (this.audioEl && this.audioEl.muted) {
        try { this.audioEl.muted = false; } catch (_) {}
      }
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
    // Route remote audio to the user's chosen output device. Only
    // supported on Chromium + recent WebKit; failures are silent on
    // Safari iOS / older browsers.
    if (this._outputDeviceId && typeof this.audioEl.setSinkId === "function") {
      this.audioEl.setSinkId(this._outputDeviceId).catch((e) => {
        this.log(`setSinkId failed: ${e.message || e}`);
      });
    }

    // Reuse the AudioContext we created during start() so we're
    // never tapping a suspended context here. If start() didn't get
    // to create one (very old browser, etc.), fall back to a fresh
    // one and try to resume — but at that point the gesture window
    // is gone and the first reply will likely come in muted to the
    // analyser. That's acceptable degraded behavior.
    let ctx = this.audioCtx;
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      ctx = new AC();
      this.audioCtx = ctx;
    }
    const tryResume = () => {
      if (ctx.state === "suspended") {
        ctx.resume().catch((e) => this.log(`audio ctx resume: ${e.message || e}`));
      }
    };
    tryResume();
    if (ctx.addEventListener) {
      ctx.addEventListener("statechange", tryResume);
    }
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

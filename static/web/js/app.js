// Top-level wiring for the browser-mode operator page.
//
// Nothing here talks to OpenAI directly — every OpenAI call rides
// through /api/web/* so the raw key stays on the server.

import { apiJson, whoami, logout } from "./auth.js";
import { WebSerialTransport, MockTransport } from "./serial.js";
import { EnvelopeFollower, DEFAULT_JAW_CALIBRATION } from "./envelope.js";
import { BehaviorEngine, DEFAULT_GAINS } from "./behavior.js";
import { MotionScheduler } from "./motion.js";
import { RealtimeSession } from "./realtime.js";
import { TypedSession } from "./typed.js";
import { LiveSpeakingContext } from "./live_speaking_context.js";
import {
  listAudioDevices,
  renderDeviceSelect,
  persistSelectTo,
  outputPickerSupported,
} from "./audio_devices.js";

// ---- auth gate ----
(async () => {
  const me = await whoami();
  if (!me.authed) location.replace("/login");
})();

// ---- element refs ----
const $ = (id) => document.getElementById(id);
const logsEl = $("logs");
const convoEl = $("convo");

function log(msg) {
  const ts = new Date().toLocaleTimeString();
  logsEl.textContent += `[${ts}] ${msg}\n`;
  logsEl.scrollTop = logsEl.scrollHeight;
}

function appendBubble(role, text) {
  const b = document.createElement("div");
  b.className = `bubble ${role}`;
  b.textContent = text;
  convoEl.appendChild(b);
  convoEl.scrollTop = convoEl.scrollHeight;
}

// ---- motion pipeline ----
const envelope = new EnvelopeFollower(DEFAULT_JAW_CALIBRATION);
const behavior = new BehaviorEngine({ envelope, gains: DEFAULT_GAINS });
// LiveSpeakingContext tracks a higher-gain behavior envelope,
// per-frame emphasis, and phrase-boundary onsets — same set of inputs
// app/pipeline.py feeds the Python BehaviorEngine while speaking.
const speakingCtx = new LiveSpeakingContext({ envelopeFollower: envelope });
let transport = null;

// Live motion config (pins/PWM ranges, behavior gains, jaw calibration)
// is served from /api/web/motion-config which reads config.yaml on the
// server. Without this, browser mode used hardcoded defaults that
// drifted from the actual hardware wiring and made only wings move.
let motionConfig = null;
(async () => {
  try {
    const m = await apiJson("/api/web/motion-config", { method: "GET" });
    if (m && m.ok) {
      motionConfig = m;
      if (m.gains) behavior.updateGains(m.gains);
      if (m.jaw_calibration) envelope.setCalibration(m.jaw_calibration);
      log(`motion config loaded from ${m.source}`);
    }
  } catch (e) {
    log(`motion config fetch failed (${e.message || e}); using baked-in defaults`);
  }
})();
const scheduler = new MotionScheduler({
  hz: 30,
  behavior,
  transport: null,
  speakingContextProvider: () => speakingCtx.snapshot(performance.now() / 1000),
  onFrame: (f) => {
    const env = envelope.behaviorEnvelope;
    const envPct = Math.round(env * 100);
    $("envBar").style.width = `${envPct}%`;
    $("envVal").textContent = env.toFixed(2);
    const jawPct = Math.round(f.jaw * 100);
    $("jawBar").style.width = `${jawPct}%`;
    $("jawVal").textContent = f.jaw.toFixed(2);
  },
});
scheduler.start();

// ---- UI state ----
let rt = null;
let typed = null;
let serialState = "disconnected";

function setSerialState(s) {
  const dot = $("serialDot");
  dot.classList.remove("ok", "warn", "err");
  if (s.connected) {
    dot.classList.add("ok");
    $("serialLabel").textContent = s.mock ? "Mock (no hardware)" : "Connected";
    // Disable Connect while a real connection is alive so a second
    // click can't spin up a parallel transport and kill the first.
    $("connectBtn").disabled = true;
    $("disconnectBtn").disabled = false;
    $("wakeBtn").disabled = false;
    $("centerBtn").disabled = false;
    $("stopBtn").disabled = false;
    $("manualModeBtn").disabled = false;
    $("rtStartBtn").disabled = false;
    $("typedSendBtn").disabled = false;
    serialState = "connected";
  } else {
    // classList.add rejects empty strings, so only add a class when
    // we actually have one to add. Passing `""` here used to crash
    // the whole connect handler with `DOMTokenList.add: token must
    // not be empty`, which is why people saw "connected" but no
    // servo motion — the exception aborted scheduler.setTransport
    // and the frame loop never got a real transport.
    if (s.phase !== "closed") dot.classList.add("warn");
    $("serialLabel").textContent = s.phase === "closed" ? "Not connected" : (s.phase || "Disconnected");
    // Connect is available again only when we're fully disconnected,
    // not during phases like "handshake" where a connect is already
    // in flight (the inline _connectInFlight guard covers those).
    // "closed" = our own disconnect(). "disconnected" = OS yanked the
    // USB cable / driver dropped it.
    if (s.phase === "closed" || s.phase === "disconnected") {
      $("connectBtn").disabled = false;
    }
    $("disconnectBtn").disabled = true;
    $("wakeBtn").disabled = true;
    $("centerBtn").disabled = true;
    $("stopBtn").disabled = true;
    // Manual mode only makes sense while connected; un-check it so
    // reconnects always start in driven mode.
    $("manualModeBtn").disabled = true;
    $("manualModeBtn").checked = false;
    serialState = "disconnected";
  }
}

function setSessionState(s) {
  const dot = $("sessionDot");
  dot.classList.remove("ok", "warn", "err");
  $("sessionLabel").textContent = s;
  if (["connected", "listening", "thinking", "speaking"].includes(s)) dot.classList.add("ok");
  else if (s === "connecting") dot.classList.add("warn");
  else if (s === "error") dot.classList.add("err");
}

// ---- compat checks ----
(function compatCheck() {
  const missing = [];
  if (!("serial" in navigator)) missing.push("Web Serial (use Chrome/Edge over HTTPS)");
  if (!window.RTCPeerConnection) missing.push("WebRTC");
  if (!window.isSecureContext) missing.push("secure context (HTTPS)");
  if (missing.length) {
    $("compatHint").textContent = `Heads up: browser is missing ${missing.join(", ")}. Some features will be unavailable.`;
  }
})();

// ---- connect / disconnect ----
function resolvedChannels() {
  // Start from the server-provided map (falls back to transport-side
  // defaults if motion-config hasn't loaded yet). If the operator
  // picked a jaw-pin override, patch only the jaw pin here — PWM
  // ranges / inversion / slew stay on whatever the server shipped.
  const base = motionConfig && motionConfig.channels
    ? JSON.parse(JSON.stringify(motionConfig.channels))
    : null;
  const override = getJawPinOverride();
  if (override != null && base && base.jaw) {
    base.jaw.pin = override;
    log(`jaw pin override -> ${override}`);
  }
  return base || undefined;
}

let _connectInFlight = false;
$("connectBtn").addEventListener("click", async () => {
  // Hard guard against double-clicks. Without these checks a second
  // click would (a) spin up a brand-new transport on the same USB
  // port, (b) time out its handshake because the first transport is
  // still holding the port, (c) call disconnect() on failure which
  // closes the port — killing the perfectly-good first connection.
  if (_connectInFlight) {
    log("connect: already in progress, ignoring duplicate click");
    return;
  }
  if (transport && transport.isConnected && transport.isConnected()) {
    log("connect: already connected, ignoring duplicate click");
    return;
  }
  _connectInFlight = true;
  $("connectBtn").disabled = true;
  try {
    const mock = $("mockBtn").checked;
    const channels = resolvedChannels();
    transport = mock
      ? new MockTransport({ log, onState: setSerialState })
      : new WebSerialTransport({ log, onState: setSerialState, channels });
    await transport.connect();
    scheduler.setTransport(transport);
  } catch (e) {
    log(`connect failed: ${e.message || e}`);
    setSerialState({ connected: false, phase: "closed" });
    transport = null;
    scheduler.setTransport(null);
  } finally {
    _connectInFlight = false;
    // Re-enable the Connect button only if we aren't successfully
    // connected — the connected-state handler in setSerialState
    // already handles the "connected" branch by leaving it disabled.
    if (!transport || !transport.isConnected || !transport.isConnected()) {
      $("connectBtn").disabled = false;
    }
  }
});

$("disconnectBtn").addEventListener("click", async () => {
  if (!transport) return;
  await transport.disconnect();
  scheduler.setTransport(null);
  transport = null;
});

$("wakeBtn").addEventListener("click", async () => {
  if (transport && transport.isConnected()) await transport.wakeSweep();
});
$("centerBtn").addEventListener("click", async () => {
  if (transport && transport.isConnected()) await transport.center();
});
$("stopBtn").addEventListener("click", async () => {
  if (transport && transport.isConnected()) await transport.safeStop();
});
$("manualModeBtn").addEventListener("change", async (ev) => {
  const want = ev.target.checked;
  if (!transport || !transport.isConnected()) {
    ev.target.checked = false;
    return;
  }
  ev.target.disabled = true;
  try {
    await transport.setManualMode(want);
  } catch (e) {
    log(`manual mode toggle failed: ${e.message || e}`);
    ev.target.checked = !want;
  } finally {
    ev.target.disabled = false;
  }
});

// ---- sliders ----
$("jawGain").addEventListener("input", (ev) => {
  envelope.setCalibration({ gain: parseFloat(ev.target.value) });
});
$("wingStrength").addEventListener("input", (ev) => {
  behavior.updateGains({ wingStrength: parseFloat(ev.target.value) });
});
$("headDrift").addEventListener("input", (ev) => {
  const v = parseFloat(ev.target.value);
  behavior.updateGains({ headLrDrift: v, headUdDrift: v * 0.7 });
});

// Persist harmless UI prefs (never secrets).
const PREF_KEY = "maxwell_web_prefs_v1";
function savePrefs() {
  try {
    localStorage.setItem(PREF_KEY, JSON.stringify({
      voice: $("voiceSelect").value,
      mode: $("modeSelect").value,
      micMode: $("micMode").value,
      jawGain: $("jawGain").value,
      wingStrength: $("wingStrength").value,
      headDrift: $("headDrift").value,
      jawPin: $("jawPinSelect").value,
      jawPinCustom: $("jawPinCustom").value,
    }));
  } catch (_) {}
}
function loadPrefs() {
  try {
    const raw = localStorage.getItem(PREF_KEY);
    if (!raw) return;
    const p = JSON.parse(raw);
    if (p.voice) $("voiceSelect").value = p.voice;
    if (p.mode) $("modeSelect").value = p.mode;
    if (p.micMode) $("micMode").value = p.micMode;
    if (p.jawGain) { $("jawGain").value = p.jawGain; envelope.setCalibration({ gain: parseFloat(p.jawGain) }); }
    if (p.wingStrength) { $("wingStrength").value = p.wingStrength; behavior.updateGains({ wingStrength: parseFloat(p.wingStrength) }); }
    if (p.headDrift) { $("headDrift").value = p.headDrift; const v = parseFloat(p.headDrift); behavior.updateGains({ headLrDrift: v, headUdDrift: v * 0.7 }); }
    if (p.jawPin != null) $("jawPinSelect").value = p.jawPin;
    if (p.jawPinCustom != null) $("jawPinCustom").value = p.jawPinCustom;
    applyJawPinVisibility();
  } catch (_) {}
}

// ---- jaw pin override ----
// Lets the operator rewire the jaw servo to a different GPIO without
// redeploying. Empty/"Default" == follow config.yaml; "custom" reads
// from the adjacent number input. Override takes effect next connect.
function applyJawPinVisibility() {
  const sel = $("jawPinSelect");
  const custom = $("jawPinCustom");
  custom.hidden = sel.value !== "custom";
}
function getJawPinOverride() {
  const v = $("jawPinSelect").value;
  if (v === "") return null;
  if (v === "custom") {
    const n = parseInt($("jawPinCustom").value, 10);
    return Number.isFinite(n) ? n : null;
  }
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
}
$("jawPinSelect").addEventListener("change", () => {
  applyJawPinVisibility();
  savePrefs();
});
$("jawPinCustom").addEventListener("input", savePrefs);

loadPrefs();
["voiceSelect", "modeSelect", "micMode", "jawGain", "wingStrength", "headDrift"].forEach(id =>
  $(id).addEventListener("change", savePrefs));

// ---- mode toggle ----
function applyMode() {
  const m = $("modeSelect").value;
  $("typedControls").hidden = m !== "typed";
  $("realtimeControls").hidden = m === "typed";
  applyMicMode();
}
function applyMicMode() {
  const ptt = $("micMode").value === "ptt";
  $("talkBtn").hidden = !ptt;
  $("talkBtn").disabled = !rt || !rt.isRunning();
}
$("modeSelect").addEventListener("change", applyMode);
$("micMode").addEventListener("change", applyMicMode);
applyMode();

// ---- audio device pickers ----
const MIC_DEVICE_KEY = "maxwell:operator:mic";
const OUT_DEVICE_KEY = "maxwell:operator:output";
const micSelect = $("micDeviceSelect");
const outSelect = $("outputDeviceSelect");

async function refreshAudioDeviceLists() {
  const { inputs, outputs } = await listAudioDevices();
  renderDeviceSelect(micSelect, inputs, MIC_DEVICE_KEY, { defaultLabel: "System default microphone" });
  if (outputPickerSupported()) {
    renderDeviceSelect(outSelect, outputs, OUT_DEVICE_KEY, { defaultLabel: "System default speaker" });
  } else {
    // Browser won't let us re-route output; show the select disabled
    // with a hint rather than silently pretending it works.
    outSelect.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(output routing not supported in this browser)";
    outSelect.appendChild(opt);
    outSelect.disabled = true;
  }
}
refreshAudioDeviceLists();
// Browser only hands us device labels after mic permission is granted,
// and the list changes whenever someone plugs in headphones. Re-render
// on both events so the dropdowns stay accurate.
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => refreshAudioDeviceLists());
}
persistSelectTo(MIC_DEVICE_KEY, micSelect);
persistSelectTo(OUT_DEVICE_KEY, outSelect, async (deviceId) => {
  // Apply to a live session immediately so users can audition speakers
  // without restarting realtime.
  if (rt && rt.isRunning() && typeof rt.setOutputDevice === "function") {
    await rt.setOutputDevice(deviceId);
  }
});

// ---- realtime ----
$("rtStartBtn").addEventListener("click", async () => {
  if (rt && rt.isRunning()) return;
  rt = new RealtimeSession({
    envelope,
    behavior,
    speakingContext: speakingCtx,
    log,
    onState: (s) => setSessionState(s),
    onTranscript: ({ role, text }) => appendBubble(role === "user" ? "user" : "bot", text),
  });
  try {
    await rt.start({
      voice: $("voiceSelect").value,
      pttMode: $("micMode").value === "ptt",
      micDeviceId: micSelect.value || "",
      outputDeviceId: outSelect.value || "",
    });
    // Now that we've actually held the mic once, device labels are
    // populated — re-render so the dropdowns show real names.
    refreshAudioDeviceLists();
    behavior.setState("listening");
    $("rtStopBtn").disabled = false;
    $("rtStartBtn").disabled = true;
    applyMicMode();
  } catch (e) {
    log(`realtime start failed: ${e.message || e}`);
    rt = null;
    setSessionState("error");
  }
});

$("rtStopBtn").addEventListener("click", async () => {
  if (!rt) return;
  await rt.stop();
  rt = null;
  behavior.setState("idle");
  setSessionState("idle");
  $("rtStartBtn").disabled = false;
  $("rtStopBtn").disabled = true;
  $("talkBtn").disabled = true;
});

// PTT button + global release listeners (matches the /admits UX in
// the local app — a quick tap should never leave the mic open).
function pttDown() { if (rt && rt.isRunning()) rt.pttDown(); }
function pttUp() { if (rt && rt.isRunning()) rt.pttUp(); }
$("talkBtn").addEventListener("mousedown", (e) => { e.preventDefault(); pttDown(); });
$("talkBtn").addEventListener("touchstart", (e) => { e.preventDefault(); pttDown(); }, { passive: false });
window.addEventListener("mouseup", pttUp);
window.addEventListener("touchend", pttUp);
window.addEventListener("touchcancel", pttUp);

// ---- typed ----
typed = new TypedSession({
  envelope,
  behavior,
  log,
  onState: (s) => setSessionState(s),
  onTranscript: ({ role, text }) => appendBubble(role === "user" ? "user" : "bot", text),
});
async function sendTyped() {
  const text = ($("typedInput").value || "").trim();
  if (!text) return;
  $("typedInput").value = "";
  await typed.send(text, { voice: $("voiceSelect").value });
}
$("typedSendBtn").addEventListener("click", sendTyped);
$("typedInput").addEventListener("keydown", (ev) => { if (ev.key === "Enter") sendTyped(); });

// ---- logout + unload ----
$("logoutBtn").addEventListener("click", () => logout());
window.addEventListener("beforeunload", async () => {
  try { if (rt) await rt.stop(); } catch (_) {}
  try { if (transport) await transport.disconnect(); } catch (_) {}
});

// Small session hint in the top bar.
(async () => {
  const me = await whoami();
  if (me.authed && me.exp) {
    const mins = Math.max(0, Math.round((me.exp - Date.now() / 1000) / 60));
    $("sessionHint").textContent = `session · ${mins}m`;
  }
})();

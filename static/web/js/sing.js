// Song lip-sync ("jukebox") page wiring.
//
// Flow: the guest searches a song -> we get a 30s preview URL (Spotify,
// with an iTunes fallback resolved server-side) -> we proxy + decode the
// clip in the browser -> we PRE-COMPUTE a vocal-band envelope timeline
// offline -> during playback we drive Maxwell's jaw from
// envelope[currentTime + lookAhead] so the motion is tightly synced and
// even leads the audio slightly to hide servo latency.
//
// Vocal isolation here is intentionally lightweight (mid-channel +
// vocal-band EQ + a mild bass sidechain + a noise gate) so everything
// runs instantly client-side. It biases toward sung vocals (usually
// centered, mid-frequency) without any heavy ML separation.
//
// Reuses the exact same motion pipeline as the operator/admits pages:
// EnvelopeFollower + BehaviorEngine + LiveSpeakingContext + MotionScheduler
// + WebSerialTransport, all fed from config.yaml via /api/web/motion-config.

import { apiJson, whoami } from "./auth.js";
import { WebSerialTransport } from "./serial.js";
import { EnvelopeFollower, DEFAULT_JAW_CALIBRATION } from "./envelope.js";
import { BehaviorEngine, DEFAULT_GAINS } from "./behavior.js";
import { MotionScheduler } from "./motion.js";
import { LiveSpeakingContext } from "./live_speaking_context.js";
import {
  listAudioDevices,
  renderDeviceSelect,
  persistSelectTo,
  applyOutputDevice,
  outputPickerSupported,
} from "./audio_devices.js";

(async () => {
  const me = await whoami();
  if (!me.authed) location.replace("/login?next=/sing");
})();

const $ = (id) => document.getElementById(id);

// ---- tuning constants ----
// Look-ahead: sample the envelope slightly into the future so the jaw
// opens just before the sound, which reads as "in sync" once you add
// serial + servo latency.
const LOOK_AHEAD_S = 0.10;
// Envelope timeline resolution (50 fps — finer than the 30 Hz motion tick).
const HOP_S = 0.02;
// Map the loud parts of the song to roughly speech-RMS scale. The
// behavior envelope applies a fixed 6x gain internally (see
// LiveSpeakingContext), so ~0.12 here lands near a full-but-not-pinned
// jaw swing and keeps headroom for emphasis.
const TARGET_PEAK = 0.12;
// Below this (post-normalization) the jaw fully closes — silences the
// jaw during intros / instrumental breaks.
const GATE_FLOOR = 0.012;

// ---- UI primitives ----

function setStatus(msg, cls) {
  const el = $("status");
  el.textContent = msg;
  el.className = cls || "";
}

// ---- motion pipeline (mirrors admits.js) ----

const silentLog = () => {};
const envelope = new EnvelopeFollower(DEFAULT_JAW_CALIBRATION);
const behavior = new BehaviorEngine({ envelope, gains: DEFAULT_GAINS });
const speakingCtx = new LiveSpeakingContext({ envelopeFollower: envelope });
let transport = null;

let motionConfig = null;
const motionConfigReady = (async () => {
  try {
    const m = await apiJson("/api/web/motion-config", { method: "GET" });
    if (m && m.ok) {
      motionConfig = m;
      if (m.gains) behavior.updateGains(m.gains);
      if (m.jaw_calibration) envelope.setCalibration(m.jaw_calibration);
    }
  } catch (_) { /* fall back to baked-in defaults */ }
})();

const scheduler = new MotionScheduler({
  hz: 30,
  behavior,
  transport: null,
  speakingContextProvider: () => speakingCtx.snapshot(performance.now() / 1000),
  onFrame: () => {},
});
scheduler.start();

// ---- output device picker ----

const OUT_DEVICE_KEY = "maxwell:sing:output";

async function refreshAudioDeviceLists() {
  const { outputs } = await listAudioDevices();
  const outSel = $("outputDeviceSelect");
  if (outputPickerSupported()) {
    renderDeviceSelect(outSel, outputs, OUT_DEVICE_KEY, { defaultLabel: "Default speaker" });
  } else {
    outSel.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(not supported in this browser)";
    outSel.appendChild(opt);
    outSel.disabled = true;
  }
}
refreshAudioDeviceLists();
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => refreshAudioDeviceLists());
}
persistSelectTo(OUT_DEVICE_KEY, $("outputDeviceSelect"), async (deviceId) => {
  if (audioEl) await applyOutputDevice(audioEl, deviceId);
});

// ---- motion sliders (mirrors admits.js) ----

const MOTION_PREFS_KEY = "maxwell:sing:motion";

function setSliderDisplay(id, value) {
  const el = $(id + "Val");
  if (el) el.textContent = Number(value).toFixed(id === "jawGain" ? 1 : 2);
}
function applyJawGain(v) { envelope.setCalibration({ gain: v }); setSliderDisplay("jawGain", v); }
function applyWingStrength(v) { behavior.updateGains({ wingStrength: v }); setSliderDisplay("wingStrength", v); }
function applyHeadDrift(v) {
  behavior.updateGains({ headLrDrift: v, headUdDrift: v * 0.7 });
  setSliderDisplay("headDrift", v);
}
function saveMotionPrefs() {
  try {
    localStorage.setItem(MOTION_PREFS_KEY, JSON.stringify({
      jawGain: $("jawGain").value,
      wingStrength: $("wingStrength").value,
      headDrift: $("headDrift").value,
    }));
  } catch (_) {}
}
function loadMotionPrefs() {
  let p = {};
  try { p = JSON.parse(localStorage.getItem(MOTION_PREFS_KEY) || "{}"); } catch (_) {}
  if (p.jawGain != null) { $("jawGain").value = p.jawGain; }
  if (p.wingStrength != null) { $("wingStrength").value = p.wingStrength; }
  if (p.headDrift != null) { $("headDrift").value = p.headDrift; }
  applyJawGain(parseFloat($("jawGain").value));
  applyWingStrength(parseFloat($("wingStrength").value));
  applyHeadDrift(parseFloat($("headDrift").value));
}
$("jawGain").addEventListener("input", (e) => applyJawGain(parseFloat(e.target.value)));
$("wingStrength").addEventListener("input", (e) => applyWingStrength(parseFloat(e.target.value)));
$("headDrift").addEventListener("input", (e) => applyHeadDrift(parseFloat(e.target.value)));
["jawGain", "wingStrength", "headDrift"].forEach((id) =>
  $(id).addEventListener("change", saveMotionPrefs));
motionConfigReady.then(() => loadMotionPrefs());

// ---- serial connect (silent where possible, mirrors admits.js) ----

let serialReady = false;

function setSerialReady(ok) {
  serialReady = ok;
  const connectBtn = $("connectBtn");
  if (ok) connectBtn.classList.remove("visible");
  else connectBtn.classList.add("visible");
  refreshPlayButton();
}

async function connectHardware({ requireUserGesture } = { requireUserGesture: false }) {
  if (!WebSerialTransport.isSupported()) {
    setStatus("This browser can't talk to Maxwell. Open on Chrome or Edge.", "error");
    return;
  }
  const channels = (motionConfig && motionConfig.channels) || undefined;
  transport = new WebSerialTransport({ log: silentLog, onState: () => {}, channels });
  try {
    if (requireUserGesture) {
      await transport.connect();
    } else {
      const ok = await transport.tryAutoConnect();
      if (!ok) {
        setStatus("Tap \u201CConnect Maxwell\u201D once to hook up the USB cable.");
        setSerialReady(false);
        return;
      }
    }
    scheduler.setTransport(transport);
    setStatus("Ready \u2014 search for a song!");
    setSerialReady(true);
  } catch (e) {
    setStatus("Couldn't connect to Maxwell: " + (e.message || e), "error");
    setSerialReady(false);
  }
}

$("connectBtn").addEventListener("click", () =>
  connectHardware({ requireUserGesture: true }));
connectHardware({ requireUserGesture: false });

// ---- audio context + playback state ----

let actx = null;
let audioEl = null;          // <audio> element for actual sound output
let blobUrl = null;          // object URL backing the audio element
let timeline = null;         // { env: Float32Array, hopSec }
let currentTrack = null;
let playing = false;
let rafId = 0;

function ensureAudioContext() {
  if (!actx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (AC) actx = new AC();
  }
  if (actx && actx.state === "suspended") actx.resume().catch(() => {});
  return actx;
}

// ---- search ----

async function runSearch() {
  const q = ($("searchInput").value || "").trim();
  if (!q) return;
  const btn = $("searchBtn");
  btn.disabled = true;
  setStatus("Searching\u2026", "busy");
  try {
    const res = await apiJson("/api/web/song/search?q=" + encodeURIComponent(q) + "&limit=8");
    renderResults((res && res.tracks) || []);
    setStatus(
      (res.tracks && res.tracks.length) ? "Pick a song to play." : "No playable songs found \u2014 try another search.",
      "",
    );
  } catch (e) {
    const msg = (e && e.body && e.body.error === "spotify_not_configured")
      ? "Song search isn't set up on the server yet."
      : "Search failed: " + (e.message || e);
    setStatus(msg, "error");
    renderResults([]);
  } finally {
    btn.disabled = false;
  }
}

function renderResults(tracks) {
  const wrap = $("results");
  wrap.innerHTML = "";
  for (const t of tracks) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "track";
    const img = document.createElement("img");
    img.src = t.art || "/static/maxwell.png";
    img.alt = "";
    const meta = document.createElement("div");
    meta.className = "meta";
    const title = document.createElement("div");
    title.className = "t-title";
    title.textContent = t.title || "Unknown";
    const artist = document.createElement("div");
    artist.className = "t-artist";
    artist.textContent = t.artist || "";
    meta.appendChild(title);
    meta.appendChild(artist);
    card.appendChild(img);
    card.appendChild(meta);
    card.addEventListener("click", () => selectTrack(t));
    wrap.appendChild(card);
  }
}

$("searchBtn").addEventListener("click", runSearch);
$("searchInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); runSearch(); }
});

// ---- track selection: fetch + decode + precompute ----

async function selectTrack(track) {
  if (!track || !track.preview_url) return;
  stopPlayback();
  currentTrack = track;
  $("nowCard").style.display = "block";
  $("nowArt").src = track.art || "/static/maxwell.png";
  $("nowTitle").textContent = track.title || "Unknown";
  $("nowArtist").textContent = track.artist || "";
  $("playBtn").disabled = true;
  $("playBtn").textContent = "\u2026";
  setStatus("Loading the clip\u2026", "busy");

  // Create the AudioContext now (we're inside a click gesture) so
  // decode + later playback aren't blocked by autoplay suspension.
  ensureAudioContext();

  try {
    const resp = await fetch(
      "/api/web/song/audio?url=" + encodeURIComponent(track.preview_url),
      { credentials: "same-origin" },
    );
    if (!resp.ok) throw new Error("audio fetch HTTP " + resp.status);
    const bytes = await resp.arrayBuffer();

    // Decode a private copy for analysis (decodeAudioData detaches the
    // buffer it's given).
    const audioBuf = await actx.decodeAudioData(bytes.slice(0));
    setStatus("Analyzing the vocals\u2026", "busy");
    timeline = await precomputeVocalEnvelope(audioBuf);

    // Wire up a fresh <audio> element for playback so we get setSinkId
    // output routing + simple currentTime-based sync.
    if (blobUrl) { URL.revokeObjectURL(blobUrl); blobUrl = null; }
    const mime = resp.headers.get("content-type") || "audio/mpeg";
    blobUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
    setupAudioElement(blobUrl);

    $("playBtn").disabled = false;
    $("playBtn").textContent = "Play";
    setStatus(serialReady ? "Ready \u2014 hit play!" : "Connect Maxwell, then hit play.", "live");
  } catch (e) {
    setStatus("Couldn't load that song: " + (e.message || e), "error");
    $("playBtn").disabled = true;
    $("playBtn").textContent = "Play";
  }
}

function setupAudioElement(src) {
  if (audioEl) {
    audioEl.pause();
    audioEl.removeAttribute("src");
  }
  audioEl = new Audio();
  audioEl.src = src;
  audioEl.preload = "auto";
  audioEl.addEventListener("ended", () => stopPlayback());
  // Apply the chosen output device if the picker is supported.
  const dev = ($("outputDeviceSelect") && $("outputDeviceSelect").value) || "";
  applyOutputDevice(audioEl, dev);
}

// ---- offline vocal-envelope precompute ----

async function precomputeVocalEnvelope(audioBuf) {
  const sr = audioBuf.sampleRate;
  const len = audioBuf.length;

  // Mid channel = (L + R) / 2. Vocals usually sit dead-center, so the
  // mid channel already biases toward them and away from hard-panned
  // instruments.
  const mid = new Float32Array(len);
  const L = audioBuf.getChannelData(0);
  if (audioBuf.numberOfChannels > 1) {
    const R = audioBuf.getChannelData(1);
    for (let i = 0; i < len; i++) mid[i] = 0.5 * (L[i] + R[i]);
  } else {
    mid.set(L);
  }

  const band = await renderFiltered(mid, sr, len, [
    { type: "highpass", frequency: 200, Q: 0.7 },
    { type: "lowpass", frequency: 4000, Q: 0.7 },
  ]);
  // Low band ~ kick/bass that leaks into the vocal band's low edge; we
  // sidechain a fraction of it out so a thumping kick doesn't drive the
  // jaw.
  const low = await renderFiltered(mid, sr, len, [
    { type: "lowpass", frequency: 150, Q: 0.7 },
  ]);

  const hop = Math.max(1, Math.round(sr * HOP_S));
  const n = Math.ceil(len / hop);
  const raw = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s0 = i * hop;
    const s1 = Math.min(len, s0 + hop);
    let sv = 0, sl = 0;
    for (let j = s0; j < s1; j++) { sv += band[j] * band[j]; sl += low[j] * low[j]; }
    const cnt = Math.max(1, s1 - s0);
    const vr = Math.sqrt(sv / cnt);
    const lr = Math.sqrt(sl / cnt);
    raw[i] = Math.max(0, vr - 0.6 * lr);
  }

  // Robust normalize against the 95th percentile so loud and quiet
  // songs both reach a full jaw swing, then gate the quiet bits to 0.
  const sorted = Float32Array.from(raw).sort();
  const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] || 1e-6;
  const scale = p95 > 1e-6 ? TARGET_PEAK / p95 : 0;
  const env = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = raw[i] * scale;
    if (v < GATE_FLOOR) v = 0;
    env[i] = v;
  }
  return { env, hopSec: HOP_S };
}

async function renderFiltered(samples, sr, len, filters) {
  const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const offline = new OAC(1, len, sr);
  const buf = offline.createBuffer(1, len, sr);
  buf.copyToChannel(samples, 0);
  const src = offline.createBufferSource();
  src.buffer = buf;
  let node = src;
  for (const f of filters) {
    const biquad = offline.createBiquadFilter();
    biquad.type = f.type;
    biquad.frequency.value = f.frequency;
    if (f.Q != null) biquad.Q.value = f.Q;
    node.connect(biquad);
    node = biquad;
  }
  node.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

// ---- playback + motion drive ----

function startPlayback() {
  if (!timeline || !audioEl || playing) return;
  if (!serialReady) {
    setStatus("Connect Maxwell first so he can sing along.", "busy");
  }
  ensureAudioContext();
  envelope.reset();
  behavior.setState("speaking");
  playing = true;
  $("playBtn").textContent = "Stop";
  setStatus("Singing! \uD83C\uDFB6", "live");
  audioEl.currentTime = 0;
  audioEl.play().catch((e) => {
    setStatus("Couldn't play audio: " + (e.message || e), "error");
    stopPlayback();
  });
  driveMotion();
}

function driveMotion() {
  if (!playing) return;
  const t = (audioEl ? audioEl.currentTime : 0) + LOOK_AHEAD_S;
  const idx = Math.floor(t / timeline.hopSec);
  const v = (idx >= 0 && idx < timeline.env.length) ? timeline.env[idx] : 0;
  envelope.processRms(v);
  speakingCtx.updateFromRms(v);
  const dur = (audioEl && audioEl.duration) || 30;
  $("progressBar").style.width = Math.min(100, 100 * (audioEl ? audioEl.currentTime : 0) / dur) + "%";
  rafId = requestAnimationFrame(driveMotion);
}

function stopPlayback() {
  playing = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
  if (audioEl) { try { audioEl.pause(); } catch (_) {} }
  behavior.setState("idle");
  envelope.reset();
  speakingCtx.updateFromRms(0);
  $("progressBar").style.width = "0%";
  if (timeline) {
    $("playBtn").textContent = "Play";
    $("playBtn").disabled = false;
  }
  if (currentTrack) setStatus("Pick another song or replay.", "");
}

$("playBtn").addEventListener("click", () => {
  if (playing) stopPlayback();
  else startPlayback();
});

function refreshPlayButton() {
  // Play is allowed once a clip is loaded; we still nudge the guest to
  // connect Maxwell so the song doesn't play to a frozen bird.
  if (!timeline) return;
  $("playBtn").disabled = false;
}

// ---- cleanup ----

window.addEventListener("beforeunload", async () => {
  try { stopPlayback(); } catch (_) {}
  try { if (transport) await transport.disconnect(); } catch (_) {}
  if (blobUrl) { try { URL.revokeObjectURL(blobUrl); } catch (_) {} }
});

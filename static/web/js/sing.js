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
import { TypedSession } from "./typed.js";
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
// Map the loud parts of the song to roughly speech-RMS scale. The
// behavior envelope applies a fixed 6x gain internally (see
// LiveSpeakingContext), so ~0.12 here lands near a full-but-not-pinned
// jaw swing and keeps headroom for emphasis.
const TARGET_PEAK = 0.12;
// Peak-relative gate (fraction of the song's loud-peak) below which the
// jaw fully closes — silences the mouth during intros / instrumental
// breaks before the adaptive compression lifts everything above it.
const GATE_FLOOR_N = 0.10;

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

// Sing-mode playback state. Declared up here (before the scheduler) so
// the scheduler's onFrame override can safely read them on its very
// first tick. The jaw tracks the isolated VOCAL envelope; the rest of
// the body gets an explicit, energy-scaled "dance" overlay driven by the
// full-mix MUSIC envelope (continuous sway/bob LFOs + beat-synced wing
// pumps), because the conversational behavior engine's motion is far too
// subtle to read as dancing.
const TAU = Math.PI * 2;
let playing = false;
let jawSmoothed = 0;
let danceLevel = 0;   // slow-moving overall music energy (sway amplitude)
let danceBeat = 0;    // punchy per-beat energy (bob + wing pumps)
let voxLevel = 0;     // sustained vocal presence (0..1): high while singing
let yawTarget = 0;    // organic head-turn target in [-1,1], wanders randomly
let yawPos = 0;       // smoothed actual head turn -> avoids metronome left/right
// Discrete wing-flap "gestures": parrots snap their wings up on an accent
// and settle, they don't oscillate in/out forever. We trigger a short
// burst on a rising strong beat and otherwise keep the wings tucked.
let prevBeat = 0;
let wingStart = -10;  // time the current flap burst began (s)
let wingDur = 0.4;    // how long the burst lasts (s)
let wingAmp = 0;      // burst height (0..1), scaled by the accent
let wingFlaps = 1;    // 1 or 2 flaps in the burst
let lastWingAt = -10; // cooldown bookkeeping
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

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
  // While a song is playing we replace the body motion with a dance.
  // Two ideas keep it from looking like a metronome:
  //   - Head YAW is an organic random wander (yawPos), not a sine.
  //   - When he's actually singing words (voxLevel high) the body calms
  //     and faces forward (danceFactor shrinks), then opens back up to
  //     dance during instrumental breaks. The jaw carries the vocals.
  onFrame: (frame) => {
    if (!playing) return;
    const now = performance.now() / 1000;
    // Baseline groove (0.45) so he's never frozen, scaling up with energy.
    const amp = 0.45 + 0.55 * Math.min(1, danceLevel * 1.6);
    // 1.0 during instrumental, ~0.35 while sustaining a vocal line (kept
    // off the floor so there's still visible head motion when singing).
    const danceFactor = 1 - 0.65 * voxLevel;
    const sway = 0.40 * amp * danceFactor * yawPos;
    // Bob is mostly beat-driven (musical), plus a small groove sine; both
    // ease off while singing so the head stays forward-ish on long words.
    const beatBob = 0.26 * danceBeat * danceFactor;
    const grooveBob = 0.07 * amp * danceFactor * Math.sin(now * TAU * 1.1);
    // Gentle nod tied to the lyrics so "facing forward" during vocals
    // still has life (he nods along to the words) instead of going stiff.
    const singNod = 0.11 * voxLevel * jawSmoothed;
    // Wings: a small lifted "engaged" posture during energetic parts,
    // plus the discrete flap burst when one is active (see driveMotion).
    const rest = 0.12 * danceLevel;
    let wing = rest;
    const wt = (now - wingStart) / wingDur;
    if (wt >= 0 && wt < 1) {
      // wingFlaps half-sine humps that open and settle back, with a
      // slight decay so the burst tapers off naturally.
      const shape = Math.abs(Math.sin(Math.PI * wt * wingFlaps)) * (1 - 0.3 * wt);
      wing = Math.max(rest, wingAmp * shape);
    }
    frame.head_lr = clamp01(0.5 + sway);
    frame.head_ud = clamp01(0.5 - beatBob - grooveBob - singNod);
    frame.wing = clamp01(wing);
    frame.jaw = clamp01(jawSmoothed);
  },
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
let timeline = null;         // { env, musicEnv, hopSec }
let currentTrack = null;
let rafId = 0;

// ---- showtime (auto-play filler) state ----
let autoMode = false;
let fillerTimer = 0;
let fillerAttempts = 0;
let lastFillerQuery = "";
// Curated musical-theater search terms. Title + show keeps the search
// tight so we land on the right cast/soundtrack recording (which is what
// has playable previews). Picked for broad recognizability at the booth.
const SHOWTIME_QUERIES = [
  "Defying Gravity Wicked",
  "Popular Wicked",
  "Seasons of Love Rent",
  "One Day More Les Miserables",
  "I Dreamed a Dream Les Miserables",
  "Bring Him Home Les Miserables",
  "Alexander Hamilton",
  "My Shot Hamilton",
  "The Schuyler Sisters Hamilton",
  "You'll Be Back Hamilton",
  "Memory Cats musical",
  "The Phantom of the Opera",
  "Music of the Night Phantom",
  "All That Jazz Chicago",
  "Cabaret musical",
  "Tomorrow Annie",
  "Don't Rain on My Parade Funny Girl",
  "Summer Nights Grease",
  "You're the One That I Want Grease",
  "You Can't Stop the Beat Hairspray",
  "Tonight West Side Story",
  "America West Side Story",
  "Oklahoma musical",
  "Some Enchanted Evening South Pacific",
  "Waving Through a Window Dear Evan Hansen",
  "Corner of the Sky Pippin",
  "Sit Down You're Rockin' the Boat Guys and Dolls",
  "Maybe This Time Cabaret",
  "Suddenly Seymour Little Shop of Horrors",
  "Stars Les Miserables",
];

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
    card.addEventListener("click", () => {
      if (autoMode) setAutoMode(false);
      selectTrack(t);
    });
    wrap.appendChild(card);
  }
}

$("searchBtn").addEventListener("click", runSearch);
$("searchInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); runSearch(); }
});

// ---- track selection: fetch + decode + precompute ----

async function selectTrack(track, { auto = false } = {}) {
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
    if (auto && autoMode) {
      startPlayback();
    } else {
      setStatus(serialReady ? "Ready \u2014 hit play!" : "Connect Maxwell, then hit play.", "live");
    }
  } catch (e) {
    setStatus("Couldn't load that song: " + (e.message || e), "error");
    $("playBtn").disabled = true;
    $("playBtn").textContent = "Play";
    // In showtime mode a bad clip shouldn't dead-end the loop — skip on.
    if (auto && autoMode) scheduleNextFiller(1500);
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
  audioEl.addEventListener("ended", () => {
    stopPlayback();
    if (autoMode) scheduleNextFiller();
  });
  // Apply the chosen output device if the picker is supported.
  const dev = ($("outputDeviceSelect") && $("outputDeviceSelect").value) || "";
  applyOutputDevice(audioEl, dev);
}

// ---- offline vocal-envelope precompute (STFT vocal separation) ----
//
// A plain EQ can't tell a snare from a sung syllable — both have energy
// in the 200 Hz-4 kHz "vocal band", which is why the jaw used to pop on
// every drum hit. This does a light spectral separation instead, all
// offline on the decoded 30s clip:
//
//   1. CENTER extraction: vocals are (almost always) panned dead-center,
//      so per FFT bin we keep min(|L|,|R|) — the component common to both
//      channels. Hard-panned guitars/synths/stereo reverb largely drop
//      out.
//   2. VOCAL-BAND weighting: emphasize ~300 Hz-3.5 kHz (vocal formants +
//      presence), hard-cut sub-bass/kick (<150 Hz) and air/cymbals
//      (>5.5 kHz).
//   3. PERCUSSIVE suppression: drum hits are spectrally FLAT/broadband;
//      sung notes are PEAKY/harmonic. We weight each frame by
//      (1 - spectral_flatness), so kicks/snares/claps stop driving the
//      jaw while sustained vocals keep it moving.
//
// It's not a neural separator, but it tracks "is someone singing right
// now and how loud" far better than the band-pass did, and runs in well
// under a second so playback still starts near-instantly.

const FFT_SIZE = 2048;
const FFT_HOP = 512;

async function precomputeVocalEnvelope(audioBuf) {
  const sr = audioBuf.sampleRate;
  const len = audioBuf.length;
  const L = audioBuf.getChannelData(0);
  const R = audioBuf.numberOfChannels > 1 ? audioBuf.getChannelData(1) : L;

  const fft = makeFFT(FFT_SIZE);
  const half = FFT_SIZE >> 1;
  const win = hannWindow(FFT_SIZE);

  // Per-bin vocal-band weight + which bins to include in the flatness
  // (harmonic vs percussive) measure.
  const binHz = sr / FFT_SIZE;
  const bandW = new Float32Array(half);
  let flatLo = half, flatHi = 0;
  for (let k = 0; k < half; k++) {
    const f = k * binHz;
    bandW[k] = vocalBandWeight(f);
    if (f >= 150 && f <= 5500) { if (k < flatLo) flatLo = k; if (k > flatHi) flatHi = k; }
  }

  const reL = new Float32Array(FFT_SIZE), imL = new Float32Array(FFT_SIZE);
  const reR = new Float32Array(FFT_SIZE), imR = new Float32Array(FFT_SIZE);

  const nFrames = Math.max(1, Math.floor((len - FFT_SIZE) / FFT_HOP) + 1);
  const raw = new Float32Array(nFrames);
  // Full-range "music" energy of the whole mix (all frequencies, incl.
  // bass + drums). This is what makes the head/wings dance to the beat,
  // separate from the vocal envelope that drives the jaw.
  const musicRaw = new Float32Array(nFrames);

  for (let fr = 0; fr < nFrames; fr++) {
    const off = fr * FFT_HOP;
    // Broadband mix RMS over this hop (time domain) for the dance.
    let mixSq = 0, mixN = 0;
    const hopEnd = Math.min(len, off + FFT_HOP);
    for (let s = off; s < hopEnd; s++) { const m = 0.5 * (L[s] + R[s]); mixSq += m * m; mixN++; }
    musicRaw[fr] = mixN ? Math.sqrt(mixSq / mixN) : 0;

    for (let i = 0; i < FFT_SIZE; i++) {
      const w = win[i];
      const s = off + i;
      reL[i] = (s < len ? L[s] : 0) * w; imL[i] = 0;
      reR[i] = (s < len ? R[s] : 0) * w; imR[i] = 0;
    }
    fft(reL, imL);
    fft(reR, imR);

    let bandEnergy = 0;
    let logSum = 0, linSum = 0, flatN = 0;
    for (let k = 1; k < half; k++) {
      const magL = Math.hypot(reL[k], imL[k]);
      const magR = Math.hypot(reR[k], imR[k]);
      const centered = Math.min(magL, magR); // common (center-panned) part
      const masked = centered * bandW[k];
      bandEnergy += masked * masked;
      if (k >= flatLo && k <= flatHi) {
        const m = centered + 1e-9;
        logSum += Math.log(m);
        linSum += m;
        flatN++;
      }
    }

    // Spectral flatness in [0,1]: ~1 for flat/noisy (drums), ~0 for
    // peaky/harmonic (voice). harmonicWeight emphasizes the latter.
    let harmonicWeight = 1;
    if (flatN > 0 && linSum > 0) {
      const flatness = Math.exp(logSum / flatN) / (linSum / flatN);
      harmonicWeight = Math.pow(Math.max(0, 1 - flatness), 1.5);
    }
    raw[fr] = Math.sqrt(bandEnergy) * harmonicWeight;
  }

  // Light 1-pole smoothing so the vocal envelope tracks syllables, not
  // every FFT frame. The music envelope gets a snappier smoothing so the
  // beat stays punchy for the dance.
  for (let i = 1; i < nFrames; i++) raw[i] = 0.6 * raw[i] + 0.4 * raw[i - 1];
  for (let i = 1; i < nFrames; i++) musicRaw[i] = 0.5 * musicRaw[i] + 0.5 * musicRaw[i - 1];

  // Jaw envelope: gate the quiet bits, then adaptively compress so a
  // song whose syllables are mostly small still drives pronounced mouth
  // movement (the peak stays capped, the average gets pulled up).
  const vocalEnv = compressVocalEnvelope(raw, GATE_FLOOR_N);
  // Music envelope: only a tiny gate so the body keeps dancing through
  // purely instrumental sections.
  const musicEnv = normalizeEnvelope(musicRaw, 0.006);
  return { env: vocalEnv, musicEnv, hopSec: FFT_HOP / sr };
}

// Robust-normalize an energy array to TARGET_PEAK against its 95th
// percentile, then gate values below `gate` to zero.
function normalizeEnvelope(arr, gate) {
  const n = arr.length;
  const sorted = Float32Array.from(arr).sort();
  const p95 = sorted[Math.min(n - 1, Math.floor(n * 0.95))] || 1e-6;
  const scale = p95 > 1e-6 ? TARGET_PEAK / p95 : 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = arr[i] * scale;
    if (v < gate) v = 0;
    out[i] = v;
  }
  return out;
}

// Peak-normalize + gate, then adaptive dynamic-range compression: if the
// TYPICAL (median) voiced syllable is small, raise the whole average with
// a gamma<1 curve while leaving the loud peaks pinned at the cap. Songs
// that are already dynamic (healthy median) are left alone.
function compressVocalEnvelope(arr, gate) {
  const n = arr.length;
  const sorted = Float32Array.from(arr).sort();
  const p95 = sorted[Math.min(n - 1, Math.floor(n * 0.95))] || 1e-6;

  const norm = new Float32Array(n);
  const voiced = [];
  for (let i = 0; i < n; i++) {
    let v = p95 > 1e-6 ? arr[i] / p95 : 0;
    if (v > 1) v = 1;
    if (v < gate) v = 0;
    norm[i] = v;
    if (v > 0) voiced.push(v);
  }

  // Pick a gamma that maps the current median voiced level up toward
  // TARGET_MEDIAN. gamma<1 lifts small/mid values; gamma=1 is a no-op.
  let gamma = 1;
  if (voiced.length > 8) {
    voiced.sort((a, b) => a - b);
    const median = voiced[Math.floor(voiced.length / 2)] || 0;
    const TARGET_MEDIAN = 0.5;
    if (median > 0.02 && median < TARGET_MEDIAN) {
      gamma = Math.log(TARGET_MEDIAN) / Math.log(median);
      gamma = Math.max(0.4, Math.min(1, gamma)); // bound how aggressive it gets
    }
  }

  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = norm[i] > 0 ? Math.pow(norm[i], gamma) * TARGET_PEAK : 0;
  }
  return out;
}

// Smooth vocal-band weight: raised-cosine ramps in at 150->300 Hz and
// out at 3500->5500 Hz, flat (=1) across the formant/presence region.
function vocalBandWeight(f) {
  if (f <= 150 || f >= 5500) return 0;
  if (f < 300) return 0.5 * (1 - Math.cos(Math.PI * (f - 150) / 150));
  if (f > 3500) return 0.5 * (1 + Math.cos(Math.PI * (f - 3500) / 2000));
  return 1;
}

function hannWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return w;
}

// Compact iterative radix-2 Cooley-Tukey FFT. Returns a transform(re,im)
// that operates in-place on Float32Array buffers of length n (n a power
// of two). Twiddles + bit-reversal table are precomputed once per size.
function makeFFT(n) {
  const levels = Math.round(Math.log2(n));
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let x = i, r = 0;
    for (let j = 0; j < levels; j++) { r = (r << 1) | (x & 1); x >>= 1; }
    rev[i] = r >>> 0;
  }
  const cos = new Float32Array(n >> 1);
  const sin = new Float32Array(n >> 1);
  for (let i = 0; i < (n >> 1); i++) {
    cos[i] = Math.cos((-2 * Math.PI * i) / n);
    sin[i] = Math.sin((-2 * Math.PI * i) / n);
  }
  return function transform(re, im) {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const halfSize = size >> 1;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + halfSize; j++, k += step) {
          const tre = re[j + halfSize] * cos[k] - im[j + halfSize] * sin[k];
          const tim = re[j + halfSize] * sin[k] + im[j + halfSize] * cos[k];
          re[j + halfSize] = re[j] - tre;
          im[j + halfSize] = im[j] - tim;
          re[j] += tre;
          im[j] += tim;
        }
      }
    }
  };
}

// ---- playback + motion drive ----

function startPlayback() {
  if (!timeline || !audioEl || playing) return;
  if (!serialReady) {
    setStatus("Connect Maxwell first so he can sing along.", "busy");
  }
  ensureAudioContext();
  envelope.reset();
  jawSmoothed = 0;
  danceLevel = 0;
  danceBeat = 0;
  voxLevel = 0;
  yawTarget = 0;
  yawPos = 0;
  prevBeat = 0;
  wingStart = -10;
  lastWingAt = -10;
  wingAmp = 0;
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
  const vocalV = (idx >= 0 && idx < timeline.env.length) ? timeline.env[idx] : 0;
  const musicV = (timeline.musicEnv && idx >= 0 && idx < timeline.musicEnv.length)
    ? timeline.musicEnv[idx] : 0;

  // Dance energy: musicV is normalized so its loud parts sit near
  // TARGET_PEAK, so musicV/TARGET_PEAK is ~1 on a strong beat.
  const beat = Math.min(1.3, musicV / TARGET_PEAK);
  danceBeat = Math.max(beat, danceBeat * 0.82);  // fast attack, punchy decay
  danceLevel += 0.05 * (beat - danceLevel);      // slow overall energy

  // Sustained vocal presence: rises while a vocal line holds (so long
  // words settle the head forward), decays during instrumental gaps.
  const voxInstant = Math.min(1, vocalV / TARGET_PEAK);
  voxLevel += (voxInstant > voxLevel ? 0.10 : 0.04) * (voxInstant - voxLevel);

  // Organic head-turn: re-pick a random target occasionally and ease
  // toward it, so the yaw wanders naturally instead of ticking L-R-L-R.
  if (Math.random() < 0.016) yawTarget = Math.random() * 2 - 1;
  yawPos += 0.06 * (yawTarget - yawPos);

  // Wing-flap trigger: fire a discrete burst on a rising strong beat
  // (the accent's attack), respecting a cooldown so the wings don't just
  // oscillate. Two flaps on a really big hit, otherwise one.
  const nowS = performance.now() / 1000;
  const rising = danceBeat - prevBeat;
  prevBeat = danceBeat;
  if (danceBeat > 0.45 && rising > 0.10 && (nowS - lastWingAt) > 0.45) {
    lastWingAt = nowS;
    wingStart = nowS;
    wingAmp = Math.min(1, 0.55 + 0.5 * danceBeat);
    wingFlaps = danceBeat > 0.9 ? 2 : 1;
    wingDur = wingFlaps === 2 ? 0.6 : 0.4;
  }

  // Jaw mouths the isolated vocals. Same attack/release as the jaw
  // envelope follower, with the live jaw-gain slider as the multiplier.
  const cal = envelope.calibration;
  const gain = cal.gain || 6.0;
  const target = Math.min(1, vocalV * gain);
  const coeff = Math.max(0, Math.min(1, target > jawSmoothed ? cal.attack : cal.release));
  jawSmoothed += coeff * (target - jawSmoothed);

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
  jawSmoothed = 0;
  danceLevel = 0;
  danceBeat = 0;
  voxLevel = 0;
  yawPos = 0;
  yawTarget = 0;
  prevBeat = 0;
  wingStart = -10;
  lastWingAt = -10;
  wingAmp = 0;
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

// ---- showtime mode: auto-play random musical-theater clips ----
//
// Booth filler: while the switch is on we search a random showtune,
// pick a playable preview, decode + analyze it, and auto-play it.
// When the clip ends we queue the next one. Any manual interaction
// (picking a song, typing) flips the switch off so the operator can
// take over instantly.

function pickFillerQuery() {
  let q = lastFillerQuery;
  for (let i = 0; i < 6 && q === lastFillerQuery; i++) {
    q = SHOWTIME_QUERIES[Math.floor(Math.random() * SHOWTIME_QUERIES.length)];
  }
  lastFillerQuery = q;
  return q;
}

function scheduleNextFiller(delayMs) {
  if (fillerTimer) { clearTimeout(fillerTimer); fillerTimer = 0; }
  if (!autoMode) return;
  fillerTimer = setTimeout(() => {
    fillerTimer = 0;
    playRandomFiller();
  }, delayMs == null ? 900 : delayMs);
}

async function playRandomFiller() {
  if (!autoMode) return;
  const q = pickFillerQuery();
  setStatus("Showtime \uD83C\uDFAD finding a clip\u2026", "busy");
  let res;
  try {
    res = await apiJson("/api/web/song/search?q=" + encodeURIComponent(q) + "&limit=8");
  } catch (e) {
    if (autoMode) scheduleNextFiller(2000);
    return;
  }
  if (!autoMode) return; // toggled off mid-request
  const tracks = (res && res.tracks) || [];
  if (!tracks.length) {
    // This showtune had no playable preview; try another quickly, but
    // back off after several misses so we don't hammer the API.
    fillerAttempts += 1;
    scheduleNextFiller(fillerAttempts < 6 ? 350 : 2000);
    if (fillerAttempts >= 6) fillerAttempts = 0;
    return;
  }
  fillerAttempts = 0;
  renderResults(tracks);
  const track = tracks[Math.floor(Math.random() * tracks.length)];
  await selectTrack(track, { auto: true });
}

async function setAutoMode(on) {
  autoMode = !!on;
  const tog = $("autoToggle");
  if (tog) tog.checked = autoMode;
  if (autoMode) {
    ensureAudioContext();
    if (!serialReady) {
      try { await connectHardware({ requireUserGesture: false }); } catch (_) {}
    }
    fillerAttempts = 0;
    playRandomFiller();
  } else {
    if (fillerTimer) { clearTimeout(fillerTimer); fillerTimer = 0; }
    stopPlayback();
    setStatus("Showtime off. Pick a song or type something.", "");
  }
}

$("autoToggle").addEventListener("change", (e) => setAutoMode(e.target.checked));

// ---- "make Maxwell talk" (typed speech when not singing) ----
//
// Reuses the operator/admits TypedSession (LLM reply -> TTS mp3). We pass
// it our LiveSpeakingContext so the behavior engine's jaw/head/wings move
// while he talks — note the scheduler's onFrame dance override only kicks
// in while a song is `playing`, so during typed speech the normal
// conversational behavior drives the body, exactly what we want here.

const typed = new TypedSession({
  envelope,
  behavior,
  speakingContext: speakingCtx,
  // Verbatim text-to-speech — no LLM. Maxwell says exactly what's typed.
  endpoint: "/api/web/tts",
  log: silentLog,
  onState: (state) => {
    const btn = $("sayBtn");
    if (state === "thinking") {
      btn.disabled = true;
      $("sayReply").textContent = "Warming up Maxwell's voice…";
    } else if (state === "speaking") {
      btn.disabled = true;
    } else {
      btn.disabled = false;
    }
  },
  onTranscript: ({ role, text }) => {
    if (role === "assistant") $("sayReply").textContent = `Maxwell: “${text}”`;
  },
});

async function sendSay() {
  const input = $("sayInput");
  const text = (input.value || "").trim();
  if (!text) return;
  // Operator is taking over — kill showtime filler so it doesn't queue
  // another clip on top of him talking.
  if (autoMode) setAutoMode(false);
  // Don't talk over a song — stop singing first so the jaw is free.
  if (playing) stopPlayback();
  input.value = "";
  const dev = ($("outputDeviceSelect") && $("outputDeviceSelect").value) || "";
  await typed.send(text, { voice: "ballad", outputDeviceId: dev });
}

$("sayBtn").addEventListener("click", sendSay);
$("sayInput").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") { ev.preventDefault(); sendSay(); }
});

// ---- cleanup ----

window.addEventListener("beforeunload", async () => {
  try { stopPlayback(); } catch (_) {}
  try { if (transport) await transport.disconnect(); } catch (_) {}
  if (blobUrl) { try { URL.revokeObjectURL(blobUrl); } catch (_) {} }
});

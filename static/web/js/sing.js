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
    // Wings keep flapping while singing but harder during instrumentals.
    const flap = Math.min(1, 0.2 + 0.9 * danceBeat)
      * (0.45 + 0.55 * Math.sin(now * TAU * 2.0))
      * (0.55 + 0.45 * danceFactor);
    frame.head_lr = clamp01(0.5 + sway);
    frame.head_ud = clamp01(0.5 - beatBob - grooveBob - singNod);
    frame.wing = clamp01(flap);
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

  // Jaw envelope: gate the quiet bits (intros / instrumental breaks)
  // fully closed so the mouth doesn't flap when no one is singing.
  const vocalEnv = normalizeEnvelope(raw, GATE_FLOOR);
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

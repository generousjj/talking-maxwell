// End-user facing ("admits") page wiring.
//
// Reuses the same RealtimeSession + WebSerialTransport + motion
// pipeline as the operator page, but behind a big cartoon-bubble UI.
// No server/mock toggles, no sliders — just a big round button, a
// mic-mode switcher, and a chat transcript.
//
// Hardware connection: on load we try to re-open the previously
// authorized serial port silently (that's what tryAutoConnect() does).
// First-time booth setups still need the operator to click the
// "Connect Maxwell" button once to pick the USB port; once Chrome
// remembers it for this origin, every tab on the same laptop opens
// without the chooser.

import { apiJson, whoami } from "./auth.js";
import { WebSerialTransport } from "./serial.js";
import { EnvelopeFollower, DEFAULT_JAW_CALIBRATION } from "./envelope.js";
import { BehaviorEngine, DEFAULT_GAINS } from "./behavior.js";
import { MotionScheduler } from "./motion.js";
import { RealtimeSession } from "./realtime.js";
import { LiveSpeakingContext } from "./live_speaking_context.js";
import {
  listAudioDevices,
  renderDeviceSelect,
  persistSelectTo,
  outputPickerSupported,
} from "./audio_devices.js";

(async () => {
  const me = await whoami();
  if (!me.authed) location.replace("/login?next=/admits");
})();

const $ = (id) => document.getElementById(id);

// ---- UI primitives ----

function setStatus(msg, cls) {
  const el = $("status");
  el.textContent = msg;
  el.className = cls || "";
}

function bubble(role, text) {
  const empty = $("emptyHint");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = "bubble " + (role === "you" ? "you" : "bird");
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "you" ? "You" : "Maxwell";
  const body = document.createElement("span");
  body.textContent = text;
  div.appendChild(who);
  div.appendChild(body);
  $("chat").appendChild(div);
  $("chat").scrollTop = $("chat").scrollHeight;
}

function setActive(group, dataKey, value) {
  for (const b of group.querySelectorAll("button")) {
    b.classList.toggle("active", b.dataset[dataKey] === value);
  }
}

// ---- motion pipeline (silent log sink; admits never shows logs) ----

const silentLog = () => {};
const envelope = new EnvelopeFollower(DEFAULT_JAW_CALIBRATION);
const behavior = new BehaviorEngine({ envelope, gains: DEFAULT_GAINS });
const speakingCtx = new LiveSpeakingContext({ envelopeFollower: envelope });
let transport = null;

let motionConfig = null;
// Resolved once we've applied (or given up on) the server-side
// config.yaml. Downstream code (motion sliders) chains onto this
// so the user's localStorage overrides land *after* the config
// defaults, not before.
const motionConfigReady = (async () => {
  try {
    const m = await apiJson("/api/web/motion-config", { method: "GET" });
    if (m && m.ok) {
      motionConfig = m;
      if (m.gains) behavior.updateGains(m.gains);
      if (m.jaw_calibration) envelope.setCalibration(m.jaw_calibration);
    }
  } catch (_) { /* fallback to baked-in defaults */ }
})();

const scheduler = new MotionScheduler({
  hz: 30,
  behavior,
  transport: null,
  speakingContextProvider: () => speakingCtx.snapshot(performance.now() / 1000),
  onFrame: () => {},
});
scheduler.start();

// ---- audio device pickers ----

const MIC_DEVICE_KEY = "maxwell:admits:mic";
const OUT_DEVICE_KEY = "maxwell:admits:output";

async function refreshAudioDeviceLists() {
  const { inputs, outputs } = await listAudioDevices();
  renderDeviceSelect($("micDeviceSelect"), inputs, MIC_DEVICE_KEY, { defaultLabel: "Default microphone" });
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
persistSelectTo(MIC_DEVICE_KEY, $("micDeviceSelect"));
persistSelectTo(OUT_DEVICE_KEY, $("outputDeviceSelect"), async (deviceId) => {
  if (rt && rt.isRunning() && typeof rt.setOutputDevice === "function") {
    await rt.setOutputDevice(deviceId);
  }
});

// ---- motion sliders (mirrors the operator page controls) ----
// Stored in localStorage so a single booth laptop keeps the guest's
// last tuning across refreshes.

const MOTION_PREFS_KEY = "maxwell:admits:motion";

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
    const p = {
      jawGain: $("jawGain").value,
      wingStrength: $("wingStrength").value,
      headDrift: $("headDrift").value,
    };
    localStorage.setItem(MOTION_PREFS_KEY, JSON.stringify(p));
  } catch (_) {}
}

function loadMotionPrefs() {
  let p = {};
  try { p = JSON.parse(localStorage.getItem(MOTION_PREFS_KEY) || "{}"); } catch (_) {}
  if (p.jawGain != null)      { $("jawGain").value = p.jawGain;           applyJawGain(parseFloat(p.jawGain)); }
  else                        {                                          applyJawGain(parseFloat($("jawGain").value)); }
  if (p.wingStrength != null) { $("wingStrength").value = p.wingStrength; applyWingStrength(parseFloat(p.wingStrength)); }
  else                        {                                          applyWingStrength(parseFloat($("wingStrength").value)); }
  if (p.headDrift != null)    { $("headDrift").value = p.headDrift;       applyHeadDrift(parseFloat(p.headDrift)); }
  else                        {                                          applyHeadDrift(parseFloat($("headDrift").value)); }
}

$("jawGain").addEventListener("input", (e) => applyJawGain(parseFloat(e.target.value)));
$("wingStrength").addEventListener("input", (e) => applyWingStrength(parseFloat(e.target.value)));
$("headDrift").addEventListener("input", (e) => applyHeadDrift(parseFloat(e.target.value)));
["jawGain", "wingStrength", "headDrift"].forEach(id =>
  $(id).addEventListener("change", saveMotionPrefs));

// Apply slider prefs only after server-side motion config has landed,
// otherwise the async config fetch would clobber the user's saved
// values on every page load.
motionConfigReady.then(() => loadMotionPrefs());

// ---- event / context presets ----
// The selected context is appended to Maxwell's base personality
// server-side (see compose_instructions). Built-in events cover the
// recurring TEA showcases; guests/operators can also save their own
// custom prompts which persist in this browser.

const CUSTOM_PROMPTS_KEY = "maxwell:admits:customPrompts";
const EVENT_ID_KEY = "maxwell:admits:eventId";
const NEW_PROMPT_VALUE = "__new__";

const BUILTIN_EVENTS = [
  {
    id: "admit",
    label: "Admit Weekend Fair",
    context:
      "You are at the Stanford admit weekend fair, meeting admitted students "
      + "who are deciding whether to come to Stanford. Be warm and welcoming, "
      + "talk up Stanford TEA (Theater & Entertainment Arts) and the community.",
  },
  {
    id: "nso",
    label: "New Student Orientation (NSO)",
    context:
      "You are at New Student Orientation (NSO), greeting brand-new Stanford "
      + "students as they arrive on campus. Help them feel at home and tell "
      + "them about Stanford TEA and how to get involved.",
  },
  {
    id: "breadbowl",
    label: "Bay Area Breadbowl",
    context:
      "You are at the Bay Area Breadbowl, a theater showcase where students "
      + "present and showcase their work to theater and entertainment "
      + "professionals from around the Bay Area. Hype up the student performers "
      + "and the showcase, and be encouraging to anyone nervous about presenting.",
  },
];

function loadCustomPrompts() {
  try {
    const arr = JSON.parse(localStorage.getItem(CUSTOM_PROMPTS_KEY) || "[]");
    return Array.isArray(arr) ? arr : [];
  } catch (_) { return []; }
}

function saveCustomPrompts(list) {
  try { localStorage.setItem(CUSTOM_PROMPTS_KEY, JSON.stringify(list)); } catch (_) {}
}

function renderEventSelect() {
  const sel = $("eventSelect");
  if (!sel) return;
  const customs = loadCustomPrompts();
  const prev = sel.value || localStorage.getItem(EVENT_ID_KEY) || "admit";
  sel.innerHTML = "";

  const builtinGroup = document.createElement("optgroup");
  builtinGroup.label = "Events";
  for (const ev of BUILTIN_EVENTS) {
    const o = document.createElement("option");
    o.value = ev.id;
    o.textContent = ev.label;
    builtinGroup.appendChild(o);
  }
  sel.appendChild(builtinGroup);

  if (customs.length) {
    const customGroup = document.createElement("optgroup");
    customGroup.label = "Saved prompts";
    for (const c of customs) {
      const o = document.createElement("option");
      o.value = c.id;
      o.textContent = c.name || "Custom prompt";
      customGroup.appendChild(o);
    }
    sel.appendChild(customGroup);
  }

  const newGroup = document.createElement("optgroup");
  newGroup.label = "—";
  const newOpt = document.createElement("option");
  newOpt.value = NEW_PROMPT_VALUE;
  newOpt.textContent = "➕ New custom prompt…";
  newGroup.appendChild(newOpt);
  sel.appendChild(newGroup);

  const stillThere = Array.from(sel.options).some((o) => o.value === prev);
  sel.value = stillThere ? prev : "admit";
}

// Resolve the context text for whatever is currently selected.
function currentContextText() {
  const sel = $("eventSelect");
  if (!sel) return "";
  const v = sel.value;
  if (v === NEW_PROMPT_VALUE) {
    return ($("customPromptText").value || "").trim();
  }
  const builtin = BUILTIN_EVENTS.find((e) => e.id === v);
  if (builtin) return builtin.context;
  const custom = loadCustomPrompts().find((c) => c.id === v);
  return custom ? (custom.text || "") : "";
}

function syncCustomPromptEditor() {
  const sel = $("eventSelect");
  const editor = $("customPrompt");
  const delBtn = $("customPromptDelete");
  if (!sel || !editor) return;
  const v = sel.value;
  if (v === NEW_PROMPT_VALUE) {
    editor.classList.add("visible");
    $("customPromptName").value = "";
    $("customPromptText").value = "";
    delBtn.style.display = "none";
  } else {
    const custom = loadCustomPrompts().find((c) => c.id === v);
    if (custom) {
      editor.classList.add("visible");
      $("customPromptName").value = custom.name || "";
      $("customPromptText").value = custom.text || "";
      delBtn.style.display = "inline-block";
    } else {
      // Built-in event: nothing to edit.
      editor.classList.remove("visible");
    }
  }
}

async function onContextChanged() {
  const sel = $("eventSelect");
  if (sel && sel.value !== NEW_PROMPT_VALUE) {
    try { localStorage.setItem(EVENT_ID_KEY, sel.value); } catch (_) {}
  }
  syncCustomPromptEditor();
  // Context is baked into the session at mint time, so a running
  // session must be restarted to pick up the new event/prompt.
  if (rtIsRunning()) {
    await stopRealtime();
    await startRealtime();
  }
}

function wireEventControls() {
  renderEventSelect();
  syncCustomPromptEditor();
  $("eventSelect").addEventListener("change", onContextChanged);

  $("customPromptSave").addEventListener("click", () => {
    const name = ($("customPromptName").value || "").trim() || "Custom prompt";
    const text = ($("customPromptText").value || "").trim();
    if (!text) { setStatus("Add some prompt text first.", "busy"); return; }
    const sel = $("eventSelect");
    const list = loadCustomPrompts();
    let id = sel.value;
    if (id === NEW_PROMPT_VALUE || !list.some((c) => c.id === id)) {
      // Create a new saved prompt.
      id = "custom_" + Date.now().toString(36);
      list.push({ id, name, text });
    } else {
      // Update the existing saved prompt in place.
      const existing = list.find((c) => c.id === id);
      if (existing) { existing.name = name; existing.text = text; }
    }
    saveCustomPrompts(list);
    renderEventSelect();
    $("eventSelect").value = id;
    onContextChanged();
    setStatus("Saved! Maxwell will use this prompt.", "live");
  });

  $("customPromptDelete").addEventListener("click", () => {
    const sel = $("eventSelect");
    const id = sel.value;
    const list = loadCustomPrompts().filter((c) => c.id !== id);
    saveCustomPrompts(list);
    renderEventSelect();
    $("eventSelect").value = "admit";
    onContextChanged();
  });
}

wireEventControls();

// ---- realtime session ----

let micMode = "ptt";   // "auto" or "ptt" — PTT is the guest default
                       // because the fair floor is loud and Maxwell
                       // kept hearing his own voice / bystanders on
                       // auto-VAD.
let rt = null;
let serialReady = false;
let ptt = false;
let pttDownPromise = null;

function rtIsRunning() { return !!(rt && rt.isRunning && rt.isRunning()); }

function refreshTalkButton() {
  const btn = $("talkBtn");
  const hint = $("talkHint");
  const stop = $("stopBtn");
  const running = rtIsRunning();

  if (!serialReady) {
    btn.disabled = true;
    btn.textContent = "Offline";
    hint.textContent = "";
    stop.classList.remove("visible");
    return;
  }

  if (micMode === "ptt") {
    btn.textContent = running ? "Hold to speak" : "Start";
    hint.textContent = running
      ? "Hold the button (or spacebar) while you talk."
      : "Tap once to wake Maxwell up.";
  } else {
    btn.textContent = running ? "Listening…" : "Wake Maxwell";
    hint.textContent = running
      ? "Talk anytime — he'll hear you and reply live."
      : "Tap to start a live conversation.";
  }
  btn.disabled = false;
  stop.classList.toggle("visible", running);
}

async function startRealtime() {
  if (rtIsRunning()) return true;
  setStatus("Connecting to Maxwell…", "busy");
  rt = new RealtimeSession({
    envelope,
    behavior,
    speakingContext: speakingCtx,
    log: silentLog,
    onState: (s) => {
      if (s === "error") setStatus("Couldn't connect. Try again?", "error");
      else if (s === "disconnected") setStatus("Disconnected");
    },
    onTranscript: ({ role, text }) => {
      if (text) bubble(role === "user" ? "you" : "bird", text);
    },
  });
  try {
    await rt.start({
      voice: "ballad",
      pttMode: micMode === "ptt",
      micDeviceId: ($("micDeviceSelect") && $("micDeviceSelect").value) || "",
      outputDeviceId: ($("outputDeviceSelect") && $("outputDeviceSelect").value) || "",
      context: currentContextText(),
    });
    // After the mic is held once, device labels become non-empty —
    // re-render so the guest sees real device names.
    refreshAudioDeviceLists();
    behavior.setState("listening");
    setStatus(
      micMode === "ptt" ? "Hold the button to talk" : "Maxwell is listening",
      "live",
    );
    refreshTalkButton();
    return true;
  } catch (e) {
    setStatus("Couldn't start: " + (e.message || e), "error");
    rt = null;
    refreshTalkButton();
    return false;
  }
}

async function stopRealtime() {
  if (!rt) return;
  try { await rt.stop(); } catch (_) {}
  rt = null;
  behavior.setState("idle");
  setStatus("Ready");
  refreshTalkButton();
}

// ---- serial connect (silent where possible) ----

function setSerialReady(ok) {
  serialReady = ok;
  const connectBtn = $("connectBtn");
  if (ok) {
    connectBtn.classList.remove("visible");
  } else {
    connectBtn.classList.add("visible");
  }
  refreshTalkButton();
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
      // Silent path: only succeed if a port is already authorized.
      const ok = await transport.tryAutoConnect();
      if (!ok) {
        setStatus(
          "Tap \u201CConnect Maxwell\u201D once to hook up the USB cable.",
        );
        setSerialReady(false);
        return;
      }
    }
    scheduler.setTransport(transport);
    setStatus("Ready");
    setSerialReady(true);
  } catch (e) {
    setStatus("Couldn't connect to Maxwell: " + (e.message || e), "error");
    setSerialReady(false);
  }
}

$("connectBtn").addEventListener("click", () =>
  connectHardware({ requireUserGesture: true })
);

// Try the zero-friction path first; if it fails we show the explicit
// Connect Maxwell button (inside the setSerialReady(false) path).
connectHardware({ requireUserGesture: false });

// ---- mic mode pills ----

$("micGroup").addEventListener("click", async (e) => {
  const t = e.target;
  if (!t.dataset.mic || t.classList.contains("active")) return;
  micMode = t.dataset.mic;
  setActive($("micGroup"), "mic", micMode);
  // If realtime is already running, restart so VAD/PTT mode flips.
  if (rtIsRunning()) {
    await stopRealtime();
    await startRealtime();
  } else {
    refreshTalkButton();
  }
});

// ---- talk button ----

async function pttDown() {
  if (micMode !== "ptt" || ptt || !serialReady) return;
  ptt = true;
  $("talkBtn").classList.add("held");
  setStatus("Listening…", "live");
  pttDownPromise = (async () => {
    if (!rtIsRunning()) await startRealtime();
    if (!rtIsRunning()) {
      ptt = false;
      $("talkBtn").classList.remove("held");
      return;
    }
    try { rt.pttDown(); } catch (_) {}
  })();
  try { await pttDownPromise; } finally { pttDownPromise = null; }
}

async function pttUp() {
  if (!ptt) return;
  ptt = false;
  $("talkBtn").classList.remove("held");
  setStatus("Maxwell is thinking…", "busy");
  if (pttDownPromise) {
    try { await pttDownPromise; } catch (_) {}
  }
  try { rt && rt.pttUp(); } catch (_) {}
}

$("talkBtn").addEventListener("mousedown", (e) => {
  if (micMode === "ptt") { e.preventDefault(); pttDown(); }
});
$("talkBtn").addEventListener("touchstart", (e) => {
  if (micMode === "ptt") { e.preventDefault(); pttDown(); }
}, { passive: false });
window.addEventListener("mouseup", () => { if (ptt) pttUp(); });
window.addEventListener("touchend", () => { if (ptt) pttUp(); });
window.addEventListener("touchcancel", () => { if (ptt) pttUp(); });

$("talkBtn").addEventListener("click", async () => {
  if (!serialReady) return;
  if (micMode === "auto") {
    if (!rtIsRunning()) await startRealtime();
  } else if (!rtIsRunning()) {
    // PTT mode: let the guest tap once to arm the session. Next press
    // of the button will hold-to-talk.
    await startRealtime();
  }
});

// Spacebar = PTT when in push-to-talk mode.
let spaceHeld = false;
window.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat || micMode !== "ptt") return;
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  e.preventDefault();
  if (!spaceHeld) { spaceHeld = true; pttDown(); }
});
window.addEventListener("keyup", (e) => {
  if (e.code !== "Space") return;
  if (spaceHeld) { spaceHeld = false; pttUp(); }
});

// ---- stop pill ----

$("stopBtn").addEventListener("click", async () => {
  if (ptt) await pttUp();
  await stopRealtime();
});

// ---- cleanup on unload ----

window.addEventListener("beforeunload", async () => {
  try { if (rt) await rt.stop(); } catch (_) {}
  try { if (transport) await transport.disconnect(); } catch (_) {}
});

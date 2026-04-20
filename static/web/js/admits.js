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
(async () => {
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

// ---- realtime session ----

let micMode = "auto";   // "auto" or "ptt"
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
    await rt.start({ voice: "ballad", pttMode: micMode === "ptt" });
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

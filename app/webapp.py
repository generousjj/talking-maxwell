"""Minimal web UI for the Maxwell parrot bot.

One aiohttp server that keeps a single ConversationPipeline alive across
requests (so the 2-3s handshake only happens once), plus a single-page
front-end with:

  * text box + Run button to make Maxwell speak arbitrary text
  * Replay button for a WAV file
  * live jaw tuning (invert, gain, max_pwm) that takes effect on the next
    utterance without restart
  * a log stream so you can see what's happening

Run it with:

    python -m app.webapp --backend bottango_serial --config config.example.yaml

then open http://localhost:8787 in a browser.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cli import _build_backend, _build_providers, build_arg_parser  # noqa: E402
from app.config import AppConfig, load_config  # noqa: E402
from app.pipeline import ConversationPipeline  # noqa: E402
from motion.state_machine import ConversationStateMachine  # noqa: E402

log = logging.getLogger("maxwell.webapp")


class _LogBuffer(logging.Handler):
    """In-memory ring buffer of recent log lines for the UI."""

    def __init__(self, capacity: int = 200) -> None:
        super().__init__()
        self.records: Deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:  # noqa: BLE001
            msg = record.getMessage()
        self.records.append(msg)

    def snapshot(self) -> list[str]:
        return list(self.records)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Maxwell Parrot Bot</title>
<style>
  :root { color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
  body { max-width: 760px; margin: 2rem auto; padding: 0 1rem; line-height: 1.4; }
  h1 { margin: 0 0 .25rem; }
  .sub { opacity: .65; margin-bottom: 1.5rem; }
  textarea, input[type=text] { width: 100%; box-sizing: border-box; font-size: 1rem;
    font-family: inherit; padding: .5rem; border-radius: 6px; border: 1px solid #8883; }
  textarea { height: 6rem; }
  button { padding: .55rem 1rem; font-size: 1rem; border-radius: 6px; border: 1px solid #8886;
    background: #f5f5f5; cursor: pointer; }
  button.primary { background: #ff8f3c; color: white; border-color: #d9722a; font-weight: 600; }
  button.stop   { background: #dc4d4d; color: white; border-color: #a43333; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .row { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0 1rem; }
  .row > * { flex: 0 0 auto; }
  .grid { display: grid; grid-template-columns: 160px 1fr 60px; gap: .4rem .75rem;
    align-items: center; margin: .5rem 0; }
  label { font-size: .9rem; }
  pre { background: #1116; color: #ddd; padding: .75rem; border-radius: 6px;
    height: 14rem; overflow: auto; white-space: pre-wrap; font-size: 12px; }
  .status { font-size: .95rem; }
  .status.busy { color: #d97706; }
  .status.ok { color: #16a34a; }
  .status.err { color: #dc2626; }
  section { margin: 1.5rem 0; }
  h2 { font-size: 1rem; margin: 0 0 .5rem; opacity: .85; }
</style>
</head>
<body>
<h1>Maxwell</h1>
<div class="sub">Laptop parrot bot • <span id="backend"></span></div>

<section>
  <h2>Speak</h2>
  <textarea id="text" placeholder="Type what Maxwell should say...">Hello friend. I'm Maxwell. It's lovely to meet you today!</textarea>
  <div class="row">
    <button id="speak" class="primary">Speak</button>
    <button id="replay">Replay test clip</button>
    <button id="center">Center servos</button>
    <button id="stop" class="stop">Stop</button>
  </div>
  <div id="status" class="status">ready</div>
</section>

<section>
  <h2>Conversation</h2>
  <div class="row">
    <button id="talk" class="primary">Talk to Maxwell</button>
    <label style="display:flex;align-items:center;gap:.35rem;font-size:.9rem;">
      <input type="checkbox" id="autoLoop"/> Keep conversing
    </label>
  </div>
  <div class="sub" style="font-size:.85rem;margin-top:-.5rem;">
    Click Talk, speak into the mic, then pause ~1 second. Maxwell listens,
    transcribes, replies, and speaks back. "Keep conversing" auto-restarts
    listening after each reply.
  </div>
  <div id="convStatus" class="status" style="margin:.5rem 0;">idle</div>
  <div id="convLog" style="display:flex;flex-direction:column;gap:.35rem;
    max-height:18rem;overflow:auto;padding:.25rem 0;"></div>
</section>

<section>
  <h2>Voice &amp; personality</h2>
  <div class="grid" style="grid-template-columns: 160px 1fr;">
    <label for="voice">Voice</label>
    <select id="voice" style="padding:.45rem;font-size:1rem;border-radius:6px;border:1px solid #8883;">
      <option value="verse">verse — expressive, lively (parrot-ish)</option>
      <option value="ash">ash — characterful, raspy</option>
      <option value="fable">fable — British storyteller</option>
      <option value="sage">sage — bright, chipper</option>
      <option value="shimmer">shimmer — airy, playful</option>
      <option value="coral">coral — warm, friendly</option>
      <option value="nova">nova — bright, peppy</option>
      <option value="ballad">ballad — melodic</option>
      <option value="alloy">alloy — neutral</option>
      <option value="echo">echo — calm</option>
      <option value="onyx">onyx — deep (less parrot-y)</option>
    </select>

    <label for="instructions">Voice style</label>
    <textarea id="instructions" style="height:4rem;" placeholder="Style hints for the TTS voice (tone, accent, cadence)..."></textarea>

    <label for="personality">Personality</label>
    <textarea id="personality" style="height:4rem;" placeholder="System prompt that shapes Maxwell's replies..."></textarea>
  </div>
  <button id="applyVoice">Apply voice &amp; personality</button>
  <div class="sub" style="font-size:.85rem;margin-top:.3rem;">Takes effect on the next utterance. No reconnect needed.</div>
</section>

<section>
  <h2>Motion tuning</h2>
  <div class="grid">
    <label for="intensity">Overall intensity</label>
    <input type="range" id="intensity" min="0" max="100" step="5"/>
    <span id="intensityVal"></span>

    <label for="invert">Invert jaw</label>
    <input type="checkbox" id="invert" style="justify-self:start;"/>
    <span></span>

    <label for="gain">Jaw gain</label>
    <input type="range" id="gain" min="1.0" max="6.0" step="0.1"/>
    <span id="gainVal"></span>

    <label for="maxpwm">Jaw max_pwm (μs)</label>
    <input type="range" id="maxpwm" min="1500" max="2200" step="25"/>
    <span id="maxpwmVal"></span>

    <label for="minpwm">Jaw min_pwm (μs)</label>
    <input type="range" id="minpwm" min="900" max="1700" step="25"/>
    <span id="minpwmVal"></span>

    <label for="slew">Jaw slew (μs/s)</label>
    <input type="range" id="slew" min="400" max="3000" step="100"/>
    <span id="slewVal"></span>

    <label for="rate">Jaw update rate (Hz)</label>
    <input type="range" id="rate" min="4" max="30" step="1"/>
    <span id="rateVal"></span>

    <label for="delta">Jaw dead-band (%)</label>
    <input type="range" id="delta" min="0" max="8" step="1"/>
    <span id="deltaVal"></span>
  </div>
  <button id="applyTuning">Apply PWM / slew changes (reconnect)</button>
  <div class="sub" style="font-size:.85rem;margin-top:.3rem;">
    Intensity, invert, gain, rate &amp; dead-band apply live. PWM ranges and
    slew rate require a reconnect. <b>If your jaw has extension wires and
    stays still during speech</b>, pull the slew rate down (1000-1500μs/s)
    and the update rate down (8-12 Hz) — that reduces signal noise over
    the long wire and lets the servo actually track its target.
  </div>
</section>

<section>
  <h2>Diagnose &amp; recover</h2>
  <div style="display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;">
    <button id="fullReset" class="primary">Full reset &amp; wake (jaw stuck?)</button>
    <button id="wake">Wake servos (sweep all)</button>
    <button id="testJaw">Wiggle jaw (slow, extremes)</button>
    <button id="reregister">Re-register servos</button>
  </div>
  <div class="sub" style="font-size:.85rem;margin-top:.3rem;">
    <b>If the jaw stops moving mid-session</b>, click <b>Full reset</b>.
    That's our best software equivalent of unplugging/replugging GPIO 9:
    we deregister all servos, re-register them, then hammer the jaw pin
    with ~10 fast extreme swings to try to bridge a marginal dupont
    contact. If it still doesn't move after that, the wire really is
    disconnected and needs a physical reseat.
  </div>
  <div style="margin-top:.75rem;">
    <label style="display:flex;align-items:center;gap:.4rem;font-size:.9rem;">
      <input type="checkbox" id="prewarm"/> Pre-warm jaw before each utterance
    </label>
    <div class="sub" style="font-size:.8rem;">Runs a quick ~400ms jaw wiggle before every speech to keep pin 9 awake. Adds a fraction of a second of latency but often prevents the "jaw went silent" problem.</div>
  </div>
  <div class="sub" style="font-size:.85rem;margin-top:.3rem;">Wiggle commands the jaw fully open then fully closed, 3 cycles, 1.5s hold each. If the jaw doesn't move here, the issue is servo / wiring / pin — not software.</div>
  <hr style="margin:1rem 0;border:none;border-top:1px solid #eee;"/>
  <div style="display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;">
    <label for="jawPin" style="font-weight:600;">Jaw pin:</label>
    <input type="number" id="jawPin" min="0" max="39" value="9" style="width:5rem;"/>
    <button id="pinSwap">Move jaw to this pin &amp; wiggle</button>
  </div>
  <div class="sub" style="font-size:.85rem;margin-top:.3rem;">To test whether GPIO 9 itself is the issue: physically move the mouth servo's cable into one of the working ports (say the wing port — pin 3), then enter <b>3</b> above and click. If the mouth moves, GPIO 9 on the ESP32 is bad. If it still doesn't move with working pins, the servo itself is dead or jammed.</div>
</section>

<section>
  <h2>Log</h2>
  <pre id="log"></pre>
</section>

<script>
const $ = (id) => document.getElementById(id);
async function api(path, body) {
  const r = await fetch(path, {
    method: body ? 'POST' : 'GET',
    headers: {'content-type': 'application/json'},
    body: body ? JSON.stringify(body) : undefined,
  });
  return r.json();
}

function setStatus(msg, cls) {
  const el = $('status');
  el.textContent = msg;
  el.className = 'status ' + (cls || '');
}

async function refreshLog() {
  try {
    const data = await api('/api/log');
    $('log').textContent = data.lines.join('\\n');
    $('log').scrollTop = $('log').scrollHeight;
  } catch (e) {}
}
setInterval(refreshLog, 1000);

async function loadConfig() {
  const cfg = await api('/api/config');
  $('backend').textContent = cfg.backend + (cfg.port ? ' (' + cfg.port + ')' : '');
  $('invert').checked = cfg.jaw.invert;
  $('gain').value = cfg.jaw.gain;
  $('gainVal').textContent = cfg.jaw.gain.toFixed(1);
  $('maxpwm').value = cfg.jaw.max_pwm;
  $('maxpwmVal').textContent = cfg.jaw.max_pwm + 'μs';
  $('minpwm').value = cfg.jaw.min_pwm;
  $('minpwmVal').textContent = cfg.jaw.min_pwm + 'μs';
  $('slew').value = cfg.jaw.max_pwm_per_sec;
  $('slewVal').textContent = cfg.jaw.max_pwm_per_sec + 'μs/s';
  const rateHz = Math.round(1.0 / (cfg.jaw_min_send_interval_s || 0.08));
  $('rate').value = rateHz;
  $('rateVal').textContent = rateHz + ' Hz';
  const deltaPct = Math.round((cfg.jaw_min_delta ?? 0.02) * 100);
  $('delta').value = deltaPct;
  $('deltaVal').textContent = deltaPct + '%';
  const pct = Math.round((cfg.intensity ?? 1.0) * 100);
  $('intensity').value = pct;
  $('intensityVal').textContent = pct + '%';
  if (cfg.voice) $('voice').value = cfg.voice;
  if (cfg.instructions != null) $('instructions').value = cfg.instructions;
  if (cfg.personality != null) $('personality').value = cfg.personality;
  $('prewarm').checked = !!cfg.prewarm_jaw;
}
loadConfig();

$('gain').oninput = () => $('gainVal').textContent = (+$('gain').value).toFixed(1);
$('maxpwm').oninput = () => $('maxpwmVal').textContent = $('maxpwm').value + 'μs';
$('minpwm').oninput = () => $('minpwmVal').textContent = $('minpwm').value + 'μs';
$('slew').oninput = () => $('slewVal').textContent = $('slew').value + 'μs/s';
$('rate').oninput = () => $('rateVal').textContent = $('rate').value + ' Hz';
$('delta').oninput = () => $('deltaVal').textContent = $('delta').value + '%';
$('rate').onchange = () => api('/api/tuning', {jaw_rate_hz: +$('rate').value});
$('delta').onchange = () => api('/api/tuning', {jaw_min_delta: +$('delta').value / 100});
$('intensity').oninput = () => $('intensityVal').textContent = $('intensity').value + '%';
$('intensity').onchange = () => api('/api/tuning', {intensity: +$('intensity').value / 100});

$('reregister').onclick = async () => {
  setStatus('re-registering servos...', 'busy');
  const r = await api('/api/reregister', {});
  setStatus(r.ok ? 're-registered' : ('error: ' + r.error), r.ok ? 'ok' : 'err');
};

$('testJaw').onclick = async () => {
  setStatus('wiggling jaw (~10s)...', 'busy');
  const r = await api('/api/test-jaw', {});
  setStatus(r.ok ? 'jaw wiggle done' : ('error: ' + r.error), r.ok ? 'ok' : 'err');
};

$('wake').onclick = async () => {
  setStatus('waking servos...', 'busy');
  const r = await api('/api/wake', {});
  setStatus(r.ok ? 'all servos awake' : ('error: ' + r.error), r.ok ? 'ok' : 'err');
};

$('fullReset').onclick = async () => {
  setStatus('full reset (jaw hammer + wake)...', 'busy');
  const r = await api('/api/full-reset', {});
  setStatus(r.ok ? 'full reset done — try speaking' : ('error: ' + r.error), r.ok ? 'ok' : 'err');
};

$('prewarm').onchange = () => api('/api/tuning', {prewarm_jaw: $('prewarm').checked});

$('pinSwap').onclick = async () => {
  const pin = +$('jawPin').value;
  setStatus('re-registering jaw on pin ' + pin + '...', 'busy');
  const sw = await api('/api/pin-swap-test', {pin});
  if (!sw.ok) {
    setStatus('pin swap error: ' + sw.error, 'err');
    return;
  }
  setStatus('wiggling jaw on pin ' + pin + '...', 'busy');
  const r = await api('/api/test-jaw', {});
  setStatus(r.ok ? ('wiggle on pin ' + pin + ' done') : ('error: ' + r.error), r.ok ? 'ok' : 'err');
};

$('speak').onclick = async () => {
  const text = $('text').value.trim();
  if (!text) return;
  setStatus('speaking...', 'busy');
  $('speak').disabled = true; $('replay').disabled = true;
  try {
    const r = await api('/api/speak', {text});
    setStatus(r.ok ? 'done' : ('error: ' + r.error), r.ok ? 'ok' : 'err');
  } catch (e) { setStatus('error: ' + e, 'err'); }
  $('speak').disabled = false; $('replay').disabled = false;
  refreshLog();
};

$('replay').onclick = async () => {
  setStatus('playing clip...', 'busy');
  $('speak').disabled = true; $('replay').disabled = true;
  try {
    const r = await api('/api/replay', {});
    setStatus(r.ok ? 'done' : ('error: ' + r.error), r.ok ? 'ok' : 'err');
  } catch (e) { setStatus('error: ' + e, 'err'); }
  $('speak').disabled = false; $('replay').disabled = false;
  refreshLog();
};

$('center').onclick = async () => {
  setStatus('centering servos...', 'busy');
  await api('/api/center', {});
  setStatus('centered', 'ok');
};

$('stop').onclick = async () => {
  await api('/api/stop', {});
  setStatus('stopped', 'err');
};

$('invert').onchange = () => api('/api/tuning', {invert: $('invert').checked});
$('gain').onchange = () => api('/api/tuning', {gain: +$('gain').value});
$('applyTuning').onclick = async () => {
  setStatus('reconnecting serial...', 'busy');
  const r = await api('/api/tuning', {
    invert: $('invert').checked,
    gain: +$('gain').value,
    max_pwm: +$('maxpwm').value,
    min_pwm: +$('minpwm').value,
    jaw_max_pwm_per_sec: +$('slew').value,
    jaw_rate_hz: +$('rate').value,
    jaw_min_delta: +$('delta').value / 100,
    reconnect: true,
  });
  setStatus(r.ok ? 'reconnected' : ('error: ' + r.error), r.ok ? 'ok' : 'err');
  refreshLog();
};

$('applyVoice').onclick = async () => {
  const r = await api('/api/tuning', {
    voice: $('voice').value,
    instructions: $('instructions').value,
    personality: $('personality').value,
  });
  setStatus(r.ok ? 'voice updated' : ('error: ' + r.error), r.ok ? 'ok' : 'err');
};

function setConvStatus(msg, cls) {
  const el = $('convStatus');
  el.textContent = msg;
  el.className = 'status ' + (cls || '');
}

function convLine(role, text) {
  const row = document.createElement('div');
  row.style.padding = '.4rem .6rem';
  row.style.borderRadius = '6px';
  row.style.border = '1px solid #8882';
  row.style.background = role === 'you' ? '#fff4e6' : '#eef6ff';
  const label = document.createElement('div');
  label.style.fontSize = '.75rem';
  label.style.opacity = '.7';
  label.style.marginBottom = '.15rem';
  label.textContent = role === 'you' ? 'you' : 'Maxwell';
  const body = document.createElement('div');
  body.textContent = text;
  row.appendChild(label);
  row.appendChild(body);
  $('convLog').appendChild(row);
  $('convLog').scrollTop = $('convLog').scrollHeight;
}

let conversing = false;
async function runOneTurn() {
  setConvStatus('listening...', 'busy');
  $('talk').disabled = true;
  try {
    const r = await api('/api/converse', {});
    if (!r.ok) {
      setConvStatus('error: ' + r.error, 'err');
      return false;
    }
    if (!r.user_text && !r.reply) {
      setConvStatus('heard nothing — try again', 'err');
      return false;
    }
    if (r.user_text) convLine('you', r.user_text);
    if (r.reply) convLine('maxwell', r.reply);
    setConvStatus('idle', 'ok');
    return true;
  } catch (e) {
    setConvStatus('error: ' + e, 'err');
    return false;
  } finally {
    $('talk').disabled = false;
  }
}

$('talk').onclick = async () => {
  if (conversing) {
    conversing = false;
    $('talk').textContent = 'Talk to Maxwell';
    setConvStatus('stopping after this turn...', 'busy');
    return;
  }
  const loop = $('autoLoop').checked;
  if (loop) {
    conversing = true;
    $('talk').textContent = 'Stop conversation';
    while (conversing) {
      const ok = await runOneTurn();
      if (!ok) break;
      await new Promise(r => setTimeout(r, 300));
    }
    conversing = false;
    $('talk').textContent = 'Talk to Maxwell';
    setConvStatus('idle', 'ok');
  } else {
    await runOneTurn();
  }
};
</script>
</body>
</html>
"""


class AppState:
    """Holds the long-lived pipeline so web requests reuse one serial link."""

    def __init__(self, config: AppConfig, args: argparse.Namespace) -> None:
        self.config = config
        self.args = args
        self.pipeline: Optional[ConversationPipeline] = None
        self.lock = asyncio.Lock()
        self.backend = None
        # If True, do a ~400ms jaw wiggle before every spoken utterance.
        # Acts as a pre-emptive reseat for the flaky GPIO 9 connection.
        self.prewarm_jaw = False

    async def maybe_prewarm_jaw(self) -> None:
        """Run a short jaw hammer before speaking if the user enabled it."""
        if not self.prewarm_jaw or self.backend is None:
            return
        if not hasattr(self.backend, "jaw_hammer"):
            return
        try:
            await self.backend.jaw_hammer(cycles=3, period_s=0.07)
        except Exception:  # noqa: BLE001
            log.exception("prewarm jaw failed; continuing")

    async def start(self) -> None:
        await self._create_pipeline()

    async def _create_pipeline(self) -> None:
        stt, llm, tts = _build_providers(self.config, self.args)
        state_machine = ConversationStateMachine()
        self.backend = _build_backend(self.config, self.args)
        pipeline = ConversationPipeline(
            stt=stt, llm=llm, tts=tts,
            backend=self.backend,
            state_machine=state_machine,
            jaw_calibration=self.config.motion.jaw,
            behavior_gains=self.config.motion.behavior,
            rate_hz=self.config.motion.rate_hz,
            personality=self.config.personality,
            audio_frame_ms=self.config.audio.frame_ms,
            audio_input_sample_rate=self.config.audio.input_sample_rate,
            playback_device=self.args.playback_device or self.config.audio.playback_device,
            mic_max_s=self.config.audio.max_utterance_s,
            mic_silence_threshold=self.config.audio.vad_threshold,
            mic_silence_hangover_s=self.config.audio.vad_silence_hangover_s,
        )
        await pipeline.__aenter__()
        self.pipeline = pipeline
        log.info("web app: pipeline ready; serial backend connected")
        # Wake sweep: exercises every servo once so a flaky GPIO contact
        # (usually the jaw) seats itself before the first utterance. Also
        # surfaces any dead channel before audio starts. Safe to skip if
        # the backend doesn't implement it.
        if hasattr(self.backend, "wake_sweep"):
            try:
                await self.backend.wake_sweep()
            except Exception:  # noqa: BLE001
                log.exception("wake sweep failed; continuing")

    async def recreate(self) -> None:
        async with self.lock:
            if self.pipeline is not None:
                try:
                    await self.pipeline.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    log.exception("error shutting down old pipeline")
                self.pipeline = None
            await self._create_pipeline()

    async def stop(self) -> None:
        """Soft stop: send servos to a safe center pose.

        Previously this tore down the entire pipeline, which left the web
        UI permanently in an "error: pipeline not ready" state until a
        server restart. The pipeline stays alive now; the Stop button is
        just a panic/center control.
        """
        from motion.models import MotionFrame
        if self.backend is not None:
            try:
                await self.backend.send_frame(
                    MotionFrame(jaw_open=0.0, head_lr=0.5, head_ud=0.5, wing=0.0)
                )
            except Exception:  # noqa: BLE001
                log.exception("stop: failed to center servos")

    async def ensure_pipeline(self) -> None:
        """Rebuild the pipeline if it went missing (defensive guard)."""
        if self.pipeline is None:
            log.warning("pipeline missing; rebuilding")
            await self._create_pipeline()

    async def shutdown(self) -> None:
        """Full teardown, only used at process exit."""
        async with self.lock:
            if self.pipeline is not None:
                try:
                    await self.pipeline.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    log.exception("error shutting down pipeline")
                self.pipeline = None


def _attach_log_buffer() -> _LogBuffer:
    buf = _LogBuffer(capacity=400)
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s", "%H:%M:%S")
    buf.setFormatter(fmt)
    logging.getLogger().addHandler(buf)
    return buf


def build_app(state: AppState, log_buffer: _LogBuffer) -> web.Application:
    app = web.Application()

    async def index(_req: web.Request) -> web.Response:
        return web.Response(text=INDEX_HTML, content_type="text/html")

    async def api_config(_req: web.Request) -> web.Response:
        cfg = state.config
        jaw = cfg.bottango.serial.jaw
        intensity = 1.0
        if state.backend is not None:
            intensity = float(getattr(state.backend, "motion_intensity", 1.0))
        voice = cfg.providers.tts_voice
        instructions = cfg.providers.tts_instructions
        personality = cfg.personality
        if state.pipeline is not None:
            voice = getattr(state.pipeline.tts, "voice", voice) or voice
            instructions = getattr(state.pipeline.tts, "instructions", instructions) or instructions
            personality = state.pipeline.personality or personality
        jaw_min_delta = 0.020
        jaw_min_send_interval_s = 0.08
        if state.backend is not None:
            jaw_min_delta = float(getattr(state.backend, "jaw_min_delta", 0.020))
            jaw_min_send_interval_s = float(
                getattr(state.backend, "jaw_min_send_interval_s", 0.08)
            )
        return web.json_response(
            {
                "backend": state.args.backend or cfg.backend,
                "port": cfg.bottango.serial.port or "auto",
                "intensity": intensity,
                "jaw": {
                    "invert": bool(jaw.invert),
                    "gain": cfg.motion.jaw.gain,
                    "min_pwm": int(jaw.min_pwm),
                    "max_pwm": int(jaw.max_pwm),
                    "max_pwm_per_sec": int(jaw.max_pwm_per_sec),
                },
                "jaw_min_delta": jaw_min_delta,
                "jaw_min_send_interval_s": jaw_min_send_interval_s,
                "voice": voice,
                "instructions": instructions,
                "personality": personality,
                "prewarm_jaw": bool(state.prewarm_jaw),
            }
        )

    async def api_log(_req: web.Request) -> web.Response:
        return web.json_response({"lines": log_buffer.snapshot()[-200:]})

    async def api_speak(req: web.Request) -> web.Response:
        body = await req.json()
        text = (body.get("text") or "").strip()
        if not text:
            return web.json_response({"ok": False, "error": "no text"})
        try:
            async with state.lock:
                await state.ensure_pipeline()
                await state.maybe_prewarm_jaw()
                await state.pipeline.say(text)
            return web.json_response({"ok": True})
        except Exception as e:  # noqa: BLE001
            log.exception("speak failed")
            return web.json_response({"ok": False, "error": str(e)})

    async def api_replay(_req: web.Request) -> web.Response:
        wav = "/tmp/maxwell_short.wav"
        if not Path(wav).exists():
            return web.json_response({"ok": False, "error": f"{wav} not found"})
        try:
            async with state.lock:
                await state.ensure_pipeline()
                await state.maybe_prewarm_jaw()
                await state.pipeline.speak_wav_file(wav)
            return web.json_response({"ok": True})
        except Exception as e:  # noqa: BLE001
            log.exception("replay failed")
            return web.json_response({"ok": False, "error": str(e)})

    async def api_center(_req: web.Request) -> web.Response:
        from motion.models import MotionFrame
        if state.backend is None:
            return web.json_response({"ok": False, "error": "no backend"})
        try:
            await state.backend.send_frame(
                MotionFrame(jaw_open=0.0, head_lr=0.5, head_ud=0.5, wing=0.0)
            )
            return web.json_response({"ok": True})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(e)})

    async def api_stop(_req: web.Request) -> web.Response:
        await state.stop()
        return web.json_response({"ok": True})

    async def api_tuning(req: web.Request) -> web.Response:
        body = await req.json()
        cfg = state.config
        changed_serial = False
        if "invert" in body:
            cfg.bottango.serial.jaw.invert = bool(body["invert"])
            if state.backend is not None:
                state.backend._channel_invert["jaw"] = bool(body["invert"])
        if "gain" in body:
            cfg.motion.jaw.gain = float(body["gain"])
        if "intensity" in body and state.backend is not None:
            state.backend.motion_intensity = float(body["intensity"])
            log.info("motion intensity set to %.0f%%", float(body["intensity"]) * 100)
        if "max_pwm" in body:
            cfg.bottango.serial.jaw.max_pwm = int(body["max_pwm"])
            changed_serial = True
        if "min_pwm" in body:
            cfg.bottango.serial.jaw.min_pwm = int(body["min_pwm"])
            changed_serial = True
        if "jaw_max_pwm_per_sec" in body:
            cfg.bottango.serial.jaw.max_pwm_per_sec = int(body["jaw_max_pwm_per_sec"])
            changed_serial = True
        if "jaw_rate_hz" in body:
            hz = max(1.0, float(body["jaw_rate_hz"]))
            interval = 1.0 / hz
            cfg.bottango.serial.jaw_min_send_interval_s = interval
            if state.backend is not None:
                state.backend.jaw_min_send_interval_s = interval
            log.info("jaw update rate set to %.1f Hz", hz)
        if "jaw_min_delta" in body:
            d = max(0.0, float(body["jaw_min_delta"]))
            cfg.bottango.serial.jaw_min_delta = d
            if state.backend is not None:
                state.backend.jaw_min_delta = d
            log.info("jaw min_delta set to %.3f (%.1f%%)", d, d * 100.0)
        if "voice" in body and state.pipeline is not None:
            v = str(body["voice"]).strip()
            if v and hasattr(state.pipeline.tts, "voice"):
                state.pipeline.tts.voice = v
                cfg.providers.tts_voice = v
                log.info("tts voice set to %s", v)
        if "instructions" in body and state.pipeline is not None:
            inst = str(body["instructions"])
            if hasattr(state.pipeline.tts, "instructions"):
                state.pipeline.tts.instructions = inst
                cfg.providers.tts_instructions = inst
                log.info("tts instructions updated (%d chars)", len(inst))
        if "personality" in body and state.pipeline is not None:
            p = str(body["personality"])
            state.pipeline.personality = p
            cfg.personality = p
            log.info("personality updated (%d chars)", len(p))
        if "prewarm_jaw" in body:
            state.prewarm_jaw = bool(body["prewarm_jaw"])
            log.info("pre-warm jaw set to %s", state.prewarm_jaw)
        if body.get("reconnect") and changed_serial:
            try:
                await state.recreate()
            except Exception as e:  # noqa: BLE001
                log.exception("recreate failed")
                return web.json_response({"ok": False, "error": str(e)})
        return web.json_response({"ok": True})

    async def api_wake(_req: web.Request) -> web.Response:
        """Exercise every servo through a min/max/mid sweep.

        Handy when the jaw (or any channel) goes quiet from a loose GPIO
        contact — the physical motion reseats the dupont pin in software,
        no need for the user to unplug/replug.
        """
        if state.backend is None or not hasattr(state.backend, "wake_sweep"):
            return web.json_response({"ok": False, "error": "no backend"})
        try:
            async with state.lock:
                await state.backend.wake_sweep()
            return web.json_response({"ok": True})
        except Exception as e:  # noqa: BLE001
            log.exception("wake failed")
            return web.json_response({"ok": False, "error": str(e)})

    async def api_converse(_req: web.Request) -> web.Response:
        """Drive one listen → transcribe → reply → speak turn.

        Uses the pipeline's mic-driven flow: VAD-based recording on the
        laptop mic, OpenAI Whisper for transcription, the configured LLM
        for the reply, then TTS + motion playback. Returns both the
        transcribed user utterance and the spoken reply so the UI can
        render a transcript.
        """
        try:
            async with state.lock:
                await state.ensure_pipeline()
                await state.maybe_prewarm_jaw()
                user_text, reply = await state.pipeline.handle_live_turn()
            return web.json_response(
                {"ok": True, "user_text": user_text, "reply": reply}
            )
        except Exception as e:  # noqa: BLE001
            log.exception("converse failed")
            return web.json_response({"ok": False, "error": str(e)})

    async def api_full_reset(_req: web.Request) -> web.Response:
        """Software equivalent of physically unplugging / replugging pin 9.

        Deregisters all effectors, re-registers them, then hammers the
        jaw pin with ~10 fast extreme swings to try to bridge a marginal
        dupont contact. Also runs the full servo wake-sweep so every
        channel gets exercised. Meant to be a one-click recovery when
        the jaw has stopped responding mid-session.
        """
        if state.backend is None or not hasattr(state.backend, "full_reset"):
            return web.json_response({"ok": False, "error": "no backend"})
        try:
            async with state.lock:
                await state.backend.full_reset()
            return web.json_response({"ok": True})
        except Exception as e:  # noqa: BLE001
            log.exception("full reset failed")
            return web.json_response({"ok": False, "error": str(e)})

    async def api_reregister(_req: web.Request) -> web.Response:
        if state.backend is None or not hasattr(state.backend, "_needs_reregister"):
            return web.json_response({"ok": False, "error": "no serial backend"})
        evt = state.backend._needs_reregister
        if evt is None:
            return web.json_response({"ok": False, "error": "backend not connected"})
        evt.set()
        return web.json_response({"ok": True})

    async def api_test_jaw(_req: web.Request) -> web.Response:
        """Drive the jaw servo directly with extreme-and-hold steps.

        Sends three full open / three full close cycles with 1.5-second holds
        so any slew-rate smoothing in the firmware has time to settle, and
        the servo has ample time to complete each move. If the jaw does not
        visibly move at all here, the issue is between the ESP32 pin and the
        physical servo (wiring / pin / servo hardware), not our software.
        """
        from motion.models import MotionFrame
        if state.backend is None:
            return web.json_response({"ok": False, "error": "no backend"})
        try:
            log.info("jaw wiggle test: slow extreme endpoints, 3 cycles")
            sequence = [1.0, 0.0] * 3
            for i, target in enumerate(sequence):
                label = "OPEN" if target == 1.0 else "CLOSE"
                log.info("jaw wiggle step %d/%d: %s (jaw_open=%.2f)",
                         i + 1, len(sequence), label, target)
                await state.backend.send_frame(
                    MotionFrame(
                        jaw_open=target,
                        head_lr=0.5,
                        head_ud=0.5,
                        wing=0.0,
                    )
                )
                await asyncio.sleep(1.5)
            await state.backend.send_frame(
                MotionFrame(jaw_open=0.0, head_lr=0.5, head_ud=0.5, wing=0.0)
            )
            log.info("jaw wiggle test: complete")
            return web.json_response({"ok": True})
        except Exception as e:  # noqa: BLE001
            log.exception("jaw wiggle failed")
            return web.json_response({"ok": False, "error": str(e)})

    async def api_pin_swap_test(req: web.Request) -> web.Response:
        """Temporarily re-register the jaw servo on a different pin.

        Use this to isolate whether GPIO 9 is the problem. The user plugs
        the mouth servo cable into the pin listed below (e.g. swap with the
        wing servo's cable), posts the new pin number, and we re-register
        the jaw effector there. After the test you can switch it back.
        """
        body = await req.json()
        new_pin = int(body.get("pin", 4))
        if state.backend is None or not hasattr(state.backend, "jaw"):
            return web.json_response({"ok": False, "error": "no serial backend"})
        try:
            state.config.bottango.serial.jaw.pin = new_pin
            state.backend.jaw.pin = new_pin
            await state.recreate()
            log.info("jaw re-registered on pin %d", new_pin)
            return web.json_response({"ok": True, "pin": new_pin})
        except Exception as e:  # noqa: BLE001
            log.exception("pin swap failed")
            return web.json_response({"ok": False, "error": str(e)})

    app.router.add_get("/", index)
    app.router.add_get("/api/config", api_config)
    app.router.add_get("/api/log", api_log)
    app.router.add_post("/api/speak", api_speak)
    app.router.add_post("/api/replay", api_replay)
    app.router.add_post("/api/center", api_center)
    app.router.add_post("/api/stop", api_stop)
    app.router.add_post("/api/tuning", api_tuning)
    app.router.add_post("/api/reregister", api_reregister)
    app.router.add_post("/api/test-jaw", api_test_jaw)
    app.router.add_post("/api/pin-swap-test", api_pin_swap_test)
    app.router.add_post("/api/converse", api_converse)
    app.router.add_post("/api/wake", api_wake)
    app.router.add_post("/api/full-reset", api_full_reset)

    async def _on_cleanup(_app: web.Application) -> None:
        await state.shutdown()

    app.on_cleanup.append(_on_cleanup)
    return app


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    parser.add_argument("--host", default="127.0.0.1", help="Web UI host (default 127.0.0.1)")
    parser.add_argument("--port", dest="web_port", type=int, default=8787, help="Web UI port")
    args = parser.parse_args(argv)
    # Default to bottango_serial backend if not specified.
    if not args.backend:
        args.backend = "bottango_serial"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log_buffer = _attach_log_buffer()
    config = load_config(args.config, env_file=args.env_file)

    state = AppState(config, args)

    async def _run() -> None:
        await state.start()
        app = build_app(state, log_buffer)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.web_port)
        await site.start()
        log.info("Maxwell web UI on http://%s:%d", args.host, args.web_port)
        try:
            # Keep running until interrupted.
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

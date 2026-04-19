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
import os
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
<div class="sub">Laptop parrot bot • <span id="backend"></span> • <a href="/admits" style="color:#8b5cf6;text-decoration:none;font-weight:600;">Open admits view ↗</a></div>

<section>
  <h2>Connection</h2>
  <div class="row" style="align-items:center;">
    <button id="connect" class="primary">Connect</button>
    <button id="endSession" class="stop">End session</button>
    <span id="connStatus" class="status" style="padding:.25rem .6rem;">checking...</span>
  </div>
  <div class="sub" style="font-size:.85rem;margin-top:-.4rem;">
    "Connect" opens the serial port + builds the pipeline (auto-runs on
    boot). "End session" stops realtime mode if active, centers all
    servos, then closes the serial port so you can safely unplug
    Maxwell or hand him off.
  </div>
</section>

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
  <hr style="margin:.75rem 0;border:none;border-top:1px solid #eee;"/>
  <div class="row" style="align-items:center;">
    <button id="rtToggle">Start realtime mode</button>
    <span id="rtStatus" class="status" style="padding:.25rem .6rem;">off</span>
    <label style="display:flex;align-items:center;gap:.35rem;font-size:.9rem;margin-left:.5rem;">
      <input type="checkbox" id="rtPtt"/> Push-to-talk (hold space)
    </label>
    <button id="rtPttBtn" disabled style="padding:.35rem .8rem;font-size:.85rem;border-radius:6px;border:1px solid #8b5cf6;background:#f3f0ff;color:#5b3eb5;cursor:pointer;">
      Hold to talk
    </button>
    <label for="rtVoice" style="font-size:.85rem;margin-left:.5rem;">Realtime voice</label>
    <select id="rtVoice" style="padding:.4rem;font-size:.95rem;border-radius:6px;border:1px solid #8883;">
      <option value="ballad">ballad — melodic (default)</option>
      <option value="marin">marin — warm, conversational</option>
      <option value="cedar">cedar — deeper, grounded</option>
      <option value="alloy">alloy — neutral</option>
      <option value="ash">ash — characterful, raspy</option>
      <option value="coral">coral — friendly</option>
      <option value="echo">echo — calm</option>
      <option value="sage">sage — bright</option>
      <option value="shimmer">shimmer — airy</option>
      <option value="verse">verse — expressive, lively</option>
    </select>
  </div>
  <div class="sub" style="font-size:.85rem;margin-top:.25rem;">
    Realtime mode opens a continuous OpenAI speech-to-speech session.
    Just talk — Maxwell listens, replies, and can interrupt himself if
    you start talking again. Lower latency than the turn-based path.
    Voice changes apply on the next response (the session is restarted
    automatically if it's already running).
  </div>
  <details style="margin-top:.5rem;">
    <summary style="cursor:pointer;font-size:.85rem;color:#444;">
      Mic sensitivity (use these if Maxwell triggers on background noise)
    </summary>
    <div style="margin-top:.5rem;display:grid;grid-template-columns:160px 1fr 70px;gap:.4rem .6rem;align-items:center;">
      <label for="rtNoise" style="font-size:.85rem;">Noise reduction</label>
      <select id="rtNoise" style="padding:.35rem;font-size:.9rem;border-radius:6px;border:1px solid #8883;grid-column:2/4;">
        <option value="far_field">far_field — laptop mic in noisy room (recommended)</option>
        <option value="near_field">near_field — headset / AirPods</option>
        <option value="off">off — raw mic</option>
      </select>
      <label for="rtVadType" style="font-size:.85rem;">Turn detection</label>
      <select id="rtVadType" style="padding:.35rem;font-size:.9rem;border-radius:6px;border:1px solid #8883;grid-column:2/4;">
        <option value="server_vad">server_vad — silence based (predictable)</option>
        <option value="semantic_vad">semantic_vad — content aware (less interruption)</option>
      </select>
      <label for="rtThreshold" style="font-size:.85rem;">VAD threshold</label>
      <input id="rtThreshold" type="range" min="0" max="1" step="0.05" />
      <span id="rtThresholdVal" class="sub" style="text-align:right;">0.70</span>
      <label for="rtSilence" style="font-size:.85rem;">Silence to end turn (ms)</label>
      <input id="rtSilence" type="range" min="200" max="2000" step="50" />
      <span id="rtSilenceVal" class="sub" style="text-align:right;">700</span>
      <label for="rtPrefix" style="font-size:.85rem;">Prefix padding (ms)</label>
      <input id="rtPrefix" type="range" min="0" max="1000" step="50" />
      <span id="rtPrefixVal" class="sub" style="text-align:right;">300</span>
      <label for="rtEager" style="font-size:.85rem;">Semantic eagerness</label>
      <select id="rtEager" style="padding:.35rem;font-size:.9rem;border-radius:6px;border:1px solid #8883;grid-column:2/4;">
        <option value="low">low — let user take their time</option>
        <option value="medium">medium</option>
        <option value="high">high — chunk fast</option>
        <option value="auto">auto</option>
      </select>
      <label for="rtHalfDuplex" style="font-size:.85rem;">Echo guard (half-duplex)</label>
      <div style="grid-column:2/4;display:flex;align-items:center;gap:.6rem;">
        <input id="rtHalfDuplex" type="checkbox" />
        <span class="sub" style="font-size:.8rem;">Mute mic while Maxwell is speaking (prevents him from hearing himself)</span>
      </div>
      <label for="rtTail" style="font-size:.85rem;">Playback tail (ms)</label>
      <input id="rtTail" type="range" min="0" max="1500" step="50" />
      <span id="rtTailVal" class="sub" style="text-align:right;">400</span>
      <label for="rtBargeIn" style="font-size:.85rem;">Allow interruption</label>
      <div style="grid-column:2/4;display:flex;align-items:center;gap:.6rem;">
        <input id="rtBargeIn" type="checkbox" />
        <span class="sub" style="font-size:.8rem;">Talk loudly over Maxwell to cut him off mid-sentence</span>
      </div>
      <label for="rtBargeRms" style="font-size:.85rem;">Barge-in loudness</label>
      <input id="rtBargeRms" type="range" min="0.01" max="0.20" step="0.01" />
      <span id="rtBargeRmsVal" class="sub" style="text-align:right;">0.06</span>
      <label for="rtBargeFactor" style="font-size:.85rem;">Above-ambient factor</label>
      <input id="rtBargeFactor" type="range" min="1.5" max="10" step="0.5" />
      <span id="rtBargeFactorVal" class="sub" style="text-align:right;">5.0</span>
      <label for="rtBargeFrames" style="font-size:.85rem;">Min frames (40ms each)</label>
      <input id="rtBargeFrames" type="range" min="1" max="20" step="1" />
      <span id="rtBargeFramesVal" class="sub" style="text-align:right;">4</span>
    </div>
    <div class="sub" style="font-size:.8rem;margin-top:.4rem;">
      Higher threshold + longer silence = less likely to mistake background
      noise for speech. Far-field noise reduction is server-side and adds
      a tiny bit of latency. Semantic VAD ignores the threshold/silence
      sliders and uses content classification instead. Echo guard is the
      fix for "Maxwell keeps hearing himself" — leave it on unless you're
      using a headset / AirPods and want barge-in. Tail = how long after
      his last syllable we wait before re-opening the mic; bump it up if
      your speakers are loud or the room is reverberant. <b>Allow
      interruption</b> brings barge-in back even with echo guard on:
      while Maxwell speaks his mic is muted, but if you talk noticeably
      louder than the speaker leakage for ~150 ms the mic re-opens and
      he stops mid-sentence. Raise loudness threshold or above-ambient
      factor if false interrupts happen; lower min-frames if it feels
      sluggish. Changes apply immediately (session is restarted if
      active).
    </div>
  </details>
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
  if (cfg.realtime && cfg.realtime.voice) {
    $('rtVoice').value = cfg.realtime.voice;
  }
  if (cfg.realtime) {
    if (cfg.realtime.noise_reduction) $('rtNoise').value = cfg.realtime.noise_reduction;
    if (cfg.realtime.vad_type) $('rtVadType').value = cfg.realtime.vad_type;
    if (cfg.realtime.vad_threshold != null) {
      $('rtThreshold').value = cfg.realtime.vad_threshold;
      $('rtThresholdVal').textContent = (+cfg.realtime.vad_threshold).toFixed(2);
    }
    if (cfg.realtime.vad_silence_duration_ms != null) {
      $('rtSilence').value = cfg.realtime.vad_silence_duration_ms;
      $('rtSilenceVal').textContent = cfg.realtime.vad_silence_duration_ms;
    }
    if (cfg.realtime.vad_prefix_padding_ms != null) {
      $('rtPrefix').value = cfg.realtime.vad_prefix_padding_ms;
      $('rtPrefixVal').textContent = cfg.realtime.vad_prefix_padding_ms;
    }
    if (cfg.realtime.vad_eagerness) $('rtEager').value = cfg.realtime.vad_eagerness;
    if (cfg.realtime.half_duplex != null) $('rtHalfDuplex').checked = !!cfg.realtime.half_duplex;
    if (cfg.realtime.playback_tail_ms != null) {
      $('rtTail').value = cfg.realtime.playback_tail_ms;
      $('rtTailVal').textContent = cfg.realtime.playback_tail_ms;
    }
    if (cfg.realtime.barge_in_enabled != null) $('rtBargeIn').checked = !!cfg.realtime.barge_in_enabled;
    if (cfg.realtime.barge_in_rms_threshold != null) {
      $('rtBargeRms').value = cfg.realtime.barge_in_rms_threshold;
      $('rtBargeRmsVal').textContent = (+cfg.realtime.barge_in_rms_threshold).toFixed(2);
    }
    if (cfg.realtime.barge_in_above_ambient_factor != null) {
      $('rtBargeFactor').value = cfg.realtime.barge_in_above_ambient_factor;
      $('rtBargeFactorVal').textContent = (+cfg.realtime.barge_in_above_ambient_factor).toFixed(1);
    }
    if (cfg.realtime.barge_in_min_frames != null) {
      $('rtBargeFrames').value = cfg.realtime.barge_in_min_frames;
      $('rtBargeFramesVal').textContent = cfg.realtime.barge_in_min_frames;
    }
    if (cfg.realtime.push_to_talk != null) {
      $('rtPtt').checked = !!cfg.realtime.push_to_talk;
    }
  }
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

let rtRunning = false;
function setRtUi(running) {
  rtRunning = running;
  $('rtToggle').textContent = running ? 'Stop realtime mode' : 'Start realtime mode';
  $('rtStatus').textContent = running ? 'live' : 'off';
  $('rtStatus').className = 'status ' + (running ? 'ok' : '');
  // PTT button only meaningful while realtime is live AND PTT mode is on.
  const pttEnabled = $('rtPtt') && $('rtPtt').checked;
  if ($('rtPttBtn')) {
    $('rtPttBtn').disabled = !running || !pttEnabled;
    $('rtPttBtn').style.opacity = ($('rtPttBtn').disabled) ? '0.5' : '1';
  }
}
async function refreshRtStatus() {
  try {
    const r = await api('/api/realtime/status');
    setRtUi(!!r.running);
  } catch (e) {}
}
$('rtVoice').onchange = async () => {
  const r = await api('/api/realtime/config', {voice: $('rtVoice').value});
  if (r.ok && r.running) {
    $('rtStatus').textContent = 'restarting...';
  }
};
async function pushRtConfig(patch) {
  const r = await api('/api/realtime/config', patch);
  if (r.ok && r.running) {
    $('rtStatus').textContent = 'restarting...';
  }
}
$('rtNoise').onchange = () => pushRtConfig({noise_reduction: $('rtNoise').value});
$('rtVadType').onchange = () => pushRtConfig({vad_type: $('rtVadType').value});
$('rtEager').onchange = () => pushRtConfig({vad_eagerness: $('rtEager').value});
$('rtThreshold').oninput = () => $('rtThresholdVal').textContent = (+$('rtThreshold').value).toFixed(2);
$('rtThreshold').onchange = () => pushRtConfig({vad_threshold: +$('rtThreshold').value});
$('rtSilence').oninput = () => $('rtSilenceVal').textContent = $('rtSilence').value;
$('rtSilence').onchange = () => pushRtConfig({vad_silence_duration_ms: +$('rtSilence').value});
$('rtPrefix').oninput = () => $('rtPrefixVal').textContent = $('rtPrefix').value;
$('rtPrefix').onchange = () => pushRtConfig({vad_prefix_padding_ms: +$('rtPrefix').value});
$('rtHalfDuplex').onchange = () => pushRtConfig({half_duplex: $('rtHalfDuplex').checked});
$('rtTail').oninput = () => $('rtTailVal').textContent = $('rtTail').value;
$('rtTail').onchange = () => pushRtConfig({playback_tail_ms: +$('rtTail').value});
$('rtBargeIn').onchange = () => pushRtConfig({barge_in_enabled: $('rtBargeIn').checked});
$('rtBargeRms').oninput = () => $('rtBargeRmsVal').textContent = (+$('rtBargeRms').value).toFixed(2);
$('rtBargeRms').onchange = () => pushRtConfig({barge_in_rms_threshold: +$('rtBargeRms').value});
$('rtBargeFactor').oninput = () => $('rtBargeFactorVal').textContent = (+$('rtBargeFactor').value).toFixed(1);
$('rtBargeFactor').onchange = () => pushRtConfig({barge_in_above_ambient_factor: +$('rtBargeFactor').value});
$('rtBargeFrames').oninput = () => $('rtBargeFramesVal').textContent = $('rtBargeFrames').value;
$('rtBargeFrames').onchange = () => pushRtConfig({barge_in_min_frames: +$('rtBargeFrames').value});
$('rtToggle').onclick = async () => {
  $('rtToggle').disabled = true;
  try {
    if (!rtRunning) {
      $('rtStatus').textContent = 'connecting...';
      const r = await api('/api/realtime/start', {});
      if (!r.ok) {
        $('rtStatus').textContent = 'error: ' + r.error;
        $('rtStatus').className = 'status err';
      } else {
        setRtUi(true);
      }
    } else {
      const r = await api('/api/realtime/stop', {});
      setRtUi(!!r.running);
    }
  } finally {
    $('rtToggle').disabled = false;
  }
};
refreshRtStatus();
setInterval(refreshRtStatus, 3000);

// ---- Connection (top of page) ----
let connected = null;
function setConnUi(c) {
  connected = c;
  const el = $('connStatus');
  el.textContent = c ? 'connected' : 'disconnected';
  el.className = 'status ' + (c ? 'ok' : 'err');
  $('connect').disabled = !!c;
  $('endSession').disabled = !c;
}
async function refreshConnStatus() {
  try {
    const r = await api('/api/connection/status');
    if (r.ok) setConnUi(!!r.connected);
  } catch (e) {}
}
$('connect').onclick = async () => {
  $('connect').disabled = true;
  $('connStatus').textContent = 'connecting...';
  const r = await api('/api/connect', {});
  if (r.ok) setConnUi(true); else { $('connStatus').textContent = 'error: ' + r.error; }
};
$('endSession').onclick = async () => {
  $('endSession').disabled = true;
  $('connStatus').textContent = 'centering + closing...';
  const r = await api('/api/disconnect', {});
  if (r.ok) setConnUi(false); else { $('connStatus').textContent = 'error: ' + r.error; }
};
refreshConnStatus();
setInterval(refreshConnStatus, 5000);

// ---- Push-to-talk ----
function setPttUi(enabled) {
  $('rtPtt').checked = enabled;
  $('rtPttBtn').disabled = !enabled || !rtRunning;
  $('rtPttBtn').style.opacity = ($('rtPttBtn').disabled) ? '0.5' : '1';
}
$('rtPtt').onchange = async () => {
  const enabled = $('rtPtt').checked;
  await pushRtConfig({push_to_talk: enabled});
  setPttUi(enabled);
};
async function pttDown() {
  if (!$('rtPtt').checked || !rtRunning) return;
  $('rtPttBtn').textContent = 'Listening...';
  $('rtPttBtn').style.background = '#fde68a';
  await api('/api/realtime/ptt', {action: 'down'});
}
async function pttUp() {
  if (!$('rtPtt').checked || !rtRunning) return;
  $('rtPttBtn').textContent = 'Hold to talk';
  $('rtPttBtn').style.background = '#f3f0ff';
  await api('/api/realtime/ptt', {action: 'up'});
}
$('rtPttBtn').addEventListener('mousedown', pttDown);
$('rtPttBtn').addEventListener('mouseup', pttUp);
$('rtPttBtn').addEventListener('mouseleave', pttUp);
$('rtPttBtn').addEventListener('touchstart', e => { e.preventDefault(); pttDown(); });
$('rtPttBtn').addEventListener('touchend', e => { e.preventDefault(); pttUp(); });
let spaceHeld = false;
window.addEventListener('keydown', e => {
  if (e.code === 'Space' && !e.repeat && !spaceHeld) {
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) return;
    if (!$('rtPtt').checked || !rtRunning) return;
    e.preventDefault(); spaceHeld = true; pttDown();
  }
});
window.addEventListener('keyup', e => {
  if (e.code === 'Space' && spaceHeld) {
    spaceHeld = false; pttUp();
  }
});

let rtTranscriptCursor = 0;
async function pollRtTranscripts() {
  if (!rtRunning) return;
  try {
    const r = await api('/api/realtime/transcripts?since=' + rtTranscriptCursor);
    if (r.ok && r.items) {
      for (const item of r.items) {
        convLine(item.role === 'user' ? 'you' : 'maxwell', item.text);
      }
      if (r.last_id != null) rtTranscriptCursor = r.last_id;
    }
  } catch (e) {}
}
setInterval(pollRtTranscripts, 800);
</script>
</body>
</html>
"""


ADMITS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Talk to Maxwell</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=Fredoka:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    /* Maxwell's palette: blue body, purple wing, brown beak/legs */
    --maxwell-blue:   #3b8edb;
    --maxwell-blue-2: #2b6fb8;
    --maxwell-purple: #8b5cf6;
    --maxwell-purple-2: #6d3fd6;
    --maxwell-brown:  #8b5e34;
    --maxwell-brown-2: #5f3f1f;
    --cream:          #fff8ef;
    --bubble-you:     #ede9fe;
    --bubble-you-text:#4c1d95;
    --bubble-bird:    #dbeafe;
    --bubble-bird-text:#1e3a8a;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; min-height: 100vh;
    font-family: 'Baloo 2', 'Fredoka', system-ui, -apple-system, sans-serif;
    background: linear-gradient(160deg, #f0f4ff 0%, #fef3ec 100%);
    color: var(--maxwell-brown-2);
  }
  .frame {
    max-width: 720px; margin: 1.5rem auto 3rem; padding: 0 1rem;
  }
  header {
    display: flex; align-items: center; gap: 1rem;
    padding: 1rem 1.25rem;
    background: white;
    border-radius: 28px;
    box-shadow: 0 8px 24px rgba(59,142,219,.18);
    border: 3px solid var(--maxwell-blue);
  }
  .mascot {
    width: 72px; height: 72px;
    flex: 0 0 auto;
    display: flex; align-items: center; justify-content: center;
  }
  .mascot img {
    max-width: 100%; max-height: 100%;
    width: auto; height: auto;
    object-fit: contain;
    display: block;
  }
  header h1 {
    font-family: 'Baloo 2', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    margin: 0;
    color: var(--maxwell-blue-2);
    letter-spacing: -.5px;
  }
  header .tag {
    font-size: .95rem;
    color: var(--maxwell-brown);
    font-weight: 500;
    margin-top: 2px;
  }

  .card {
    background: white;
    border-radius: 28px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
    box-shadow: 0 6px 18px rgba(139,94,246,.10);
    border: 2px solid #ede9fe;
  }
  .card h2 {
    font-weight: 700;
    font-size: 1.1rem;
    margin: 0 0 .75rem;
    color: var(--maxwell-purple-2);
  }
  .row { display: flex; gap: .75rem; flex-wrap: wrap; align-items: center; }

  .pill-group {
    display: inline-flex;
    background: #f3f0ff;
    border-radius: 999px;
    padding: 4px;
    gap: 2px;
  }
  .pill-group button {
    border: none;
    background: transparent;
    padding: .55rem 1.1rem;
    font-family: inherit;
    font-size: 1rem;
    font-weight: 600;
    border-radius: 999px;
    cursor: pointer;
    color: var(--maxwell-purple-2);
    transition: background .15s;
  }
  .pill-group button.active {
    background: var(--maxwell-purple);
    color: white;
    box-shadow: 0 3px 10px rgba(139,94,246,.35);
  }
  .pill-group button:disabled { opacity: .35; cursor: not-allowed; }

  .talk-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: .5rem;
    margin: 1rem 0 .5rem;
  }
  #talkBtn {
    width: 200px; height: 200px;
    border-radius: 50%;
    border: 6px solid var(--maxwell-blue-2);
    background: radial-gradient(circle at 30% 30%,
      var(--maxwell-blue) 0%,
      var(--maxwell-blue-2) 70%,
      var(--maxwell-purple-2) 100%);
    color: white;
    font-family: 'Baloo 2', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    cursor: pointer;
    box-shadow:
      0 12px 30px rgba(43,111,184,.4),
      inset 0 -8px 0 rgba(0,0,0,.15);
    transition: transform .1s, box-shadow .1s;
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
  }
  #talkBtn:active, #talkBtn.held {
    transform: translateY(4px);
    box-shadow:
      0 4px 12px rgba(43,111,184,.4),
      inset 0 -2px 0 rgba(0,0,0,.15);
    background: radial-gradient(circle at 30% 30%,
      var(--maxwell-purple) 0%,
      var(--maxwell-purple-2) 70%,
      var(--maxwell-brown) 100%);
  }
  #talkBtn:disabled {
    opacity: .55;
    cursor: not-allowed;
    background: #cbd5e1;
    border-color: #94a3b8;
  }
  #talkHint {
    font-size: 1rem;
    color: var(--maxwell-brown);
    font-weight: 500;
    text-align: center;
    min-height: 1.4em;
  }
  #stopBtn {
    display: none;
    background: white;
    color: var(--maxwell-purple-2);
    border: 2px solid var(--maxwell-purple);
    border-radius: 999px;
    padding: .55rem 1.4rem;
    font-family: 'Baloo 2', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(94,66,153,.18);
    transition: transform .08s, background .15s, color .15s;
  }
  #stopBtn:hover {
    background: var(--maxwell-purple);
    color: white;
  }
  #stopBtn:active {
    transform: translateY(2px);
  }
  #stopBtn.visible { display: inline-block; }

  #status {
    text-align: center;
    font-weight: 600;
    font-size: 1rem;
    margin-top: .25rem;
    color: var(--maxwell-purple-2);
    min-height: 1.4em;
  }
  #status.live   { color: var(--maxwell-blue-2); }
  #status.busy   { color: #c2410c; }
  #status.error  { color: #b91c1c; }

  .chat {
    display: flex; flex-direction: column; gap: .65rem;
    max-height: 50vh; overflow-y: auto;
    padding: .5rem 0;
  }
  .bubble {
    padding: .75rem 1.1rem;
    border-radius: 22px;
    max-width: 85%;
    font-size: 1.05rem;
    line-height: 1.4;
    font-weight: 500;
    box-shadow: 0 2px 6px rgba(0,0,0,.05);
  }
  .bubble.you {
    align-self: flex-end;
    background: var(--bubble-you);
    color: var(--bubble-you-text);
    border-bottom-right-radius: 6px;
  }
  .bubble.bird {
    align-self: flex-start;
    background: var(--bubble-bird);
    color: var(--bubble-bird-text);
    border-bottom-left-radius: 6px;
    border-left: 4px solid var(--maxwell-blue);
  }
  .bubble .who {
    display: block;
    font-weight: 700;
    font-size: .8rem;
    opacity: .7;
    margin-bottom: .15rem;
    text-transform: uppercase;
    letter-spacing: .5px;
  }
  .empty {
    text-align: center;
    color: var(--maxwell-brown);
    opacity: .6;
    padding: 1rem;
    font-size: .95rem;
  }
  footer {
    text-align: center;
    margin-top: 1.25rem;
    font-size: .85rem;
    color: var(--maxwell-brown);
    opacity: .65;
  }
  footer a { color: var(--maxwell-purple-2); text-decoration: none; }
</style>
</head>
<body>
<div class="frame">

<header>
  <div class="mascot"><img src="/static/maxwell.png" alt="Maxwell the parrot"></div>
  <div>
    <h1>Hi, I'm Maxwell!</h1>
    <div class="tag">Stanford TEA's resident parrot · come say hi</div>
  </div>
</header>

<section class="card">
  <h2>How would you like to chat?</h2>
  <div class="row" style="margin-bottom:.6rem;">
    <span style="font-weight:600;color:var(--maxwell-brown);min-width:5.5rem;">Mode</span>
    <div class="pill-group" id="modeGroup">
      <button data-mode="realtime" class="active">Realtime</button>
      <button data-mode="conversation">Take turns</button>
    </div>
  </div>
  <div class="row" id="micGroupRow">
    <span style="font-weight:600;color:var(--maxwell-brown);min-width:5.5rem;">Mic</span>
    <div class="pill-group" id="micGroup">
      <button data-mic="auto" class="active">Auto-listen (interrupt)</button>
      <button data-mic="ptt">Push-to-talk</button>
    </div>
  </div>
  <p style="margin:.6rem 0 0;font-size:.9rem;color:var(--maxwell-brown);">
    <b>Realtime</b> keeps an open line — Maxwell hears you the moment you
    talk. <b>Take turns</b> records one phrase at a time. Use
    <b>Push-to-talk</b> in noisy rooms (it's the parrot's hold-to-speak
    button below).
  </p>
</section>

<section class="card">
  <div class="talk-wrap">
    <button id="talkBtn" disabled>Connecting…</button>
    <div id="talkHint"></div>
    <button id="stopBtn" type="button">Stop</button>
    <div id="status">Loading…</div>
  </div>
</section>

<section class="card">
  <h2>Conversation</h2>
  <div id="chat" class="chat">
    <div class="empty" id="emptyHint">Say hi to Maxwell to get started!</div>
  </div>
</section>

<footer>
  Built for Stanford TEA. <a href="/">Admin view</a>
</footer>

</div>

<script>
const $ = id => document.getElementById(id);
async function api(path, body) {
  const r = await fetch(path, {
    method: body ? 'POST' : 'GET',
    headers: {'content-type': 'application/json'},
    body: body ? JSON.stringify(body) : undefined,
  });
  return r.json();
}

let mode = 'realtime';      // 'realtime' | 'conversation'
let micMode = 'auto';       // 'auto' | 'ptt'
let rtRunning = false;
let convBusy = false;
let rtTranscriptCursor = 0;

function setStatus(msg, cls) {
  const el = $('status');
  el.textContent = msg;
  el.className = cls || '';
}

function bubble(role, text) {
  $('emptyHint')?.remove();
  const div = document.createElement('div');
  div.className = 'bubble ' + (role === 'you' ? 'you' : 'bird');
  const who = document.createElement('span');
  who.className = 'who';
  who.textContent = role === 'you' ? 'You' : 'Maxwell';
  const body = document.createElement('span');
  body.textContent = text;
  div.appendChild(who); div.appendChild(body);
  $('chat').appendChild(div);
  $('chat').scrollTop = $('chat').scrollHeight;
}

function setActive(group, dataAttr, value) {
  for (const b of group.querySelectorAll('button')) {
    b.classList.toggle('active', b.dataset[dataAttr] === value);
  }
}

function refreshTalkButton() {
  const btn = $('talkBtn');
  const hint = $('talkHint');
  const stopBtn = $('stopBtn');
  if (mode === 'realtime') {
    if (micMode === 'ptt') {
      btn.textContent = rtRunning ? 'Hold to speak' : 'Start';
      btn.disabled = false;
      hint.textContent = rtRunning
        ? 'Hold the button (or spacebar) while you talk.'
        : 'Tap once to wake Maxwell up.';
    } else {
      btn.textContent = rtRunning ? 'Listening…' : 'Wake Maxwell';
      btn.disabled = false;
      hint.textContent = rtRunning
        ? 'Talk anytime — he\\'ll hear you and reply live.'
        : 'Tap to start a live conversation.';
    }
    stopBtn.classList.toggle('visible', rtRunning);
  } else {
    btn.textContent = convBusy ? 'Listening…' : 'Tap & talk';
    btn.disabled = convBusy;
    hint.textContent = convBusy
      ? 'Speak now, then pause.'
      : 'Tap, speak one thing, pause, and Maxwell will reply.';
    stopBtn.classList.remove('visible');
  }
}

// Mode switching
$('modeGroup').addEventListener('click', async (e) => {
  const t = e.target;
  if (!t.dataset.mode || t.classList.contains('active')) return;
  // Tear down realtime if leaving it
  if (mode === 'realtime' && t.dataset.mode !== 'realtime' && rtRunning) {
    await api('/api/realtime/stop', {});
    rtRunning = false;
  }
  mode = t.dataset.mode;
  setActive($('modeGroup'), 'mode', mode);
  $('micGroupRow').style.display = (mode === 'realtime') ? 'flex' : 'none';
  refreshTalkButton();
  setStatus(mode === 'realtime' ? 'Ready' : 'Ready (take-turns mode)');
});

$('micGroup').addEventListener('click', async (e) => {
  const t = e.target;
  if (!t.dataset.mic || t.classList.contains('active')) return;
  micMode = t.dataset.mic;
  setActive($('micGroup'), 'mic', micMode);
  // Tell server PTT preference; live-toggles without restart
  await api('/api/realtime/config', {push_to_talk: micMode === 'ptt'});
  refreshTalkButton();
});

// Realtime start/stop
async function startRealtime() {
  setStatus('Connecting to Maxwell…', 'busy');
  const r = await api('/api/realtime/start', {});
  if (!r.ok) {
    setStatus('Couldn\\'t start: ' + (r.error || 'unknown error'), 'error');
    return false;
  }
  rtRunning = true;
  setStatus(micMode === 'ptt' ? 'Hold the button to talk' : 'Maxwell is listening', 'live');
  refreshTalkButton();
  return true;
}
async function stopRealtime() {
  await api('/api/realtime/stop', {});
  rtRunning = false;
  setStatus('Stopped');
  refreshTalkButton();
}

// PTT — set ptt=true synchronously on press so a fast press+release
// can't slip through the await on startRealtime() and leave the mic
// stuck open. pttUp waits for any in-flight pttDown to finish so the
// "down" PTT signal is always delivered to the server before "up".
let ptt = false;
let pttDownPromise = null;
async function pttDown() {
  if (mode !== 'realtime' || micMode !== 'ptt' || ptt) return;
  ptt = true;
  $('talkBtn').classList.add('held');
  setStatus('Listening…', 'live');
  pttDownPromise = (async () => {
    if (!rtRunning) await startRealtime();
    if (!rtRunning) {
      ptt = false;
      $('talkBtn').classList.remove('held');
      return;
    }
    await api('/api/realtime/ptt', {action: 'down'});
  })();
  try { await pttDownPromise; } finally { pttDownPromise = null; }
}
async function pttUp() {
  if (!ptt) return;
  ptt = false;
  $('talkBtn').classList.remove('held');
  setStatus('Maxwell is thinking…', 'busy');
  if (pttDownPromise) {
    try { await pttDownPromise; } catch (e) {}
  }
  await api('/api/realtime/ptt', {action: 'up'});
}

// Talk button press/release. pttDown auto-starts the session if it
// isn't running yet, so the user can do "tap to wake + hold to talk"
// in one motion. Window-level release listeners catch pointerup /
// touchend that fired off the button (so PTT never gets stuck open
// when the cursor or finger slides away mid-press).
$('talkBtn').addEventListener('mousedown', e => {
  if (mode === 'realtime' && micMode === 'ptt') { e.preventDefault(); pttDown(); }
});
$('talkBtn').addEventListener('touchstart', e => {
  if (mode === 'realtime' && micMode === 'ptt') {
    e.preventDefault(); pttDown();
  }
}, {passive:false});
window.addEventListener('mouseup', () => { if (ptt) pttUp(); });
window.addEventListener('touchend', () => { if (ptt) pttUp(); });
window.addEventListener('touchcancel', () => { if (ptt) pttUp(); });

$('talkBtn').addEventListener('click', async (e) => {
  // Click semantics depend on mode
  if (mode === 'realtime') {
    if (micMode === 'auto') {
      if (!rtRunning) await startRealtime();
      // While running in auto-listen, the big button is just a status
      // chip ("Listening…"). The user stops the session via the
      // dedicated Stop pill below.
    } else if (!rtRunning) {
      await startRealtime();
    }
    return;
  }
  // Conversation (take-turns) mode
  if (convBusy) return;
  convBusy = true;
  refreshTalkButton();
  setStatus('Listening…', 'live');
  try {
    const r = await api('/api/converse', {});
    if (!r.ok) {
      setStatus('Error: ' + (r.error || 'unknown'), 'error');
    } else {
      if (r.user_text) bubble('you', r.user_text);
      if (r.reply) bubble('bird', r.reply);
      setStatus('Your turn');
    }
  } finally {
    convBusy = false;
    refreshTalkButton();
  }
});

// Stop pill — visible whenever realtime is running in either mic mode.
$('stopBtn').addEventListener('click', async () => {
  if (!rtRunning) return;
  // If PTT is held when stop is clicked, release it cleanly first.
  if (ptt) await pttUp();
  await stopRealtime();
});

// Spacebar = PTT in realtime+ptt
let spaceHeld = false;
window.addEventListener('keydown', e => {
  if (e.code !== 'Space' || e.repeat) return;
  if (mode !== 'realtime' || micMode !== 'ptt') return;
  e.preventDefault();
  if (!spaceHeld) { spaceHeld = true; pttDown(); }
});
window.addEventListener('keyup', e => {
  if (e.code !== 'Space') return;
  if (spaceHeld) { spaceHeld = false; pttUp(); }
});

// Poll transcripts (works for realtime mode; conversation mode uses /api/converse return values)
async function pollTranscripts() {
  if (mode !== 'realtime' || !rtRunning) return;
  try {
    const r = await api('/api/realtime/transcripts?since=' + rtTranscriptCursor);
    if (r.ok && r.items) {
      for (const item of r.items) {
        bubble(item.role === 'user' ? 'you' : 'bird', item.text);
      }
      if (r.last_id != null) rtTranscriptCursor = r.last_id;
    }
  } catch (e) {}
}
setInterval(pollTranscripts, 800);

// Boot: load config to pick up the current PTT setting + connection status
(async () => {
  const cfg = await api('/api/config');
  if (cfg && cfg.realtime && cfg.realtime.push_to_talk) {
    micMode = 'ptt';
    setActive($('micGroup'), 'mic', 'ptt');
  }
  const cs = await api('/api/connection/status');
  if (!cs.connected) {
    setStatus('Maxwell is offline. Ask the operator to plug him in.', 'error');
    $('talkBtn').disabled = true;
    $('talkBtn').textContent = 'Offline';
    $('talkHint').textContent = '';
    return;
  }
  const rs = await api('/api/realtime/status');
  rtRunning = !!rs.running;
  setStatus(rtRunning ? 'Maxwell is listening' : 'Ready', rtRunning ? 'live' : '');
  refreshTalkButton();
})();
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
        # Realtime transcript ring buffer surfaced to the UI. Each entry
        # is a monotonically-increasing dict so the UI can poll with a
        # ``since`` cursor and only fetch what's new since its last
        # call. Capped to keep memory bounded across long sessions.
        self._rt_transcripts: list[dict] = []
        self._rt_transcript_seq = 0
        self._rt_transcript_max = 200

    def push_rt_transcript(self, role: str, text: str) -> None:
        self._rt_transcript_seq += 1
        self._rt_transcripts.append(
            {"id": self._rt_transcript_seq, "role": role, "text": text}
        )
        if len(self._rt_transcripts) > self._rt_transcript_max:
            self._rt_transcripts = self._rt_transcripts[-self._rt_transcript_max:]

    def rt_transcripts_since(self, since: int) -> list[dict]:
        return [t for t in self._rt_transcripts if t["id"] > since]

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

    async def admits_page(_req: web.Request) -> web.Response:
        return web.Response(text=ADMITS_HTML, content_type="text/html")

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
                "realtime": {
                    "model": cfg.realtime.model,
                    "voice": cfg.realtime.voice,
                    "instructions": cfg.realtime.instructions,
                    "vad_type": cfg.realtime.vad_type,
                    "vad_threshold": cfg.realtime.vad_threshold,
                    "vad_prefix_padding_ms": cfg.realtime.vad_prefix_padding_ms,
                    "vad_silence_duration_ms": cfg.realtime.vad_silence_duration_ms,
                    "vad_eagerness": cfg.realtime.vad_eagerness,
                    "noise_reduction": cfg.realtime.noise_reduction,
                    "half_duplex": cfg.realtime.half_duplex,
                    "playback_tail_ms": cfg.realtime.playback_tail_ms,
                    "barge_in_enabled": cfg.realtime.barge_in_enabled,
                    "barge_in_rms_threshold": cfg.realtime.barge_in_rms_threshold,
                    "barge_in_above_ambient_factor": cfg.realtime.barge_in_above_ambient_factor,
                    "barge_in_min_frames": cfg.realtime.barge_in_min_frames,
                    "push_to_talk": cfg.realtime.push_to_talk,
                },
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

    async def _start_realtime_with_current_config(api_key: str) -> None:
        rt = state.config.realtime
        await state.pipeline.start_realtime(
            api_key=api_key,
            model=rt.model,
            voice=rt.voice,
            instructions=rt.instructions or state.config.personality,
            vad_type=rt.vad_type,
            vad_threshold=rt.vad_threshold,
            vad_prefix_padding_ms=rt.vad_prefix_padding_ms,
            vad_silence_duration_ms=rt.vad_silence_duration_ms,
            vad_eagerness=rt.vad_eagerness,
            noise_reduction=rt.noise_reduction,
            half_duplex=rt.half_duplex,
            playback_tail_ms=rt.playback_tail_ms,
            barge_in_enabled=rt.barge_in_enabled,
            barge_in_rms_threshold=rt.barge_in_rms_threshold,
            barge_in_above_ambient_factor=rt.barge_in_above_ambient_factor,
            barge_in_min_frames=rt.barge_in_min_frames,
            push_to_talk=rt.push_to_talk,
            transcript_callback=state.push_rt_transcript,
        )

    async def api_realtime_start(_req: web.Request) -> web.Response:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return web.json_response(
                {"ok": False, "error": "OPENAI_API_KEY not set"}
            )
        try:
            async with state.lock:
                await state.ensure_pipeline()
                await _start_realtime_with_current_config(api_key)
            return web.json_response({"ok": True, "running": True})
        except Exception as e:  # noqa: BLE001
            log.exception("realtime start failed")
            return web.json_response({"ok": False, "error": str(e)})

    async def api_realtime_stop(_req: web.Request) -> web.Response:
        try:
            async with state.lock:
                if state.pipeline is not None:
                    await state.pipeline.stop_realtime()
            return web.json_response({"ok": True, "running": False})
        except Exception as e:  # noqa: BLE001
            log.exception("realtime stop failed")
            return web.json_response({"ok": False, "error": str(e)})

    async def api_realtime_status(_req: web.Request) -> web.Response:
        running = bool(
            state.pipeline is not None and state.pipeline.realtime_running
        )
        return web.json_response({"ok": True, "running": running})

    async def api_realtime_ptt(req: web.Request) -> web.Response:
        """Push-to-talk down/up signal. Body: ``{"action": "down"|"up"}``.

        Down opens the mic and cancels any in-flight assistant
        response (acts as immediate barge-in). Up commits the buffered
        audio and asks for a response. Both are no-ops when PTT mode
        isn't on or the realtime session isn't running."""
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = {}
        action = str(body.get("action", "")).lower()
        if state.pipeline is None or not state.pipeline.realtime_running:
            return web.json_response(
                {"ok": False, "error": "realtime not running"}
            )
        if action == "down":
            ok = await state.pipeline.realtime_ptt_down()
        elif action == "up":
            ok = await state.pipeline.realtime_ptt_up()
        else:
            return web.json_response(
                {"ok": False, "error": "action must be 'down' or 'up'"}
            )
        return web.json_response({"ok": bool(ok), "action": action})

    async def api_connect(_req: web.Request) -> web.Response:
        """(Re)build the pipeline + open the serial backend."""
        try:
            async with state.lock:
                if state.pipeline is not None:
                    return web.json_response(
                        {"ok": True, "connected": True, "noop": True}
                    )
                await state._create_pipeline()
            return web.json_response({"ok": True, "connected": True})
        except Exception as e:  # noqa: BLE001
            log.exception("connect failed")
            return web.json_response({"ok": False, "error": str(e)})

    async def api_disconnect(_req: web.Request) -> web.Response:
        """End the session: stop realtime, center servos, tear down
        the pipeline (closing the serial port). The webapp stays up;
        click Connect to bring it back."""
        try:
            from motion.models import MotionFrame
            async with state.lock:
                if state.pipeline is not None and state.pipeline.realtime_running:
                    try:
                        await state.pipeline.stop_realtime()
                    except Exception:  # noqa: BLE001
                        log.exception("disconnect: stop_realtime failed")
                if state.backend is not None:
                    try:
                        await state.backend.send_frame(
                            MotionFrame(
                                jaw_open=0.0,
                                head_lr=0.5,
                                head_ud=0.5,
                                wing=0.0,
                            )
                        )
                        await asyncio.sleep(0.2)
                    except Exception:  # noqa: BLE001
                        log.exception("disconnect: failed to center servos")
                if state.pipeline is not None:
                    try:
                        await state.pipeline.__aexit__(None, None, None)
                    except Exception:  # noqa: BLE001
                        log.exception("disconnect: pipeline teardown failed")
                    state.pipeline = None
                state.backend = None
            return web.json_response({"ok": True, "connected": False})
        except Exception as e:  # noqa: BLE001
            log.exception("disconnect failed")
            return web.json_response({"ok": False, "error": str(e)})

    async def api_connection_status(_req: web.Request) -> web.Response:
        connected = state.pipeline is not None
        return web.json_response({"ok": True, "connected": connected})

    async def api_realtime_transcripts(req: web.Request) -> web.Response:
        try:
            since = int(req.query.get("since", "0"))
        except ValueError:
            since = 0
        items = state.rt_transcripts_since(since)
        last_id = items[-1]["id"] if items else since
        return web.json_response(
            {"ok": True, "items": items, "last_id": last_id}
        )

    async def api_realtime_config(req: web.Request) -> web.Response:
        body = await req.json()
        rt = state.config.realtime
        changed = False
        if "voice" in body:
            v = str(body["voice"]).strip()
            if v and v != rt.voice:
                rt.voice = v
                changed = True
        if "model" in body:
            m = str(body["model"]).strip()
            if m and m != rt.model:
                rt.model = m
                changed = True
        if "instructions" in body:
            inst = str(body["instructions"])
            if inst != rt.instructions:
                rt.instructions = inst
                changed = True
        if "vad_type" in body:
            vt = str(body["vad_type"]).strip()
            if vt in ("server_vad", "semantic_vad") and vt != rt.vad_type:
                rt.vad_type = vt
                changed = True
        if "vad_threshold" in body:
            try:
                t = float(body["vad_threshold"])
            except (TypeError, ValueError):
                t = rt.vad_threshold
            t = max(0.0, min(1.0, t))
            if abs(t - rt.vad_threshold) > 1e-6:
                rt.vad_threshold = t
                changed = True
        if "vad_prefix_padding_ms" in body:
            try:
                p = int(body["vad_prefix_padding_ms"])
            except (TypeError, ValueError):
                p = rt.vad_prefix_padding_ms
            p = max(0, min(2000, p))
            if p != rt.vad_prefix_padding_ms:
                rt.vad_prefix_padding_ms = p
                changed = True
        if "vad_silence_duration_ms" in body:
            try:
                s = int(body["vad_silence_duration_ms"])
            except (TypeError, ValueError):
                s = rt.vad_silence_duration_ms
            s = max(50, min(5000, s))
            if s != rt.vad_silence_duration_ms:
                rt.vad_silence_duration_ms = s
                changed = True
        if "vad_eagerness" in body:
            e = str(body["vad_eagerness"]).strip()
            if e in ("low", "medium", "high", "auto") and e != rt.vad_eagerness:
                rt.vad_eagerness = e
                changed = True
        if "noise_reduction" in body:
            n = str(body["noise_reduction"]).strip()
            if n in ("off", "near_field", "far_field") and n != rt.noise_reduction:
                rt.noise_reduction = n
                changed = True
        if "half_duplex" in body:
            hd = bool(body["half_duplex"])
            if hd != rt.half_duplex:
                rt.half_duplex = hd
                changed = True
        if "playback_tail_ms" in body:
            try:
                tail = int(body["playback_tail_ms"])
            except (TypeError, ValueError):
                tail = rt.playback_tail_ms
            tail = max(0, min(2000, tail))
            if tail != rt.playback_tail_ms:
                rt.playback_tail_ms = tail
                changed = True
        if "barge_in_enabled" in body:
            bi = bool(body["barge_in_enabled"])
            if bi != rt.barge_in_enabled:
                rt.barge_in_enabled = bi
                changed = True
        if "barge_in_rms_threshold" in body:
            try:
                v = float(body["barge_in_rms_threshold"])
            except (TypeError, ValueError):
                v = rt.barge_in_rms_threshold
            v = max(0.0, min(1.0, v))
            if abs(v - rt.barge_in_rms_threshold) > 1e-6:
                rt.barge_in_rms_threshold = v
                changed = True
        if "barge_in_above_ambient_factor" in body:
            try:
                v = float(body["barge_in_above_ambient_factor"])
            except (TypeError, ValueError):
                v = rt.barge_in_above_ambient_factor
            v = max(1.0, min(20.0, v))
            if abs(v - rt.barge_in_above_ambient_factor) > 1e-6:
                rt.barge_in_above_ambient_factor = v
                changed = True
        if "barge_in_min_frames" in body:
            try:
                v = int(body["barge_in_min_frames"])
            except (TypeError, ValueError):
                v = rt.barge_in_min_frames
            v = max(1, min(50, v))
            if v != rt.barge_in_min_frames:
                rt.barge_in_min_frames = v
                changed = True
        # Push-to-talk toggles live without restarting the WebSocket:
        # the session_update path on RealtimeSession reconfigures
        # turn_detection in place. Only flip the cached config; the
        # live toggle below applies it without a session restart.
        ptt_changed_live = False
        if "push_to_talk" in body:
            ptt = bool(body["push_to_talk"])
            if ptt != rt.push_to_talk:
                rt.push_to_talk = ptt
                ptt_changed_live = True
        was_running = bool(
            state.pipeline is not None and state.pipeline.realtime_running
        )
        if ptt_changed_live and was_running:
            try:
                await state.pipeline.realtime_set_push_to_talk(rt.push_to_talk)
            except Exception:  # noqa: BLE001
                log.exception("realtime: live PTT toggle failed")
        if changed and was_running:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return web.json_response(
                    {"ok": False, "error": "OPENAI_API_KEY not set"}
                )
            try:
                async with state.lock:
                    await _start_realtime_with_current_config(api_key)
            except Exception as e:  # noqa: BLE001
                log.exception("realtime restart failed")
                return web.json_response({"ok": False, "error": str(e)})
        log.info(
            "realtime config: voice=%s model=%s vad=%s thr=%.2f silence=%dms noise=%s (running=%s)",
            rt.voice,
            rt.model,
            rt.vad_type,
            rt.vad_threshold,
            rt.vad_silence_duration_ms,
            rt.noise_reduction,
            was_running,
        )
        return web.json_response(
            {
                "ok": True,
                "running": was_running,
                "voice": rt.voice,
                "model": rt.model,
                "vad_type": rt.vad_type,
                "vad_threshold": rt.vad_threshold,
                "vad_prefix_padding_ms": rt.vad_prefix_padding_ms,
                "vad_silence_duration_ms": rt.vad_silence_duration_ms,
                "vad_eagerness": rt.vad_eagerness,
                "noise_reduction": rt.noise_reduction,
                "half_duplex": rt.half_duplex,
                "playback_tail_ms": rt.playback_tail_ms,
            }
        )

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
    app.router.add_get("/admits", admits_page)
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
    app.router.add_post("/api/realtime/start", api_realtime_start)
    app.router.add_post("/api/realtime/stop", api_realtime_stop)
    app.router.add_get("/api/realtime/status", api_realtime_status)
    app.router.add_get("/api/realtime/transcripts", api_realtime_transcripts)
    app.router.add_post("/api/realtime/ptt", api_realtime_ptt)
    app.router.add_post("/api/connect", api_connect)
    app.router.add_post("/api/disconnect", api_disconnect)
    app.router.add_get("/api/connection/status", api_connection_status)
    app.router.add_post("/api/realtime/config", api_realtime_config)

    # Static assets (Maxwell pic, etc.). Resolved relative to the
    # repo root so it works whether the app was started from the repo
    # root or via `python -m app.webapp`.
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        app.router.add_static("/static", str(static_dir), show_index=False)

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

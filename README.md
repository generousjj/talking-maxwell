# Maxwell — talking parrot animatronic

A Bottango-driven animatronic parrot that listens, talks, and moves in real time. Built for Stanford TEA. The bird (Maxwell) is the [Bottango Maxwell kit](https://www.bottango.com/) running custom firmware over USB-serial; the brain is a small Python web app on the laptop he's plugged into.

There are two web pages once it's running:

- [http://127.0.0.1:8787/admits](http://127.0.0.1:8787/admits) — the **end-user view**. Big round talk button, parrot-friendly colors, pick "Realtime" or "Take turns" and "Auto-listen" or "Push-to-talk". This is the one to point visitors at.
- [http://127.0.0.1:8787/](http://127.0.0.1:8787/) — the **operator view**. Connect/End-Session buttons, every tuning knob exposed, raw log pane. This is for the person babysitting Maxwell at the booth.

---

## 🦜 Running Maxwell at an event (no Python knowledge needed)

If you're the person at the booth and you just want to plug Maxwell in and have him work, this section is for you. **You only need to do steps 1–3 once per laptop.** After that, every event is just step 4.

### What you need

- A Mac laptop (the project also runs on Linux; Windows works but isn't bundled with the easy launcher).
- The Maxwell kit's USB cable.
- An OpenAI API key (it'll ask you the first time and remember it). Get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
- AirPods / a small Bluetooth speaker work, but built-in speakers are fine. The mic is whatever Mac mic is the system default.

### Steps

1. **Install Python 3.10 or newer.** macOS already has `python3` installed via the Command Line Tools. If you're not sure, open Terminal and run `python3 --version`. If it complains, install from [python.org/downloads](https://www.python.org/downloads/).

2. **Get the project onto the laptop.** Either:
   - Download the zip from GitHub: [github.com/generousjj/talking-maxwell](https://github.com/generousjj/talking-maxwell) → green "Code" button → "Download ZIP", then unzip it somewhere convenient like the Desktop. **OR**
   - In Terminal: `git clone https://github.com/generousjj/talking-maxwell.git ~/Desktop/maxwell`

3. **Plug Maxwell into the laptop with the USB cable.** Wait a few seconds — macOS will register the USB-serial device.

4. **Start Maxwell.** In Finder, navigate to the project folder and **double-click `Start Maxwell.command`**. (The first time macOS may complain about an unidentified developer — Right-click → Open → "Open" in the dialog.)

   The script will:
   - Set up Python dependencies (only the first time, takes ~30 seconds)
   - Ask you to paste your OpenAI key (only the first time)
   - Detect Maxwell's USB port automatically
   - Open the admits-friendly tab in your browser
   - Start the server

   When you see `Maxwell web UI on http://127.0.0.1:8787` in the Terminal window, he's ready.

5. **Hand visitors the admits tab** (`/admits`). Have it open on a tablet or another browser window.

6. **When you're done:** click **End session** in the operator view (or hit Ctrl-C in the Terminal window). End session will gently center all servos before cutting power so nothing is left straining.

### Troubleshooting at the booth

- **Maxwell's beak doesn't move during speech.** The jaw servo's GPIO 9 line is finicky. Try:
   1. Click **Full reset** in the operator view (deregisters + re-registers all servos and "hammers" the jaw line to wake it up).
   2. If still nothing, physically unplug-and-replug the small jumper on pin 9 of the ESP32, then click **Full reset** again.
- **Mic keeps mistaking room noise for speech.** Open the operator view → realtime row → "Mic sensitivity" panel → switch to **Push-to-talk**, OR raise the threshold / silence-duration sliders.
- **Maxwell "hears himself" and won't stop talking.** The "Echo guard" toggle should already be on. If it's not, turn it on. If it is and he's still self-looping, the speakers are too close to the mic — move them, or use Push-to-talk.
- **The page won't load.** Make sure the Terminal window is still open and shows `Maxwell web UI on http://127.0.0.1:8787`. If it crashed, just double-click `Start Maxwell.command` again.
- **Want a public URL** so admits can chat from their phones? Easiest tunnel: `cloudflared tunnel --url http://localhost:8787` from a second Terminal window. They'll get a `*.trycloudflare.com` URL pointing at your laptop. The mic + speaker are still the laptop's, but visitors can type and Maxwell will speak it aloud at the booth.

---

## For developers

### Project layout

```
app/             entry points: webapp.py, cli.py, pipeline.py, config.py
conversation/    STT, LLM, TTS providers + Realtime API session
motion/          envelope follower, behavior engine, state machine, models
transport/       Bottango serial backend (talks raw protocol to ESP32)
tests/           pytest unit tests
config.yaml      live tuning (committed; edit on the booth machine if needed)
config.example.yaml   template / annotated source-of-truth defaults
```

### Setup (manual)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
python -m app.webapp --backend bottango      # real hardware
python -m app.webapp --backend mock           # logs motion only
```

### Tests

```bash
python -m pytest -q
```

30-ish unit tests cover the envelope follower, behavior engine, state machine, realtime event parsing, and config loading. They run in <2 s with no hardware required.

### Configuration

`config.example.yaml` is the source of truth for defaults; `config.yaml` overrides per-booth. The dataclass-walker auto-loads any field defined on `AppConfig`, so adding a new tunable means: define it on the dataclass with a default → it's instantly readable from YAML, no glue code.

Notable config sections:

- `personality` — system prompt for the LLM. Maxwell is the Stanford TEA mascot, English-only, warm + funny.
- `motion.jaw` — envelope-follower shaping (floor, ceiling, gain, attack, release, peak hold). Currently restored to the post-invert-fix baseline (gain=1.6, floor=0.08, ceiling=0.90).
- `motion.behavior` — magnitudes for head bobs, wing flaps, idle motion (continuous sine waves on non-harmonic periods).
- `bottango.serial` — port, baud, per-channel servo PWM ranges + invert flag, slew limits.
- `realtime` — OpenAI Realtime API tuning (voice, VAD threshold/silence, far/near-field noise reduction, half-duplex echo guard, smart barge-in, push-to-talk).

### Architecture, briefly

```
mic → STT/Realtime → LLM → TTS/Realtime audio
                              ↓
                       EnvelopeFollower (RMS → jaw target)
                              ↓
                  ConversationStateMachine (idle/listen/think/speak)
                              ↓
                  BehaviorEngine (heuristic head/wing motion)
                              ↓
                  MotionScheduler (30 Hz fixed-rate)
                              ↓
                  BottangoSerialBackend (sCI/sC commands → USB)
                              ↓
                          ESP32 → servos
```

Realtime mode replaces the STT→LLM→TTS triplet with a single OpenAI Realtime websocket session, while still feeding the playback audio's RMS into the same envelope follower so the jaw stays in sync with what the speaker is actually saying.

### Notes & quirks

- Jaw servo (GPIO 9) is unreliable on long extension wires. The "Full reset" button + a "Pre-warm jaw" toggle exist to work around it.
- Everything is fire-and-forget on the serial line: we don't wait for the firmware's `OK` per command, with per-channel coalescing and `min_delta_for_send` filtering to avoid drowning the line.
- Half-duplex echo guard mutes the mic while Maxwell is speaking; smart barge-in monitors local mic RMS and un-mutes (cancelling the in-flight response) when the user is clearly louder than the speaker leakage for ~150 ms.

---

Built quickly, intentionally — Codex helped throughout. PRs welcome.

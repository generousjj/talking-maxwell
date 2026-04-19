# Maxwell Animatronic Chatbot Prototype

A modular Python prototype that lets you talk to your Bottango-driven
**Maxwell** animatronic parrot. The laptop is the AI brain; the ESP32 keeps
its normal Bottango role as a servo driver.

## What it does

1. Listens via your laptop microphone (or accepts typed text).
2. Transcribes speech (OpenAI Whisper by default).
3. Generates a short parrot-style reply (OpenAI chat by default).
4. Speaks the reply (OpenAI TTS by default, macOS `say` fallback available).
5. Drives Maxwell's jaw/head/wing in real time while the reply plays, through
   a pluggable motion backend (mock for development, Bottango for hardware).

## Folder structure

```
maxwell_parrot_bot/
├── app/
│   ├── cli.py              # argparse CLI entry point
│   ├── config.py           # YAML + .env config loader
│   ├── main.py             # alias entry point
│   └── pipeline.py         # typed + live conversation pipelines
├── conversation/
│   ├── audio.py            # playback + mic + WAV helpers
│   ├── stt.py              # STTProvider + OpenAI Whisper + stub
│   ├── llm.py              # LLMProvider + OpenAI + offline stub
│   └── tts.py              # TTSProvider + OpenAI + macOS say + sine stub
├── motion/
│   ├── behavior_engine.py  # deterministic jaw/head/wing heuristics
│   ├── envelope.py         # RMS envelope follower w/ attack-release
│   ├── models.py           # dataclasses + normalized MotionFrame
│   ├── scheduler.py        # ~30 Hz motion scheduler
│   └── state_machine.py    # IDLE/LISTENING/THINKING/SPEAKING
├── transport/
│   ├── base.py                     # MotionBackend ABC
│   ├── bottango_protocol.py        # Bottango wire protocol + hash
│   ├── bottango_serial_backend.py  # direct USB-serial to ESP32 (default)
│   ├── bottango_backend.py         # legacy HTTP client for Bottango desktop
│   └── mock_backend.py             # console/CSV/matplotlib backend
├── tests/                  # pytest unit tests
├── config.example.yaml
├── .env.example
├── requirements.txt
└── README.md
```

The Bottango project file (`*.btngo`) and its associated media assets that live
**outside** this folder (in the parent `/Users/judestjohn`) are treated as
**read-only reference material** for this prototype. Nothing in this project
reads, writes, renames, or otherwise modifies them; their role is only to
document Maxwell's control names and servo ranges for our human reference.

## Install

Requires Python **3.11+**. Python 3.13 works too.

```bash
cd /Users/judestjohn/maxwell_parrot_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
cp .env.example .env          # then edit and set OPENAI_API_KEY
```

## Running

### Web UI (easiest — type a script, click Run)

```bash
python -m app.webapp --backend bottango_serial --config config.example.yaml
# then open http://localhost:8787
```

The web UI keeps a single pipeline alive across requests, so the 2-3 second
serial handshake only happens once at startup. Features:

- textarea + **Speak** button — type anything, Maxwell says it
- **Replay test clip** button — plays `/tmp/maxwell_short.wav`
- **Center servos** / **Stop** buttons
- live jaw tuning (invert, gain, min/max PWM) without restart
- rolling log pane

Add `--safe-providers` to use offline stub LLM + macOS `say` when you don't
have `OPENAI_API_KEY` set.

### Mock mode (works immediately, no API keys required)

```bash
python -m app.cli --mode typed --backend mock --safe-providers \
  --text "Hey Maxwell, how are you?"
```

You'll see motion frames printed to the console and, if you pass `--plot`, a
matplotlib plot of jaw / head_lr / head_ud / wing when the session exits.

```bash
python -m app.cli --mode typed --backend mock --csv motion.csv --plot
```

### Typed-text mode (Priority 1)

```bash
python -m app.cli --mode typed --backend mock
# then type a line, press enter, Maxwell replies and "moves"
```

Drop `--safe-providers` and set `OPENAI_API_KEY` in `.env` to get real OpenAI
replies and voice.

### Live conversation mode (Priority 2)

```bash
python -m app.cli --mode live --backend mock
```

This uses the laptop mic with a simple energy-threshold VAD. Settings live
under `audio:` in `config.yaml`. If you don't want OpenAI Whisper, set
`providers.stt: stub` (you'll need typed mode) or swap in a different STT.

### Replay / demo mode

Drive motion from a pre-recorded WAV file without involving the chatbot:

```bash
python -m app.cli --mode replay --wav some_audio.wav --backend mock --plot
```

## Bottango hardware setup

The default hardware path is the **serial backend**, which talks Bottango's
Arduino Driver protocol (API version 8) **directly to the ESP32 over USB**.
That means:

- **No Bottango Desktop app is required.** Close it if it's open; it will
  hold the serial port.
- **No `controlSchemes`, API-controlled inputs, or live-mode routing** to
  configure inside Bottango's GUI. We register the four servos ourselves on
  connect using the PWM ranges from `config.yaml`.
- The only Bottango step that's still required is the one you've already
  done: flash **BottangoArduinoDriver** to the ESP32 using Bottango Desktop.

### Prerequisites

1. ESP32 already has `BottangoArduinoDriver` uploaded (driver version
   `0.7.0a1` or compatible — we use the `CUSTOM8` handshake so minor driver
   version drift is fine).
2. ESP32 is plugged in over USB (shows up as a `CP2102`/`CH340`/`FTDI`
   serial device).
3. `pyserial` installed via `requirements.txt`.

### Run it

```bash
# Close Bottango Desktop first if it's open (it will hold the serial port).
python -m app.cli --mode replay --wav some_audio.wav --backend bottango_serial
# or
python -m app.cli --mode typed  --backend bottango_serial
python -m app.cli --mode live   --backend bottango_serial
```

The app will auto-detect the ESP32's serial port, perform the handshake,
register the four servos using the values in
`bottango.serial.{jaw,head_lr,head_ud,wing}`, and start streaming motion.

### Safety rails (enforced by firmware)

The PWM ranges in `config.yaml` are enforced **by the firmware itself** once
registered. Even if a bug in the behavior engine asks for a larger motion,
Bottango's `PinServoEffector` clamps to `min_pwm..max_pwm` and caps slew
rate at `max_pwm_per_sec`. Defaults match the Maxwell reference project:

| Channel  | Pin | PWM range     | Max μs/sec |
| -------- | --- | ------------- | ---------- |
| jaw      | 9   | 1450 – 1700   | 3000       |
| head_lr  | 5   | 1275 – 1725   | 1800       |
| head_ud  | 6   | 850 – 2100    | 1800       |
| wing     | 3   | 1500 – 2000   | 3000       |

Tighten any of these in `config.yaml` before first power-on if you want even
more conservative first movements.

### Troubleshooting

- **"Could not locate a Bottango-compatible serial device"** — plug the ESP32
  in, or pass `--serial-port /dev/cu.usbserial-0001`.
- **Port open fails with a permission or busy error** — Bottango Desktop is
  almost certainly still connected; close it (or use the "Disconnect Driver"
  button in the Bottango Desktop hardware panel).
- **Timed out waiting for OK after `hRQ,...`** — the firmware is in an odd
  state. Unplug/replug the ESP32 and retry. You can also run the tiny smoke
  test to isolate the problem: `python tools/smoke_test_hardware.py`.
- **`HASH_FAIL`** — shouldn't happen (we compute and include the Bottango
  sum-of-ASCII hash on every command), but if it does, open an issue with
  the exact command from the debug log.

### Audio output (playback)

Audio plays out of the **laptop speakers**, not the Arduino/ESP32. The
default in `config.example.yaml` explicitly selects
`"MacBook Pro Speakers"` so AirPods or other default devices don't steal
the audio. Override with `--playback-device "<name or index>"` or
`audio.playback_device` in `config.yaml`.

### Legacy HTTP backend (rare)

A `--backend bottango_http` option exists for Bottango builds that expose an
HTTP REST API matching `bottango.path_template`. Most builds expose the live
API over WebSocket instead, so this path is only useful if you've installed
a matching plugin. Use the serial backend unless you have a specific reason
not to.

## Configuration surface

See `config.example.yaml` for every tunable. Highlights:

- `providers.*` — pick OpenAI / stub / macOS `say` / sine stub per leg.
- `motion.rate_hz` — motion update rate (default 30).
- `motion.jaw.*` — envelope floor / ceiling / attack / release / noise floor /
  peak-hold / gain — all the knobs that shape the jaw.
- `motion.behavior.*` — drift magnitudes, nod strength, wing cooldown, etc.
- `audio.*` — mic device, playback device (defaults to MacBook Pro Speakers),
  sample rate, VAD threshold + hangover.
- `bottango.transport` — `serial` (default, direct USB to ESP32) or `http`.
- `bottango.serial.{jaw,head_lr,head_ud,wing}` — per-channel PWM safety
  ranges that the firmware enforces.
- `logging.motion_csv_path` / `logging.plot_after` — offline inspection hooks.

## What parts are placeholders

- `LocalStubSTT` simply errors out — wire up OpenAI Whisper (or any other
  STT) when you want live-mic mode without the default provider.
- `SineStubTTS` is a deliberate low-quality fallback so the motion pipeline
  works on machines without OpenAI / `say`.
- Phrase boundaries are estimated from punctuation and elapsed playback time
  — we don't yet consume provider-supplied word timings.

## Known limitations

- The VAD is a plain energy threshold; noisy rooms may need a higher
  `vad_threshold` or push-to-talk (not yet implemented).
- Serial backend assumes the ESP32 runs `BottangoArduinoDriver` v0.7.x or
  later (API version 8). Older firmware used `COMPRESSED_SIGNAL_MAX=1000`
  instead of `8192`; set `bottango.serial.compressed_signal_max: 1000` if
  needed.
- No interruption handling yet (you can't cut off the parrot mid-reply).
- Latency is dominated by OpenAI API round-trips when cloud providers are
  enabled.

## Next improvements (not in this pass)

- Push-to-talk and better VAD (e.g. `webrtcvad`).
- Interruption cancellation (cancel TTS + motion mid-utterance).
- Vowel-bias jaw modulation using phoneme hints.
- Singing/music demo mode that consumes WAV + optional score.
- Provider-side timing data (e.g. OpenAI TTS word timings) for precise
  phrase-boundary nods.
- Simple GUI (only if you actually want it — CLI is first-class).

## Running the tests

```bash
python -m pytest tests/
```

All tests run offline; they exercise the envelope follower, behavior engine,
and motion frame invariants.

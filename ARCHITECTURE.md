# Maxwell — End-to-End Architecture

This document explains how the whole project works, piece by piece, from the moment
you run a command to the moment a servo moves. It complements `README.md` (which is
the operator's quick-start guide) with the internals: what calls what, what talks to
OpenAI, how audio becomes motion, and how the two deployment modes differ.

Written by reading the actual source, not the docs — file paths and line-level
behavior are called out throughout so you can jump straight to the code.

> **Current booth setup (Treefest club fair):** the personality is framed for
> Treefest, Stanford's club fair, with emphasis on TEA being a **project-based**
> club (§6.1). Vision runs with **face tracking on**, **face memory off**, and the
> **paid "what do you see?" image analysis off** — each is a one-line switch in
> `config.yaml`'s `vision:` block (§9).

---

## 1. What Maxwell is

Maxwell is an animatronic parrot (the **Bottango Maxwell kit** — physical servos:
jaw/mouth, head left-right, head up-down, wing — driven by an ESP32 running
Bottango's open firmware) built for **Stanford TEA** (Themed Entertainment
Association). A laptop is plugged into him over USB and runs a small Python
program that:

1. Listens to (or reads typed text from) a person talking to Maxwell.
2. Sends that to an LLM (OpenAI) to generate an in-character reply.
3. Synthesizes speech for the reply (or, in **Realtime mode**, does 1–3 as one
   continuous speech-to-speech OpenAI session).
4. Analyzes the outgoing audio's loudness in real time and drives Maxwell's jaw to
   match it, while a separate heuristic "behavior engine" adds head bobs, wing
   flaps, nods, and idle fidgeting so he looks alive even when not talking.
5. Optionally watches a webcam so Maxwell's head tracks the nearest face, and can
   (still WIP, see snapshot note above) recognize returning visitors by name and
   greet them proactively.
6. Serves a small website so a person can operate/talk to him from a browser.

There are **two independent ways to run all of this**, and understanding which one
you're in is the single most important thing for reading the rest of this doc:

| | **Local operator mode** | **Hosted browser mode** |
|---|---|---|
| Entry point | `python -m app.webapp` (or `app.cli` for terminal-only) | `python -m app.web_app`, or `api/index.py` on Vercel |
| Where hardware I/O happens | **In the Python process** on the laptop, over USB-serial | **In the visitor's browser**, via the Web Serial API |
| Where the OpenAI Realtime session runs | **In the Python process**, over a WebSocket | **In the browser**, over WebRTC, using a short-lived token the server mints |
| `OPENAI_API_KEY` location | Loaded into the Python process's environment | Stays on the server only; browser never sees it |
| Vision (face tracking/recognition/scene) | Supported (Python + OpenCV/MediaPipe/InsightFace) | **Face tracking only** — client-side via the native `FaceDetector` (`js/vision.js`); recognition + scene not ported |
| Deployable to Vercel/Fly/Docker? | No — needs a real USB port | Yes |
| Who uses it | Whoever is standing at the booth with the laptop | Anyone with the URL; hardware still needs to be plugged into *some* laptop running the page |

Both modes share the exact same **behavior algorithm** (there's a Python
implementation and a hand-ported, parity-tested JavaScript implementation), the
exact same **Bottango wire protocol**, and the exact same **`config.yaml`** tuning
values — they were deliberately built to feel identical to a user, just with the
hardware/AI plumbing living in a different place.

---

## 2. Repository map

```
app/             Entry points + orchestration
  cli.py           Terminal entry point (typed / live / replay modes)
  main.py          `python -m app.main` == `python -m app.cli`
  webapp.py        LOCAL operator web server (aiohttp) — owns the serial port
  web_app.py       HOSTED browser-mode web server (aiohttp) — no hardware
  web_auth.py      aiohttp auth glue (cookies, login/logout handlers)
  auth_core.py     Framework-agnostic auth primitives (shared by web_app + api/index)
  config.py        YAML+env config loader; the AppConfig dataclass tree
  motion_config.py Serves config.yaml's motion tuning as JSON for the browser build
  personality.py   Loads the system prompt from config.yaml for the web builds
  pipeline.py      ConversationPipeline — wires providers + motion + vision together

conversation/    STT / LLM / TTS providers + the OpenAI Realtime session (Python)
  stt.py, llm.py, tts.py   Provider ABCs + OpenAI implementations + offline stubs
  audio.py                 sounddevice playback/record + envelope streaming
  realtime.py              RealtimeSession: WebSocket speech-to-speech, VAD,
                            half-duplex echo guard, barge-in, push-to-talk, tools

motion/          The "how Maxwell moves" pipeline (backend-agnostic)
  models.py            MotionFrame, SpeakingContext, GazeContext, JawCalibration,
                        BehaviorGains, ConversationState — the shared data model
  envelope.py          EnvelopeFollower: audio RMS -> smoothed jaw position
  behavior_engine.py   BehaviorEngine: state-driven head/wing/jaw heuristics
  state_machine.py     ConversationStateMachine: idle/listening/thinking/speaking
  scheduler.py         MotionScheduler: 30 Hz tick loop -> transport.send_frame()

transport/       Getting motion frames to the physical servos
  base.py                      MotionBackend ABC
  bottango_protocol.py         Bottango's ASCII wire protocol (command builders)
  bottango_serial_backend.py   USB-serial backend (the one actually used)
  bottango_backend.py          Legacy HTTP backend (Bottango Desktop's API)
  mock_backend.py              Logs/CSV/plot backend, no hardware needed

vision/          Webcam features (local operator mode only; see snapshot note)
  camera.py            OpenCV camera capture (+ a mock camera for tests)
  face_detector.py     MediaPipe / OpenCV Haar face detection
  face_tracker.py      12 Hz loop: detect -> pick primary face -> GazeContext
  face_recognizer.py   InsightFace (ArcFace) embeddings
  face_memory.py       name -> embeddings store, optional disk persistence
  recognition.py       3 Hz loop: embed -> match -> temporal-vote -> commit identity
  scene.py             On-demand "what do you see" vision-LLM call

static/web/      HOSTED browser-mode frontend (served by web_app.py / api/index.py)
  index.html, admits.html, sing.html, login.html    The four Maxwell pages
  relic.html           Separate Web Serial control panel for the Relic artifact prop
  js/serial.js         Web Serial port of transport/bottango_serial_backend.py
  js/realtime.js       WebRTC port of conversation/realtime.py
  js/behavior.js       JS port of motion/behavior_engine.py (parity-tested)
  js/envelope.js       JS port of motion/envelope.py
  js/motion.js         JS port of motion/scheduler.py
  js/live_speaking_context.js   JS port of app.pipeline.LiveSpeakingContext
  js/typed.js          Typed-turn fallback (LLM+TTS run server-side)
  js/sing.js           Song lip-sync ("jukebox") page logic
  js/admits.js, app.js, login.js, auth.js, bottango.js, audio_devices.js

api/index.py     FastAPI mirror of app/web_app.py, for Vercel's Python runtime
tools/           Standalone dev/calibration scripts (see §15)
tests/           pytest unit tests (~30, <2s, no hardware required)
config.yaml            Live tuning values (committed; edit on the booth machine)
config.example.yaml    Annotated source-of-truth defaults
Dockerfile              Runs hosted mode (app.web_app) in a container
vercel.json             Routes non-static paths to api/index.py on Vercel
pyproject.toml          Vercel's dependency list (mirror of api/requirements.txt)
run_maxwell.sh          Booth bootstrap script (venv, deps, .env, launch)
Start Maxwell.command   Double-click wrapper around run_maxwell.sh (macOS)
```

---

## 3. Starting Maxwell from the command line

### 3.1 Local operator mode — the easy way (booth laptop)

Double-clicking **`Start Maxwell.command`** just `cd`s into the repo and execs
`run_maxwell.sh`, which is the actual bootstrap logic:

1. Checks `python3` exists and is ≥3.10.
2. Creates `.venv/` if missing, `pip install -r requirements.txt` (only if
   `requirements.txt` is newer than the `.venv/.maxwell-deps-installed` sentinel
   file — so repeat launches skip this).
3. If `.env` has no `OPENAI_API_KEY=sk-...` line, prompts for one interactively
   and writes it to `.env`.
4. Best-effort scans serial ports for something that looks like an ESP32
   (`usbserial`/`usbmodem` in the device path, or `CP210`/`CH340`/`Silicon Labs`
   in the description) and pauses with a reminder if nothing is found.
5. Frees port 8787 if a previous crashed run is still holding it (`lsof` + `kill`).
6. Backgrounds a small polling loop that opens
   `http://127.0.0.1:8787/admits` in the default browser **the moment the port
   actually starts accepting connections** (not on a fixed timer — this avoids the
   "browser says can't connect" race on a slow laptop).
7. Runs `python3 -m app.webapp --backend bottango` in the foreground. If it exits
   with anything other than 0 or 130 (Ctrl-C), the window stays open with the
   traceback and common-fixes hints instead of closing immediately.

### 3.2 Local operator mode — manual

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env

python -m app.webapp --backend bottango          # real hardware, http://127.0.0.1:8787
python -m app.webapp --backend mock              # no hardware; motion just gets logged
```

`app/webapp.py` is a single aiohttp process that keeps **one long-lived
`ConversationPipeline`** alive across every HTTP request (so the ~2–3s serial
handshake only happens once, at boot, not per-request). It serves two pages —
covered in §12 — plus a JSON API the pages call.

There's also a headless terminal mode with no web server at all, useful for
scripting or quick smoke tests, via **`app/cli.py`** (equivalently
`python -m app.main`):

```bash
python -m app.cli --mode typed --backend mock                  # type, Maxwell speaks back
python -m app.cli --mode live --backend bottango_serial        # mic conversation loop
python -m app.cli --mode replay --wav clip.wav --backend mock  # drive motion from a WAV
python -m app.cli --mode typed --text "Hello!" --once          # one-shot, for scripts
```

<details>
<summary>Full CLI flag reference (<code>app/cli.py:build_arg_parser</code>)</summary>

| Flag | Values | Meaning |
|---|---|---|
| `--config` | path | YAML config (defaults to `config.yaml`, then `config.example.yaml`) |
| `--env-file` | path | `.env` to load (defaults to `./.env`) |
| `--mode` | `typed` \| `live` \| `replay` | Conversation loop type |
| `--backend` | `mock` \| `bottango` \| `bottango_serial` \| `bottango_http` | Motion transport; `bottango` honors `config.bottango.transport` (default `serial`) |
| `--serial-port` | e.g. `/dev/cu.usbmodem1101` | Override auto-detected port |
| `--playback-device` | int or name substring | Speaker output device |
| `--wav` | path | Required for `replay` mode |
| `--once` | flag | Process a single turn and exit |
| `--plot` | flag | matplotlib plot of motion channels on exit (mock backend only) |
| `--csv` | path | Write motion channels to CSV for offline inspection |
| `--safe-providers` | flag | Force offline stub STT/LLM/TTS — no network calls at all |
| `--text` | string | Speak this text and exit (typed mode) |

`app/webapp.py` adds `--host` (default `127.0.0.1`) and `--port` (default `8787`)
on top of the same parser.
</details>

### 3.3 Hosted browser mode

This is a **separate, additive** deployment — it doesn't replace or interfere with
local mode; it's a different `python -m app.<x>` entry point reading the same
`config.yaml`/`.env`.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export MAXWELL_WEB_PASSWORD_HASH="$(python -m app.web_auth hash 'hunter2')"
export SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export MAXWELL_WEB_INSECURE_COOKIE=1   # only for plain-HTTP local testing
python -m app.web_app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/login`. **`app/web_app.py` never opens a serial port
or a microphone from Python** — its only jobs are serving `static/web/`, minting
short-lived OpenAI Realtime tokens, and running the typed-turn/TTS/song-search
fallbacks server-side. Everything hardware- or mic-related happens inside
whichever browser has the page open, on whatever laptop Maxwell is plugged into.

**Docker** (`Dockerfile`) does the same thing in a container:

```bash
docker build -t maxwell .
docker run -p 8080:8080 --env-file .env maxwell
# CMD is: python -m app.web_app --host 0.0.0.0 --port 8080
```

Any Dockerfile-friendly PaaS (Fly/Render/Railway/…) works the same way, behind a
reverse proxy that terminates TLS — Web Serial and mic access both require HTTPS
(or `localhost`).

**Vercel** deploys `api/index.py` — a FastAPI rewrite of the exact same
endpoints, chosen because Vercel's Python runtime wants a WSGI/ASGI app, not a
long-running aiohttp server. `vercel.json` serves real static files first
(`"handle": "filesystem"`), then routes every other path to that one function
(`/api`) and bundles `static/**` and `config.yaml` alongside it via
`includeFiles`. Because Vercel overwrites the request URL with the function's
path, the route passes the **original path** through as a `?__path=` query
parameter and an `x-vercel-original-path` header, which `api/index.py` uses to
restore it before FastAPI routes the request. Dependencies are deliberately tiny
(`fastapi`, `httpx`, `python-dotenv`, `PyYAML` — no `numpy`/`sounddevice`/`opencv`/
etc., since none of that runs server-side in this mode) so cold starts stay under
~1s; Vercel reads them from `pyproject.toml` when it exists, so keep that in sync
with `api/requirements.txt`.

```
[ HTTPS proxy / Vercel edge ]
         |
         v
  api/index.py (FastAPI)  <-- or -->  app/web_app.py (aiohttp, same endpoints)
         |
         | mints ephemeral tokens, proxies typed-turn LLM/TTS, proxies song audio
         v
  OpenAI / Spotify / iTunes APIs

  (meanwhile, in the visitor's browser tab, entirely separately:)
  static/web/js/*  <-- WebRTC -->  OpenAI Realtime
                   <-- Web Serial -->  ESP32 (Bottango firmware) -> servos
```

Generate the password hash and session secret the same way for any hosting
target:

```bash
python -m app.web_auth hash 'whatever-password'      # -> pbkdf2_sha256$200000$...
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

---

## 4. Configuration system

`app/config.py` defines `AppConfig`, a tree of `@dataclass`es
(`ProvidersConfig`, `AudioConfig`, `MotionConfig`, `BottangoConfig`,
`RealtimeConfig`, `VisionConfig`, `LoggingConfig`) with hardcoded defaults on every
field. `load_config()`:

1. Loads `.env` (via `python-dotenv`, if present — values already in the shell
   environment are **not** overridden).
2. Reads `config.yaml` if present, else falls back to `config.example.yaml`.
3. Recursively walks the parsed YAML dict and `setattr`s onto the matching
   dataclass field (`_apply` / `_apply_dc` in `app/config.py:368-387`) — so
   **adding a new tunable is just adding a field with a default to the dataclass**,
   no separate parsing/glue code needed.

`config.example.yaml` is the annotated, source-of-truth template (copy it to
`config.yaml` and edit for your booth). Notable sections:

- **`personality`** — the system prompt. Currently: Maxwell is the Stanford TEA
  mascot at the **Treefest** club fair, pitching TEA as a **project-based** club
  (members build real things — Maxwell himself is one), warm/funny, English-only
  even if spoken to in another language, told
  explicitly *not* to write "squawk"/"polly" (TTS reads those as literal English
  words), no emojis, never admits to being an AI.
- **`providers`** — which STT/LLM/TTS implementations + model names + TTS
  voice/style instructions.
- **`audio`** — mic/speaker device selection, VAD threshold for live-mic mode.
- **`motion.jaw`** — the `JawCalibration` used by `EnvelopeFollower` (floor,
  ceiling, gain, attack/release, peak-hold).
- **`motion.behavior`** — the `BehaviorGains` used by `BehaviorEngine` (head
  drift, nod/tilt/emphasis strength, wing cooldown, idle sine periods, etc.)
- **`bottango`** — serial port, baud, per-channel pin/PWM-range/slew/invert, plus
  jaw-specific throttling knobs.
- **`realtime`** — OpenAI Realtime tuning: model/voice, VAD type/threshold/
  silence, noise reduction, half-duplex echo guard, barge-in, push-to-talk.
- **`vision`** — off by default; face tracking gaze mapping, recognition
  threshold/margin/votes, greeting cooldown, scene-understanding model/prompt.

**What's live-tunable vs. needs a restart:** the local operator UI (`/api/tuning`,
`/api/realtime/config` in `app/webapp.py`) can change jaw gain/invert, motion
intensity, TTS voice/instructions/personality, and most Realtime VAD/echo-guard
knobs **without reconnecting**. Changing PWM ranges or slew rate calls
`state.recreate()`, which tears down and rebuilds the whole pipeline (closes and
reopens the serial port) — that one does interrupt whatever's in flight.

---

## 5. End-to-end walkthrough: what happens when you talk to Maxwell

### 5.1 Local operator mode, step by step

1. **Boot** (`app/webapp.py:main` → `AppState.start` →
   `AppState._create_pipeline`): builds STT/LLM/TTS providers
   (`_build_providers`, falling back to offline stubs with a logged warning if any
   provider fails to construct — e.g. missing `openai` package or API key), builds
   the `BottangoSerialBackend` from `config.bottango.serial` (`_build_backend`),
   and constructs a `ConversationPipeline` with all of it plus the jaw
   calibration, behavior gains, personality, and `vision_config`.
2. **`pipeline.__aenter__()`** (`app/pipeline.py:154`): creates a `BehaviorEngine`,
   a shared `GazeContext` (always present, zero-confidence until vision starts
   writing to it), and a `MotionScheduler` — then `scheduler.start()` calls
   `backend.connect()`, which does the full Bottango handshake (§7.1) and spins up
   the writer/reader/recovery background tasks. If `config.vision.enabled` is
   true, face tracking starts here too.
3. `state.wake_sweep()` runs once at boot — sweeps every servo through
   min→max→mid so the operator can visually confirm nothing's dead before the
   first utterance.
4. **The `MotionScheduler`'s 30 Hz loop is now running independently of
   everything else**, for the entire lifetime of the process (§8 covers exactly
   what it does each tick). This is the key architectural fact: motion doesn't
   "turn on" when someone talks — it's always ticking, producing idle-fidget
   frames when nobody's talking and speaking-driven frames when someone is.
5. **A visitor talks to Maxwell** (Realtime mode, the default): the operator or
   admits page called `POST /api/realtime/start`, which reads
   `OPENAI_API_KEY` from the process environment and calls
   `pipeline.start_realtime()` (§6.4). This opens a `RealtimeSession` — a
   WebSocket to OpenAI's `gpt-realtime` model — and starts streaming mic audio up
   and assistant audio down, in the same Python process.
6. As assistant audio streams back, `RealtimeSession`'s playback loop feeds each
   20ms window's RMS into a `LiveSpeakingContext`/`EnvelopeFollower`, which the
   `MotionScheduler` reads every tick via its `speaking_context_provider` callback
   — this is how the jaw stays locked to what's actually coming out of the
   speaker, not to the text.
7. State transitions (`listening` → `thinking` → `speaking` → back to
   `listening`) come from Realtime server events (`speech_started`,
   `response.created`, audio deltas, `response.done`) via a `state_callback` that
   drives the same `ConversationStateMachine` the typed/live paths use — so the
   `BehaviorEngine` never needs to know which conversation mode produced the
   state.
8. If vision is enabled, a `FaceTracker` task is *also* running independently at
   ~12 Hz, writing into the shared `GazeContext` — the `BehaviorEngine` blends
   that into head motion every tick, weighted by tracking confidence.
9. **"End session"** (`POST /api/disconnect`): stops the Realtime session, sends
   one final centered `MotionFrame` (jaw closed, head/wing neutral), tears down
   the pipeline (which closes the serial port). Safe to unplug Maxwell at this
   point.

### 5.2 Hosted browser mode, step by step

1. Visitor opens the URL, logs in with the booth password → gets a signed session
   cookie (§12.2).
2. The page's JS module (`app.js`/`admits.js`/`sing.js`) fetches
   `GET /api/web/motion-config` once — the **only** trip to the server needed to
   get the browser's motion engine tuned to match `config.yaml` (pins, PWM
   ranges, behavior gains, jaw calibration). Everything from here on that touches
   motion or audio happens client-side.
3. **"Connect Maxwell"** (must be a real click — Web Serial requires a user
   gesture): `navigator.serial.requestPort()` → `WebSerialTransport.connect()`
   runs the identical Bottango handshake/registration sequence as the Python
   backend, just implemented in `serial.js`/`bottango.js` instead.
4. **"Start realtime"**: `POST /api/web/realtime/session` — the server calls
   OpenAI's `/v1/realtime/client_secrets` endpoint with the real API key and
   returns only a short-lived (~60s) `ek_...` ephemeral token. The browser never
   sees the real key.
5. The browser opens an `RTCPeerConnection`, attaches the mic track, creates a
   data channel, does an SDP offer/answer exchange **directly against
   `api.openai.com`** using the ephemeral token — from this point, audio flows
   browser ↔ OpenAI with the Maxwell server completely out of the loop.
6. Remote assistant audio plays through an `<audio>` element and is tapped by a
   `AnalyserNode` for RMS → the exact same `EnvelopeFollower`/`BehaviorEngine`
   chain (ported line-for-line into JS) → `WebSerialTransport.sendFrame()` → the
   same `sCI` commands over the same USB cable.
7. If the browser lacks Web Serial/WebRTC (Safari/Firefox) or the operator picked
   "Typed only", `POST /api/web/typed` runs the LLM+TTS turn **server-side**
   (still with the server's own key) and returns base64 MP3 for the browser to
   decode, play, and drive the envelope follower from locally.

---

## 6. The conversation & AI layer

### 6.1 Personality / system prompt

Single source of truth: the `personality:` block in `config.yaml` (see §4).
`app/personality.py:load_personality()` is what the two web builds
(`app/web_app.py`, `api/index.py`) use to read it directly out of `config.yaml`
(preferring `realtime.instructions` if that's non-empty, else falling back to
`personality`) so all four entry points — CLI, local webapp, hosted webapp,
Vercel — say the same things about Stanford TEA. If `config.yaml` is somehow
missing (e.g. a bundling mistake), there's a baked-in `FALLBACK_PERSONALITY`
matching the same flavor (also Treefest / project-based).

**Event context (hosted pages):** `static/web/js/admits.js` has built-in event
presets whose text is appended to the personality server-side
(`compose_instructions`): **Treefest Club Fair** (the default), Admit Weekend
Fair, NSO, and Bay Area Breadbowl, plus operator-saved custom prompts. The
selected preset persists in the browser's `localStorage`, so a browser that
previously picked another event keeps it until the operator switches the
dropdown. `tests/test_personality.py` guards that the config prompt (with
"Treefest" and "project-based") is what every build actually loads.

### 6.2 Three conversation paths

| Path | Where | STT | LLM | TTS | Latency | Used by |
|---|---|---|---|---|---|---|
| **Typed** | `pipeline.handle_typed_turn` | — (text input) | Chat Completions | `audio.speech` | Normal | CLI `--mode typed`, webapp "Speak" box, hosted `/api/web/typed` |
| **Live mic (turn-based)** | `pipeline.handle_live_turn` | Whisper (`whisper-1`) | Chat Completions | `audio.speech` | One record→transcribe→reply→speak round-trip per turn | CLI `--mode live`, webapp "Talk to Maxwell" button |
| **Realtime (speech-to-speech)** | `pipeline.start_realtime` / browser `RealtimeSession` | built into the Realtime model | built into the Realtime model | built into the Realtime model | Lowest — one continuous streaming session | Default UX on `/admits` and the browser pages |

### 6.3 Providers (`conversation/{stt,llm,tts}.py`)

Each is an ABC with a `build_x_provider(name, ...)` factory so config strings map
to implementations:

- **STT**: `OpenAIWhisperSTT` (`whisper-1` by default) or `LocalStubSTT` (raises
  loudly — forces you to switch to typed mode or configure a real provider).
- **LLM**: `OpenAILLM` (`gpt-4o-mini` by default, temperature 0.8, 160 max output
  tokens) or `StubLLM` (canned "squawk!" lines, fully offline, used by
  `--safe-providers` and as an automatic fallback if the real provider fails to
  construct).
- **TTS**: `OpenAITTS` (`gpt-4o-mini-tts`, configurable `voice` + free-form
  `instructions` style hint — both live-mutable from the operator UI without
  reconnecting), `MacOSSayTTS` (shells out to `say`, fully offline), or
  `SineStubTTS` (a modulated tone, for CI/headless testing — still produces a
  real envelope so the motion pipeline is exercisable without any TTS backend).

`app/cli.py:_build_providers` wraps each construction in a `try/except` that logs
a warning and falls back to the stub — so a missing package or unset API key
degrades gracefully instead of crashing the whole app.

### 6.4 The Realtime session — Python side (`conversation/realtime.py`)

This is the most intricate module in the codebase. `RealtimeSession` wraps
`AsyncOpenAI(...).realtime.connect(model="gpt-realtime")` (the **GA** Realtime
API — the module has explicit comments about the pre-GA `/v1/realtime/sessions`
shape being deprecated and event names moving to `response.output_audio*`).
`start()`/`stop()` are idempotent — calling `start()` while already running tears
down and replaces the session; the webapp leans on this so double-clicking the
toggle button can't wedge anything.

On start, it sends one `session.update` configuring:

- **Audio format**: PCM16 at 24kHz both directions (`REALTIME_SAMPLE_RATE`). If a
  device (common on some macOS mic/AirPods configs) rejects 24kHz directly via
  PortAudio, it falls back to the device's native rate and resamples on the fly
  (`_resample_int16_bytes` / `_resample_float`, plain linear interpolation — good
  enough for speech, avoids a `scipy` dependency).
- **Turn detection**: `server_vad` (threshold/prefix-padding/silence-duration from
  `config.realtime.*`) or `semantic_vad` (content-aware, ignores the threshold
  sliders), or `null` when push-to-talk is on.
- **Server-side transcription** (`whisper-1`) so both sides of the conversation
  can be shown in the UI's transcript log.
- **Noise reduction**: `off` / `near_field` / `far_field` — server-side, set to
  `far_field` by default for a laptop mic in a noisy event.
- **Tools**: `look_and_describe` only if `vision.scene_enabled`, and
  `remember_person`/`forget_person` only if `vision.recognition_enabled` — with
  both off (the current booth setup) the session gets no tools at all. See §6.6.

Three background `asyncio` tasks run for the life of the session:

- **`_mic_pump_loop`**: drains a queue fed by the `sounddevice` mic callback and
  forwards PCM16 chunks to the server as `input_audio_buffer.append` events.
- **`_event_reader_loop`**: the state machine of the whole session — classifies
  every incoming server event and reacts (buffers/plays audio deltas, flips
  `ConversationStateMachine` state, dispatches tool calls, reorders and emits
  transcript lines).
- **`_playback_loop`**: drains decoded audio out to the speaker in 20ms windows,
  computing RMS per window and calling the `envelope_callback` — this is the
  audio-out → jaw-motion link.

Two behaviors worth calling out because they solve real problems at a loud booth:

- **Half-duplex echo guard** (`half_duplex: true` by default): OpenAI's Realtime
  API has no built-in echo cancellation. With Maxwell's speaker near the laptop
  mic, his own voice loops back in and he starts talking to himself. The fix:
  while assistant audio is playing, the mic pump **drops** every chunk instead of
  forwarding it, plus a configurable `playback_tail_ms` after the last audio
  delta to swallow speaker reverb before re-opening the mic.
- **Smart barge-in** (`barge_in_enabled: true`): while muted by the echo guard,
  it still computes each dropped chunk's RMS and compares it against an
  exponentially-decaying ambient floor. If the user is clearly louder than
  speaker bleed-back for `barge_in_min_frames` consecutive 40ms chunks (default 4
  = 160ms), it un-mutes, cancels the in-flight response
  (`response.cancel`), and clears the input buffer — so a visitor really can
  interrupt Maxwell mid-sentence even with the echo guard on.
- **Push-to-talk**: when on, server VAD is disabled entirely (`turn_detection:
  null`); the mic only streams while `ptt_down()`...`ptt_up()` is held, and
  releasing explicitly commits the buffer + requests a response. Toggling PTT
  live reconfigures `turn_detection` via another `session.update` without
  dropping the WebSocket.

### 6.5 The Realtime session — browser side (`static/web/js/realtime.js`)

Functionally the same session, over WebRTC instead of a WebSocket, because a
browser tab can't hold a raw TCP/WS Realtime connection with the SDK the Python
side uses — WebRTC is OpenAI's browser-native transport. Key differences from the
Python version:

- Audio transport is a native `RTCPeerConnection` audio track — no manual
  PCM16 chunking; the browser and OpenAI's servers negotiate codecs directly.
- A `RTCDataChannel` (`"oai-events"`) carries the same JSON event stream Python
  reads off its WebSocket (`session.update`, `response.cancel`,
  `input_audio_buffer.*`, transcript deltas, etc.) — the event *names* are kept
  in sync with the Python side (including tolerating both pre-GA and GA event
  name variants, e.g. `response.audio.delta` vs `response.output_audio.delta`).
- Barge-in on push-to-talk-down does three things simultaneously: cancels the
  server response, clears the server's output audio buffer, **and** mutes the
  local `<audio>` element — because WebRTC keeps its own jitter buffer of
  already-delivered audio that a server-side cancel alone won't stop from
  playing.
- A **silence watchdog** exists because `output_audio_buffer.stopped` isn't
  reliably emitted on WebRTC peers: if the smoothed jaw envelope has been quiet
  for >1.1s and the SPEAKING state has lasted >1.5s, it force-transitions back to
  LISTENING so Maxwell never gets stuck "speaking" silently forever.
- The AudioContext is deliberately created **during the click handler**, before
  any `await`, because creating it after the gesture window closes leaves it
  `"suspended"` in Chrome — which was previously causing "he just breathes for
  the first 200ms" symptoms.

### 6.6 Vision-aware tools (Realtime function calling)

`ConversationPipeline._build_realtime_tools` (`app/pipeline.py`) hands the
Realtime session an OpenAI function-calling schema built from the vision
switches — each tool is only offered when its feature is on:

- **`look_and_describe`** — offered only when `vision.scene_enabled: true`
  (it's the one paid vision feature). The model
  calls it whenever a visitor asks "what do you see?" or shows Maxwell
  something; the handler calls `pipeline.describe_scene()` (§9.3) and feeds the
  text result back as the tool's output, and Maxwell speaks a reaction to it.
- **`remember_person(name)`** / **`forget_person(name)`** — offered only when
  `recognition_enabled`. `remember_person` flushes the currently-buffered
  "unknown person" face embeddings into `FaceMemory` under the given name;
  `forget_person` deletes a stored identity.

`_handle_realtime_tool` dispatches the call, and `_handling_tool_call` is set
while it's in flight so the function-call turn's `response.done` doesn't
prematurely flip the state machine back to `listening` before the model's
spoken follow-up actually starts.

---

## 7. Getting bits to the servos (hardware transport)

### 7.1 The Bottango wire protocol (`transport/bottango_protocol.py`, ported to
`static/web/js/bottango.js`)

Every command is a plain ASCII line, terminated `\n`, with a checksum suffix:
`<body>,h<sum>\n` where `sum` is the sum of the ASCII codes of every character in
`<body>`. The firmware rejects anything whose hash doesn't match.

| Command | Meaning |
|---|---|
| `hRQ,<random>` | Handshake request (we send this to wake the firmware up) |
| `btngoHSK,<version>,<random>,<accepting>` | Firmware's handshake reply |
| `tSYN,<ms>` | Sync the firmware's clock to ours |
| `xE` | Deregister all effectors |
| `xC` | Clear all curves |
| `rSVPin,<pin>,<minPwm>,<maxPwm>,<maxPwmPerSec>,<startingPwm>` | Register a pin-controlled servo — **this is where the min/max PWM safety rails live, enforced by the firmware regardless of what we send** |
| `sCI,<pin>,<compressed 0-8192>` | Instant curve: set a channel's target position now |
| `STOP` | Halt |
| `OK` | Firmware's per-command ack |

Connect sequence (identical in Python and JS): open serial @115200 → wait for
boot → send `hRQ` (retried until a `btngoHSK` reply lands) → `tSYN` → `xE` → `xC`
→ `rSVPin` × 4 (jaw/head_lr/head_ud/wing).

### 7.2 `BottangoSerialBackend` (`transport/bottango_serial_backend.py`) — the
Python hardware path

The one actually used at the booth (`--backend bottango` → resolves to
`bottango_serial` via `config.bottango.transport`). Beyond the protocol basics,
it implements several booth-tested reliability features:

- **Fire-and-forget motion writes**: during normal operation it does **not**
  wait for the firmware's `OK` per `sCI` command — waiting would cap throughput
  to ~20 cmd/s, far below the 120 cmd/s a 30 Hz × 4-channel scheduler generates.
  Registration commands (which must be correct) still use `_send_and_wait_ok`.
- **Per-channel coalescing + delta thresholds**: `send_frame` only transmits a
  channel if it moved more than `min_delta_for_send` (default 0.4%) since last
  sent. Jaw gets its own, larger threshold (`jaw_min_delta`, default 2%) because
  its long extension wire turns a flood of sub-percent updates into unreadable
  noise on the line.
- **Jaw-specific rate limiting with peak preservation**: jaw serial writes are
  additionally capped to `jaw_min_send_interval_s` (default 0.08s ≈ 12 Hz). While
  throttled, it tracks a running **max** of the envelope and sends *that* peak
  when the window reopens — not whatever the value happens to be at that instant
  — so brief loud syllables between rate-limited sends aren't lost.
- **Auto-recovery**: the ESP32 can brown out and reboot mid-session from servo
  inrush current. The reader loop watches for an unsolicited `BOOT` or repeated
  `errNoServoOnPin` errors and sets a flag; a dedicated recovery task then redoes
  the handshake and re-registers every servo automatically, with no user action.
- **`jaw_hammer()` / `full_reset()` / `wake_sweep()`**: operator-triggered
  recovery actions surfaced as buttons in the UI (§12.1) — see §14 for what
  they're actually for.
- **Auto-detect** (`_auto_detect_port`): scores every available serial port by
  how ESP32-ish its device path/description/manufacturer looks (`usbmodem`,
  `Silicon Labs`/`CP210`, `esp` in description/manufacturer, `WCH`, `FTDI`) and
  picks the best match if no port is configured explicitly.

### 7.3 `WebSerialTransport` (`static/web/js/serial.js`) — the browser hardware
path

Deliberately faithful port of the above onto the Web Serial API: same channel
defaults, same handshake/registration sequence, same coalesce-changed-channels
approach (compressed-value delta ≥5 to send). A few browser-specific pieces:

- **`tryAutoConnect()`**: silently reopens a port the browser already remembers
  being authorized for this origin (Chrome persists that grant) — so after the
  operator picks the USB device once, every subsequent page load/tab on the same
  laptop connects with zero prompts. `connect()` falls back to
  `navigator.serial.requestPort()` (which **must** be a user gesture) only if
  there's no remembered port.
- Clears DTR/RTS after opening, because Chrome's Web Serial asserts both on open,
  which triggers the ESP32's auto-reset circuit — then waits 2.5s for Arduino-style
  `setup()` to finish before starting the handshake.
- **Manual pose mode** (`setManualMode`): deregisters every effector so the
  firmware stops driving PWM, letting someone physically pose Maxwell by hand;
  re-registering resumes normal driven motion.

### 7.4 Other backends

- **`MockBackend`** / **`MockTransport`** — no hardware at all. Logs motion at a
  throttled rate, optionally writes a CSV trace or renders a matplotlib plot on
  exit. Used by `--backend mock`, the webapp's "Mock (no hardware)" checkbox, and
  every test that touches motion.
- **`BottangoBackend`** (`transport/bottango_backend.py`) — legacy HTTP path
  talking to Bottango Desktop's `setInputValue/{id}/{value}` API. Kept for
  completeness; the README notes most current Bottango builds only expose their
  live API over WebSocket, so this path "rarely works out of the box" — serial is
  the supported path.

### 7.5 Servo map (defaults; see `config.yaml` → `bottango.serial`)

| Channel | Pin | PWM range (µs) | Slew (µs/s) | Notes |
|---|---|---|---|---|
| Jaw (mouth) | 9 | 1450–1775 | 1200 | Inverted; the flaky one — extension-wire GPIO issues, see §14 |
| Head left-right | 5 | 1275–1725 | 1800 | |
| Head up-down | 6 | 850–2100 | 1800 | |
| Wing | 3 | 1500–2000 | 3000 | |

---

## 8. Maxwell's movements (the motion pipeline)

This is the pipeline both deployment modes implement (Python natively,
JavaScript as a parity-tested port):

```
mic → STT/Realtime → LLM → TTS/Realtime audio
                              |
                       EnvelopeFollower (RMS -> jaw target)
                              |
                  ConversationStateMachine (idle/listen/think/speak)
                              |
                  BehaviorEngine (heuristic head/wing motion, blended with gaze)
                              |
                  MotionScheduler (30 Hz fixed-rate tick)
                              |
                  BottangoSerialBackend / WebSerialTransport
                              |
                          ESP32 -> servos
```

### 8.1 Data model (`motion/models.py`)

- **`MotionFrame`** — the only thing that ever reaches a transport: four
  normalized floats in `[0,1]` (`jaw_open`, `head_lr`, `head_ud`, `wing`; head
  channels centered at 0.5) plus a timestamp.
- **`SpeakingContext`** — envelope (loudness), utterance text, playback
  progress, a phrase-boundary flag, an emphasis spike value, and
  question/excited flags — everything the behavior engine needs while SPEAKING.
- **`GazeContext`** — `target_lr`/`target_ud` (normalized head aim) and
  `confidence` (decays to 0 after `lost_face_timeout_s` with no detection), the
  vision analog of `SpeakingContext`.
- **`JawCalibration`** and **`BehaviorGains`** — the tunable dataclasses backing
  `config.yaml`'s `motion.jaw` / `motion.behavior` (documented inline in §4).

### 8.2 `EnvelopeFollower` (`motion/envelope.py`, ported to `envelope.js`)

Takes a per-frame RMS value and produces a smoothed, calibrated jaw position:

1. Below `noise_floor` → treated as silence (0).
2. Multiply by `gain`, clamp to 1.0 → the *target*.
3. Move the smoothed value toward the target using **attack** coefficient if
   rising, **release** if falling (attack is intentionally higher than release —
   this makes the jaw snap open on syllable onsets but glide closed between
   them, which reads as far more natural than symmetric smoothing).
4. Hold the current peak for `peak_hold_ms` so brief loud spikes stay visible
   even if a rate-limited send would otherwise miss them.
5. Map into `[floor, ceiling]` — this final range mapping means "silence" isn't
   necessarily `jaw=0` and "loud" isn't necessarily `jaw=1`; those are separately
   tunable so the physical mouth opening matches what looks right on *this*
   servo.

`app/pipeline.py`'s `LiveSpeakingContext` also runs a **second**, independent
envelope (`_behavior_smoothed`, fixed `rms * 6.0` gain) purely for driving wing
flaps/head bobs — deliberately decoupled from the jaw's calibration, because the
jaw is tuned to a low gain (1.6) to match the physical servo, and reusing that
low gain for behavior thresholds would leave Maxwell almost motionless while
talking.

### 8.3 `ConversationStateMachine` (`motion/state_machine.py`)

About as simple as it looks: four states (`IDLE`, `LISTENING`, `THINKING`,
`SPEAKING`), async-safe transitions, a listener-registration hook. Deliberately
*driven* by the pipeline/session code rather than inferred from events — keeps
motion behavior predictable and easy to debug, and means every conversation path
(typed, live-mic, Python Realtime, browser Realtime) can drive the exact same
state machine without knowing about each other.

### 8.4 `BehaviorEngine` (`motion/behavior_engine.py`, ported to `behavior.js`,
verified for behavioral parity by `tests/test_js_behavior_parity.py` running the
real JS through Node)

Called once per scheduler tick with the current state, elapsed time, and
(if SPEAKING) a `SpeakingContext`, plus a `GazeContext` if vision is running.
Per state:

- **IDLE / LISTENING / THINKING** — treated almost identically on purpose (per
  the code comments, "no flat holds, no abrupt starts"): a continuous
  raised-cosine wing flap-cycle (`0.5·(1−cos(2π·t/period))·strength` — eases
  through zero, no snap) layered with two independent slow head sines (a nod on
  `head_ud`, a tilt on `head_lr`) on deliberately non-harmonic periods, so the
  combined motion never repeats exactly and Maxwell never looks frozen while
  waiting. Head-drift targets snap back toward center quickly on entering any of
  these states (0.35s time constant) so SPEAKING always starts from a neutral
  pose.
- **SPEAKING** — the interesting one:
  - **Phrase-boundary nod**: on each `phrase_boundary` flag, arms a 0.4s
    half-sine dip on `head_ud`. If the phrase looks like a question, ~60% of the
    time also arms a 0.7s tilt (random left/right) on `head_lr`.
  - **Emphasis bump**: when `context.emphasis > 0.45`, adds a proportional bump
    to jaw openness and a smaller counter-bump to `head_ud`.
  - **Continuous envelope-driven head bob**: louder syllables tip the head up
    slightly, tracked continuously (not just on phrase starts) — described in
    the code as "what makes the bird look like it's actually following its own
    speech instead of just drifting."
  - **Wing flaps**: eligible whenever `envelope > 0.45` and the
    `wing_cooldown_s` (default 2s) has elapsed since the last flap; then a
    20% base chance per eligible tick, +35% if the utterance was flagged
    "excited" (exclamation mark or a word like "wow"/"amazing"/"hello" — see
    `analyze_text` in the same file).
  - **Random yaw/pitch drift**: while speaking, picks new drift targets at a
    tunable probability-per-second and eases toward them — the head "wanders"
    within `head_lr_drift`/`head_ud_drift` bounds instead of holding perfectly
    still.
- **Gaze blending** (`_apply_gaze`) — runs after the procedural drift update and
  re-bases the yaw/pitch drift target toward `GazeContext.target_lr/ud`, weighted
  by tracking confidence, *before* all the above nod/tilt/bob offsets are added
  on top — so Maxwell looks at a tracked face while still nodding/bobbing
  expressively on top of that base orientation. `gaze_idle_suppression`
  (default 0.85) fades out most of the idle wander when confidently locked onto
  a face, so he holds a steady attentive look instead of drifting away from the
  person mid-lock.
- **Output-side lowpass** (`head_smoothing_tau_s`, default 0.18s Python /
  0.08s JS default): several of the above terms can step instantly between
  ticks (the emphasis bump in particular snaps on/off at its threshold), which
  reads as a physical twitch on a real servo. A final exponential lowpass on the
  summed `head_lr`/`head_ud` output smooths that into eased motion without
  touching any of the upstream heuristics.

Randomness is seeded (`BehaviorGains.seed`) so the Python side can be made fully
deterministic for tests; the JS port uses a seeded `mulberry32` generator for the
same contract (not bit-identical to Python's Mersenne Twister, but reproducible
per-seed).

### 8.5 `MotionScheduler` (`motion/scheduler.py`, ported to `motion.js`)

The heartbeat of the whole visual performance: one `asyncio` task (or
`setTimeout` loop in JS) ticking at a fixed rate (30 Hz default), forever, for
the life of the connection:

```python
context = speaking_context_provider(now) if state == SPEAKING else None
gaze = gaze_provider(now)
output = behavior.tick(state=state_machine.state, now=now, speaking=context, gaze=gaze)
frame = output.to_frame(timestamp=now)
await backend.send_frame(frame)
```

The JS version adds one thing the Python side doesn't need: a
`visibilitychange` guard. Browsers throttle `setTimeout` to ~1 Hz on a hidden
tab; without a guard, the behavior engine would integrate ~30 frames of motion
in one delayed wakeup and the servos would visibly jump in big steps once a
second. Instead it fully pauses the scheduler while hidden, sends one neutral
center frame, and resets the engine's internal `_last_tick` on resume so the
first frame back doesn't try to integrate the entire hidden duration in one
step.

---

## 9. Maxwell's vision

*(See the snapshot note at the top — this subsystem is present and wired up on
disk but not yet committed to git. This section describes the full Python
subsystem, which runs in local operator mode only. The hosted browser build
(`index.html`) now also has **client-side face tracking** — a from-scratch
port, not this Python code: `static/web/js/vision.js` uses the browser's
native `FaceDetector` to maintain a gaze context that `js/behavior.js`'s
`_apply_gaze` blends into the head channels, mirroring `_apply_gaze` here.
Only face tracking is ported; recognition and scene understanding remain
Python/local-only.)*

Three independent features, each with its own switch in `config.yaml`'s
`vision:` block (restart the app after changing them):

| Switch | Feature | Cost | Default | Current booth |
|---|---|---|---|---|
| `enabled` | Face tracking — head follows the nearest face (§9.1) | free, local | off | **on** |
| `recognition_enabled` | Face memory — remember people by name (§9.2) | free, local | off | **off** |
| `scene_enabled` | "What do you see?" image analysis (§9.3) | **paid** OpenAI call | off | **off** |

Face tracking can also be started/stopped live from the operator page's Vision
card; the card greys out the "What do you see?" button and hides the known-faces
panel when those features are switched off (`vision_status()` reports
`scene_enabled` / `recognition_enabled`).

### 9.1 Face tracking (free, local, always-on when enabled)

`vision/face_tracker.py`'s `FaceTracker` is architecturally the vision analog of
`MotionScheduler` — one task ticking at `tracking_fps` (default 12 Hz), forever:
grab a camera frame (`vision/camera.py`, OpenCV `VideoCapture`; a `MockCameraSource`
with a moving dot exists for tests/no-webcam runs), detect faces
(`vision/face_detector.py` — MediaPipe if its legacy `solutions` API is available,
else an OpenCV Haar cascade fallback that needs no model download and works on
every Python including 3.13 where MediaPipe's old API is gone), pick the
**primary** face (`select_primary`: largest/nearest, with hysteresis so two
similarly-sized faces don't make the head flicker between them), map its
position to a normalized head target (`map_face_to_gaze`: offset from center ×
configurable gain per axis, with per-axis invert flags because "which way is
correct" depends on webcam mirroring + servo mounting and can't be known ahead of
time — `tools/vision_preview.py` exists specifically to calibrate this), and
write the smoothed result into the shared `GazeContext` the `BehaviorEngine`
reads every motion tick (§8.4).

### 9.2 Face recognition & memory (opt-in, needs InsightFace)

`insightface` and `onnxruntime` are **commented out** in `requirements.txt` so a
fresh install isn't blocked by them; install them by hand
(`pip install "insightface>=0.7" "onnxruntime>=1.17"`) before setting
`recognition_enabled: true`. Without them, enabling recognition just logs a
warning and the app carries on with tracking only.

A second, independent, slower (`recognition_fps`, default 3 Hz) task
(`vision/recognition.py`'s `RecognitionTracker`) riding on the *same* frame the
tracker already grabbed (no second camera reader):

1. `vision/face_recognizer.py` runs InsightFace's `buffalo_l` bundle (SCRFD
   detector + ArcFace recognizer, CPU-only, ONNX Runtime) to get a 512-d
   L2-normalized embedding — deliberately doing its **own** detection/alignment
   rather than reusing the tracker's box, because skipping ArcFace's expected
   5-point alignment quietly tanks match accuracy.
2. `vision/face_memory.py`'s `FaceMemory.identify()` compares against every
   stored person's embedding set by **max** similarity (not a centroid — a
   centroid blurs genuinely different lighting/angle views together) and
   rejects the match (returns "unknown") if the top score is below
   `recognition_threshold` **or** the top two candidates are within
   `recognition_margin` of each other — biased deliberately toward "I don't
   know" over a confidently wrong name.
3. **Temporal voting**: a single frame's match/no-match is just one vote in a
   rolling window; an identity is only *committed* once it has a plurality of
   `recognition_votes` (default 8) — so a single bad frame can't make Maxwell
   blurt the wrong name.
4. While the committed identity is "unknown," sharp/confident embeddings
   (gated on Laplacian-variance sharpness, to skip motion-blurred crops from the
   moving head) are buffered into a pending set; the `remember_person` Realtime
   tool flushes that buffer into `FaceMemory` under the name the visitor gave.
5. **Persistence** is off by default (session-only, nothing biometric written to
   disk) — enabling `memory_persist` writes plain JSON (no pickle) to
   `data/face_memory.json`, which is `.gitignore`'d.
6. **Greeting**: on a committed-identity *change*,
   `pipeline._on_identity_change` pushes a "Current view: you're looking at
   X / someone new" line into the live Realtime session's instructions
   (`RealtimeSession.set_context_line`) and — gated on the session actually being
   idle, so it never talks over an in-progress turn, and on a per-identity
   **last-seen cooldown** (default 10 minutes) so detection flicker doesn't
   re-greet someone mid-conversation — proactively triggers Maxwell to speak
   first via `trigger_greeting()`.

### 9.3 Scene understanding (on-demand, paid — the only vision feature with a
per-call API cost)

`vision/scene.py`'s `OpenAIVisionProvider` JPEG-encodes a single frame (reusing
the tracker's most recent frame if it's running, or grabbing one directly
otherwise) and sends it inline to a vision-capable chat model (`gpt-4o-mini` by
default) with a short prompt, capped at `scene_max_tokens`. Throttled by
`scene_min_interval_s` (default 3s) — a too-soon re-trigger returns a canned
in-character deferral line instead of hitting the API again. Reachable from the
operator UI's "What do you see?" button or, in Realtime mode, the
`look_and_describe` tool (§6.6).

**Gated by `vision.scene_enabled`** (default `false`): when off, the tool isn't
offered to the model, `describe_scene()` returns a canned line without touching
the API, `POST /api/vision/describe` returns an error, and the operator button is
disabled — so there is no way to spend money on image analysis by accident.

---

## 10. The website

### 10.1 Local operator mode (`app/webapp.py`) — everything in one aiohttp
process

No separate template/static files for this mode — `INDEX_HTML` and
`ADMITS_HTML` are large literal strings embedded directly in `webapp.py`, each
with its own inline `<style>`/`<script>`. Two pages:

- **`GET /`** — the **operator view**: connect/disconnect, a text box + Speak
  button, live/mic conversation controls, Realtime start/stop + every VAD/echo/
  barge-in slider, jaw tuning sliders, vision start/stop + "what do you see" +
  known-faces list, and a raw scrolling log pane (polled from `/api/log`, which
  reads an in-memory `_LogBuffer` attached as a `logging.Handler`).
- **`GET /admits`** — the **guest view**: no tuning knobs, just a big round talk
  button (styled like Maxwell — blue/purple), a Realtime-vs-turn-based /
  auto-listen-vs-push-to-talk picker, and a chat transcript.

<details>
<summary>Full local-mode JSON API (all under <code>/api/</code>, defined in <code>build_app</code>)</summary>

| Route | Purpose |
|---|---|
| `GET /api/config` | Current jaw/voice/personality/realtime config for the UI to populate |
| `GET /api/log` | Last 200 buffered log lines |
| `POST /api/speak` | Say arbitrary text (typed path) |
| `POST /api/replay` | Replay a test WAV clip through the motion pipeline |
| `POST /api/center`, `/api/stop` | Center servos / panic-stop |
| `POST /api/tuning` | Live jaw/voice/personality/intensity changes; `reconnect: true` rebuilds the pipeline for PWM/slew changes |
| `POST /api/reregister`, `/api/test-jaw`, `/api/pin-swap-test`, `/api/wake`, `/api/full-reset` | Hardware diagnostics — see §14 |
| `POST /api/converse` | One listen→transcribe→reply→speak turn (live-mic mode) |
| `POST /api/realtime/start`, `/stop` · `GET /status` | Realtime session control |
| `POST /api/realtime/config` | Live VAD/echo/barge-in/voice changes; restarts the session only if it's already running |
| `POST /api/realtime/ptt` | Push-to-talk down/up signal |
| `GET /api/realtime/transcripts?since=<id>` | Poll-based transcript feed (monotonic id cursor) |
| `POST /api/vision/start`, `/stop` · `GET /status` | Face tracking (+ recognition) control |
| `POST /api/vision/describe` | On-demand scene description (optionally spoken) |
| `POST /api/vision/forget` | Delete one remembered person or everyone |
| `POST /api/connect`, `/disconnect` · `GET /api/connection/status` | (Re)build the pipeline / open-close the serial port |

</details>

### 10.2 Hosted browser mode — pages, routes, auth

Static pages served from `static/web/`: `login.html`, `index.html` (operator —
includes a client-side **Vision** panel for `FaceDetector`-based face tracking,
see §9's snapshot note), `admits.html` (guest), `sing.html` (jukebox), plus
`relic.html` at `/relic` — an
unrelated Web Serial lighting/magnet control panel for the "Relic" artifact prop
(`js/relic.js`, `css/relic.css`). `app/web_app.py` (aiohttp) and
`api/index.py` (FastAPI/Vercel) implement **identical** routes and share the
exact same auth primitives from `app/auth_core.py` — the intent being that
password hashing, session signing, and rate limiting are byte-for-byte the same
regardless of which HTTP framework served the request.

**Auth** (`app/auth_core.py`):

- Passwords hashed with **PBKDF2-SHA256, 200k iterations**
  (`pbkdf2_sha256$<iters>$<salt_b64>$<hash_b64>`, Django-compatible encoding), or
  a plaintext dev-only fallback (`MAXWELL_WEB_PASSWORD`) compared with
  `hmac.compare_digest`.
- Sessions are a base64url JSON payload (`uid`, `iat`, `exp`) + an
  HMAC-SHA256 signature over it, stored in an `mxw_session` cookie
  (HttpOnly, Secure by default, SameSite=Lax, 12h TTL). If `SESSION_SECRET`
  isn't set, a random one is generated per-process — meaning a restart
  invalidates all sessions (logged as a warning).
- If **no password is configured at all**, `AuthConfig.from_env` refuses to be
  silently open: it generates a random one-session password and logs it,
  rather than defaulting to unauthenticated.
- `LoginLimiter` — in-memory per-IP sliding window, 8 attempts/15 minutes, then a
  15-minute lockout. Resets on every serverless cold start (acceptable, per the
  code comments — cookies are still cryptographically signed regardless).
- State-changing API requests are Origin-checked against `MAXWELL_ALLOWED_ORIGIN`
  (or the request's own scheme+host if unset) as belt-and-suspenders CSRF
  defense on top of `SameSite=Lax`.
- Login failures are deliberately uniform (`invalid_credentials` for everything)
  so a bad guess can't distinguish "wrong password" from any other failure mode.

<details>
<summary>Full hosted-mode JSON API</summary>

| Route | Purpose |
|---|---|
| `GET /healthz` | Uptime probe |
| `GET /login`, `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me` | Auth |
| `GET /api/web/config` | Non-sensitive defaults (`has_openai_key`, realtime voice/model) |
| `GET /api/web/motion-config` | Pins/PWM/gains/jaw-cal from `config.yaml`, for the browser motion engine |
| `POST /api/web/realtime/session` | Mint a ~60s OpenAI Realtime ephemeral token |
| `POST /api/web/typed` | Server-side LLM+TTS turn fallback → base64 MP3 |
| `POST /api/web/tts` | Verbatim TTS, no LLM (used by the Sing page) |
| `GET /api/web/song/search?q=` | Spotify track search (+ iTunes preview fallback) |
| `GET /api/web/song/audio?url=` | Allow-listed proxy for the 30s preview clip |

</details>

### 10.3 Browser-side JS module map (`static/web/js/`)

Every module is a deliberate, documented port of a specific Python module, so the
browser build behaves identically to the operator build:

| JS file | Ports | Notes |
|---|---|---|
| `auth.js` | — | `fetch` wrapper + login/logout/whoami, relies on the HttpOnly cookie |
| `bottango.js` | `transport/bottango_protocol.py` | Command framing + checksum |
| `serial.js` | `transport/bottango_serial_backend.py` | `WebSerialTransport` + a `MockTransport` |
| `envelope.js` | `motion/envelope.py` | `EnvelopeFollower` |
| `behavior.js` | `motion/behavior_engine.py` | `BehaviorEngine`; parity-tested against Python via Node |
| `motion.js` | `motion/scheduler.py` | `MotionScheduler`; adds the hidden-tab pause guard |
| `live_speaking_context.js` | `app.pipeline.LiveSpeakingContext` | Behavior-envelope + emphasis + heuristic phrase-boundary detection for streams with no TTS timing metadata |
| `realtime.js` | `conversation/realtime.py` | WebRTC `RealtimeSession` |
| `typed.js` | `pipeline.handle_typed_turn` (server-side half) | Decodes/plays the MP3, drives the envelope from an `AnalyserNode` |
| `audio_devices.js` | — | Mic/speaker `<select>` population + `setSinkId` output routing |
| `app.js` | — | Operator page wiring (`index.html`) |
| `admits.js` | — | Guest page wiring (`admits.html`) — event/context presets, PTT-by-default |
| `sing.js` | — | Jukebox page (§10.4) |
| `login.js` | — | Login form |

### 10.4 The Sing page (`/sing`, `sing.html` + `sing.js`) — song lip-sync jukebox

The newest, most elaborate feature (visible as the most recent run of commits in
`git log`). Flow: search a song → get a 30s preview URL → decode + analyze it
entirely client-side → play it back while driving jaw lip-sync and a beat-synced
"dance."

- **Search**: `GET /api/web/song/search` — server mints/caches a Spotify
  client-credentials app token (no user OAuth) and hits `/v1/search`. Spotify
  increasingly returns `preview_url: null`; the server concurrently backfills
  those via the **tokenless iTunes Search API** so more results end up playable.
- **Audio fetch**: proxied through `GET /api/web/song/audio`, which only allows
  `https://` URLs on an explicit host allow-list (`*.scdn.co`, `*.mzstatic.com`,
  `*.itunes.apple.com`) and caps the streamed size at 12MB — an anti-SSRF /
  anti-open-proxy guard, not just a CORS workaround.
- **Offline vocal isolation**: `precomputeVocalEnvelope` runs a from-scratch
  radix-2 Cooley-Tukey FFT (`makeFFT`, hand-written, no library) over the whole
  decoded clip and builds two separate envelope timelines:
  - a **vocal** envelope (drives the jaw) built from: center-channel extraction
    (`min(|L|,|R|)` per bin — vocals are almost always panned center, so
    hard-panned instruments largely cancel out), a raised-cosine vocal-band
    weighting (~300Hz–3.5kHz), and a **spectral-flatness** weighting that
    suppresses broadband/noisy content (drum hits) in favor of peaky/harmonic
    content (sung notes) — explicitly *not* a neural separator, just enough
    signal processing to stop the jaw popping on every snare hit;
  - a **music** envelope (drives the dance) from plain broadband mix RMS, so the
    body still grooves through purely instrumental sections where the vocal
    envelope is silent.
  - Both get adaptive normalization/compression (`compressVocalEnvelope`,
    `normalizeEnvelope`) so quiet songs still produce a full-range jaw swing.
- **Playback**: samples `envelope[currentTime + LOOK_AHEAD_S]` (100ms lookahead)
  each animation frame so the jaw opens just *before* the sound reaches the ear,
  compensating for serial+servo latency so it reads as in-sync.
- **The "dance"**: the scheduler's `onFrame` hook is overridden while a song is
  `playing` — beat-triggered discrete wing-flap bursts (not continuous
  oscillation — "parrots snap their wings up on an accent and settle"), an
  organic randomly-wandering head yaw (not a metronome sweep), and a
  `voxLevel`-gated "face forward and calm down while actively singing, dance
  more during instrumental gaps" behavior. This entirely replaces (rather than
  blends with) the conversational `BehaviorEngine` output for the duration of
  playback.
- **Showtime mode**: a switch that auto-picks from ~30 curated musical-theater
  search queries (Hamilton, Wicked, Les Mis, Phantom, etc. — chosen for broad
  booth recognizability) and auto-advances through them as filler between real
  conversations, with Prev/Pause/Next transport controls and a play history so
  "Prev" replays rather than re-randomizing.
- **"Make Maxwell talk"** on the same page reuses `TypedSession` pointed at
  `/api/web/tts` (verbatim speech, no LLM) so an operator can put exact words in
  his mouth between songs — it stops any playing song first so the jaw is free.

---

## 11. Every outbound API call, in one table

| Call | Model / endpoint | Triggered by | Cost profile |
|---|---|---|---|
| Realtime speech-to-speech | `gpt-realtime` via WebSocket (Python) or WebRTC (browser) | Starting Realtime mode | Continuous while the session is open |
| Chat Completions | `gpt-4o-mini` | Typed-turn / live-mic replies (`conversation/llm.py`, `/api/web/typed`) | Per turn |
| Audio transcription | `whisper-1` | Live-mic mode (`conversation/stt.py`); also Realtime's server-side input transcription | Per utterance |
| Text-to-speech | `gpt-4o-mini-tts` | Typed replies, replay, `/api/web/typed`, `/api/web/tts` | Per utterance |
| Vision (chat + image) | `gpt-4o-mini` | "What do you see" / `look_and_describe` tool (`vision/scene.py`) | Only if `vision.scene_enabled` (off by default); then on-demand, rate-limited to 1 per `scene_min_interval_s` |
| Spotify token (client-credentials) | `accounts.spotify.com/api/token` | First song search after token expiry | Cached, negligible |
| Spotify search | `api.spotify.com/v1/search` | Sing-page song search + Showtime filler picks | Per search |
| iTunes Search (tokenless) | `itunes.apple.com/search` | Fallback when Spotify's `preview_url` is null | Per missing preview |
| Song audio | Spotify/iTunes preview CDNs (`*.scdn.co`, `*.mzstatic.com`) | Proxied through `/api/web/song/audio` for browser playback | Per song play |

Everything else — the Bottango serial protocol, all motion/behavior computation,
the browser's WebRTC audio path once connected, face detection/recognition — is
local computation or a direct browser↔OpenAI/CDN connection with no Maxwell
server involved.

---

## 12. Deployment targets at a glance

| Target | Command | Hardware access | Notes |
|---|---|---|---|
| Booth laptop (easy) | Double-click `Start Maxwell.command` | Python process, USB-serial | One-time setup, then instant on repeat launches |
| Booth laptop (manual) | `python -m app.webapp --backend bottango` | Same | For anyone comfortable in a terminal |
| Any laptop, browser-hosted | `python -m app.web_app --host 127.0.0.1 --port 8080` | Browser Web Serial | Someone else's laptop, you don't want them installing Python |
| Fly / Render / Railway / any Docker host | `docker build … && docker run …` (or platform-native Dockerfile build) | Browser Web Serial | Needs `OPENAI_API_KEY`, `MAXWELL_WEB_PASSWORD_HASH`, `SESSION_SECRET`, `MAXWELL_ALLOWED_ORIGIN`, `MAXWELL_TRUST_FORWARDED_FOR=1` |
| Vercel | Import repo → set 3 env vars → deploy | Browser Web Serial | `api/index.py` + `vercel.json`; cold starts kept fast via a minimal `api/requirements.txt`; `VERCEL=1` auto-trusts `X-Forwarded-For` for rate limiting |

Known parity gap called out in the README: phrase-boundary nods / question-tilt /
emphasis spikes (which need utterance *text* early) aren't ported to browser
mode, because WebRTC doesn't reliably surface transcript text early enough — the
continuous idle motion and envelope-driven jaw/wing still carry it. The
fine-grained VAD sliders and the jaw-hammer/full-reset recovery buttons also
aren't in the browser UI yet.

---

## 13. Testing & dev tools

```bash
python -m pytest -q     # ~30 tests, <2s, no hardware required
```

Notable test files: `test_bottango_protocol.py` (wire-protocol framing/hashing),
`test_envelope.py`/`test_behavior_engine.py` (motion math), `test_realtime.py`
(Realtime event classification/dispatch without a live SDK), `test_web_auth.py`
(auth primitives), `test_web_app.py`/`test_api_index.py` (the two hosted-mode
servers, largest test files), `test_face_tracker.py`/`test_face_memory.py`/
`test_recognition_voting.py`/`test_greeting.py`/`test_gaze_behavior.py`/
`test_vision_config.py` (vision subsystem), and
**`test_js_behavior_parity.py`**, which actually shells out to `node` to run
`behavior.js` and asserts it satisfies the same behavioral contracts as the
Python engine — skipped automatically if `node` isn't on `PATH`.

Standalone tools (`tools/`), none of which need the full app running:

- **`vision_preview.py`** — opens a live camera window with detected faces boxed
  and the computed gaze crosshair drawn, for tuning gain/invert/deadzone before
  wiring vision into the bird. `--mock` runs with no webcam at all.
- **`recognition_calibrate.py`** — prints live cosine-similarity scores against a
  freshly-enrolled reference face, to pick `recognition_threshold`/`_margin` for
  your specific camera and lighting.
- **`smoke_test_hardware.py`** — gentle hardware check: serial handshake, effector
  registration, a couple of slow jaw openings — confirms the wire protocol works
  before the first real audio playback.
- **`analyze_jaw_timing.py`** — replays the exact production envelope-follower
  math against a WAV file and dumps per-20ms timing/PWM to CSV + a Markdown
  summary, for offline jaw-response debugging.

---

## 14. Operational quirks worth knowing

- **The jaw servo (GPIO 9) is the fragile one.** Long extension wires make it
  prone to dropping signal. The whole cluster of jaw-specific throttling
  (`jaw_min_delta`, `jaw_min_send_interval_s`, softer attack/release than head/
  wing) and the operator UI's "Full reset" button (deregister → re-register →
  hammer pin 9 with ~10 fast extreme swings to try to bridge a marginal dupont
  contact), "Wake sweep," "Wiggle jaw," and "Move jaw to a different pin" tools
  all exist because of this one flaky connection.
- **Everything on the serial line is fire-and-forget** during normal motion —
  correctness-sensitive commands (registration) are the only ones that wait for
  `OK`. This is intentional (§7.2), not an oversight.
- **The echo guard is the default fix for "Maxwell hears himself."** Realtime
  mode has no built-in AEC; `half_duplex: true` + a `playback_tail_ms` buffer is
  the mitigation, with smart barge-in layered on top so it doesn't come at the
  cost of natural interruptions.
- **Vision is fully optional and fails soft.** Camera/MediaPipe/InsightFace
  import failures, camera-open failures, and recognition-init failures are all
  caught and logged rather than propagated — the app is designed to run
  identically whether or not a webcam or those packages are present.
- **The two motion engines (Python, JS) are a deliberately maintained pair**, not
  a one-time port — comments throughout `behavior.js`/`envelope.js`/`motion.js`
  point back at their Python originals, defaults are duplicated with explicit
  "must stay in sync" comments, and there's a dedicated cross-language parity
  test. If you change a constant in one, the other needs the same change (or the
  `/api/web/motion-config` endpoint, which pushes `config.yaml`'s tuned values to
  the browser at runtime and is the preferred way to avoid needing to touch JS
  defaults at all).

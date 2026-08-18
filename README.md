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

## 🌐 Browser-hosted web mode (hosted on your server, zero downloads for the booth)

There is a separate, additive deployment mode that swaps the Python-on-every-laptop workflow for a **password-gated public website** the booth runner opens in Chrome. The server hosts the UI and talks to OpenAI; Maxwell's servos are driven directly from the browser over **Web Serial** on whatever laptop he's plugged into. The raw `OPENAI_API_KEY` **never** leaves the server.

This runs **in parallel** to the existing local mode — all existing commands, files, tests, and hardware flows are unchanged.

### When to use which mode

| Situation | Use |
|---|---|
| Single laptop, full operator control, every tuning knob exposed | **Local mode** (`python -m app.webapp`) |
| Someone else runs Maxwell at the fair on a laptop they control — you don't want them installing Python | **Hosted web mode** (`python -m app.web_app`) |

### Browser requirements (hosted mode)

- HTTPS (required for Web Serial and mic permission). `localhost` over HTTP also works for local testing.
- Chrome / Edge / Arc / any Chromium 89+. **Firefox and Safari do not expose Web Serial** — Maxwell's motion will be inert there, though typed conversation still works.
- Maxwell plugged via USB into the same laptop running the browser.

### Endpoints

| Path | What it does |
|---|---|
| `GET /login` | password form |
| `GET /` | operator UI (redirects to `/login` if not authed) |
| `POST /api/auth/login` | sets the signed, HttpOnly, SameSite=Lax session cookie |
| `POST /api/auth/logout` | clears session cookie |
| `GET /api/auth/me` | session introspection |
| `POST /api/web/realtime/session` | mints a short-lived OpenAI Realtime ephemeral token (~60 s) |
| `POST /api/web/typed` | typed-turn fallback: LLM + TTS run server-side, returns base64 MP3 |
| `GET /api/web/config` | non-sensitive defaults (`has_openai_key`, `realtime_voice`) |
| `GET /healthz` | uptime probe |

**The browser never sees `OPENAI_API_KEY`.** It only gets ephemeral `ek_...` tokens and server-rendered audio.

### Environment variables (hosted mode only)

Set in `.env` or the process environment:

- `OPENAI_API_KEY` — the real one; kept server-only.
- `MAXWELL_WEB_PASSWORD_HASH` — PBKDF2-SHA256 hash. Preferred.
- `MAXWELL_WEB_PASSWORD` — plaintext fallback for dev only.
- `SESSION_SECRET` — random bytes used to sign session cookies. If unset, each restart invalidates all sessions.
- `MAXWELL_ALLOWED_ORIGIN` — optional; pins state-changing API calls to a single origin.
- `MAXWELL_TRUST_FORWARDED_FOR=1` — set only when behind a trusted reverse proxy.
- `MAXWELL_WEB_INSECURE_COOKIE=1` — disables the `Secure` cookie flag for plain-HTTP local testing.

Generate a password hash:

```bash
python -m app.web_auth hash 'whatever-password-you-pick'
# -> pbkdf2_sha256$200000$...$...
```

Generate a session secret:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

### Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export MAXWELL_WEB_PASSWORD_HASH="$(python -m app.web_auth hash 'hunter2')"
export SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export MAXWELL_WEB_INSECURE_COOKIE=1   # only because this is plain HTTP
python -m app.web_app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/login`, sign in, click **Connect Maxwell**, then **Start realtime**. The mock transport toggle lets you try the UI with no hardware.

### Deploying on Vercel (GitHub → auto-deploy → custom domain)

Repo ships a `vercel.json` FastAPI entrypoint at `api/index.py`. Vercel runs it as one function (`/index`) and a `request.path` transform forwards the real URL (`/login`, `/relic`, …) into FastAPI. Do not catch-all rewrite to `/api` — CLI 58+ would then serve every page as path `/api` (`{"detail":"Not Found"}`).

1. **Push this repo to GitHub** (already done at `github.com/generousjj/talking-maxwell`).
2. [vercel.com/new](https://vercel.com/new) → **Import Git Repository** → pick `talking-maxwell`.
3. Framework preset: "**Other**" (Vercel auto-detects the Python function).  Root directory: `/`.  Build + output settings: leave defaults.
4. **Environment Variables** → add these three (plus anything else from `.env.example` you want overridden):

   | Key | Value |
   |---|---|
   | `OPENAI_API_KEY` | your real `sk-...` — kept server-side, never shipped to the browser |
   | `MAXWELL_WEB_PASSWORD_HASH` | output of `python -m app.web_auth hash 'your-booth-password'` |
   | `SESSION_SECRET` | `python -c 'import secrets; print(secrets.token_urlsafe(48))'` |

   After your domain is attached (next step), also add:

   | Key | Value |
   |---|---|
   | `MAXWELL_ALLOWED_ORIGIN` | `https://maxwell.yourdomain.com` — pins CSRF check to your domain |

5. **Deploy**. First build takes ~1 min (Vercel reads `api/requirements.txt`, which is kept minimal: `fastapi`, `httpx`, `python-dotenv`).
6. **Attach your domain**: project → **Settings → Domains** → add `maxwell.yourdomain.com` (or whatever subdomain) → Vercel shows you the CNAME / A record to set on your registrar. HTTPS cert is auto-issued.
7. Visit `https://maxwell.yourdomain.com/login`, sign in with the password you hashed in step 4, click **Connect Maxwell**, then **Start realtime**. Everything else — WebRTC to OpenAI, Web Serial to the ESP32, jaw envelope, motion scheduler — runs in the booth runner's browser.

Vercel-specific notes:

- `api/requirements.txt` is intentionally minimal (no `numpy`, `sounddevice`, `pyserial`, etc.) so cold starts stay under ~1 s.
- The in-memory login rate limiter resets on every cold start. Cookies are still cryptographically signed, so this is safe; the "8 attempts / 15 min" lockout just becomes per-invocation. If you want cross-invocation lockouts, swap `LoginLimiter` for a Vercel KV / Upstash Redis-backed one.
- `VERCEL=1` is set automatically by the runtime, which makes the auth code trust `x-forwarded-for` for per-IP rate-limiting. No action needed.
- You do **not** need `MAXWELL_WEB_INSECURE_COOKIE` on Vercel — HTTPS is default.
- The local operator app (`python -m app.webapp`) is **not** deployed by Vercel and cannot be — it needs direct USB-serial access, which doesn't exist in a serverless function. That remains a local-laptop-only thing.

### Deploying on Fly / Render / Railway / any Dockerfile host

Any Python-friendly PaaS works. A typical flow with a reverse proxy terminating TLS:

```
[ your HTTPS proxy (Caddy / Cloudflare / Nginx / Fly / Render) ]
                 |   forwards to
                 v
  python -m app.web_app --host 0.0.0.0 --port 8080
```

Minimal Dockerfile:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "-m", "app.web_app", "--host", "0.0.0.0", "--port", "8080"]
```

Environment variables to set on the platform:

- `OPENAI_API_KEY`
- `MAXWELL_WEB_PASSWORD_HASH`
- `SESSION_SECRET`
- `MAXWELL_ALLOWED_ORIGIN=https://your.domain`
- `MAXWELL_TRUST_FORWARDED_FOR=1` (if your platform puts you behind a proxy)

Do **not** set `MAXWELL_WEB_INSECURE_COOKIE` in production.

### Security notes

- PBKDF2-SHA256 (200k iters) for password hashing; plain-password fallback only meant for dev.
- Signed session cookies (HMAC-SHA256) — Secure + HttpOnly + SameSite=Lax.
- Per-IP login rate limit (8 attempts per 15 min; then 15-min lockout).
- State-changing API calls are Origin-checked.
- Login responses are deliberately uniform — a bad password returns `invalid_credentials`, never hints at which half was wrong.
- Operator-configurable prefs (voice, gain sliders) persist in `localStorage`; **password and API key never do**.
- The OpenAI Realtime client secret handed to the browser is short-lived (~60 s) and scoped to one session.

### Known parity gaps (browser mode vs local mode)

- Phrase-boundary nods, question-tilt, and emphasis spikes (text-driven) are not ported — browser WebRTC doesn't reliably give us the utterance text early enough. Continuous-sine idle + envelope-driven jaw/wings still look lively.
- The Python realtime implementation's fine-grained VAD controls are not exposed in the browser UI yet; it uses reasonable server VAD defaults or push-to-talk.
- Advanced operator features (jaw-hammer re-registration animation, full-reset button) aren't in the browser UI yet — `Disconnect` + `Connect` + `Wake sweep` is the equivalent flow.

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

#!/usr/bin/env bash
# Maxwell booth bootstrap.
#
# One-time setup + launcher for whoever's running Maxwell at the fair.
# Plug Maxwell into the laptop's USB, then run:
#
#     ./run_maxwell.sh
#
# It will:
#   1. Make sure Python 3.10+ is installed (warn if not).
#   2. Create a local virtual environment (./.venv) on first run.
#   3. Install all Python dependencies into that venv.
#   4. Prompt you for an OpenAI API key (saved to .env so you only do
#      this once on the laptop).
#   5. Auto-detect the USB serial port for Maxwell.
#   6. Open the admits-friendly tab in the default browser.
#   7. Start the web server.
#
# Subsequent runs are instant — it skips anything already done. You
# can always re-run this script after plugging Maxwell in; it's safe.

set -e
cd "$(dirname "$0")"

echo ""
echo "  ╭──────────────────────────────────────────╮"
echo "  │     🦜  Starting Maxwell  🦜              │"
echo "  ╰──────────────────────────────────────────╯"
echo ""

# ---------- 1. Check Python ----------
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ Python 3 isn't installed. Install it from https://www.python.org/downloads/ and rerun this script."
  exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_OK=$(python3 -c 'import sys; print(int(sys.version_info >= (3, 10)))')
if [ "$PY_OK" != "1" ]; then
  echo "❌ Python ${PY_VER} found, but Maxwell needs 3.10 or newer."
  echo "   Install from https://www.python.org/downloads/ and rerun this script."
  exit 1
fi
echo "✅ Python ${PY_VER} ready"

# ---------- 2. Virtualenv ----------
if [ ! -d ".venv" ]; then
  echo "📦 Creating Python environment (one-time, ~10 seconds)..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------- 3. Dependencies ----------
SENTINEL=".venv/.maxwell-deps-installed"
if [ ! -f "$SENTINEL" ] || [ requirements.txt -nt "$SENTINEL" ]; then
  echo "📥 Installing/updating dependencies (one-time, ~30 seconds)..."
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
  touch "$SENTINEL"
fi
echo "✅ Dependencies ready"

# ---------- 4. OpenAI API key ----------
if [ ! -f .env ] || ! grep -q '^OPENAI_API_KEY=sk-' .env 2>/dev/null; then
  echo ""
  echo "🔑 An OpenAI API key is needed for Maxwell to think + talk."
  echo "   Get one at: https://platform.openai.com/api-keys"
  echo "   (Starts with 'sk-...'.  Paste it below — it'll be saved to .env"
  echo "   so you don't have to enter it again on this laptop.)"
  echo ""
  printf "OpenAI API key: "
  read -r OPENAI_KEY
  if [ -z "$OPENAI_KEY" ]; then
    echo "❌ No key entered. Aborting."
    exit 1
  fi
  # Replace any existing OPENAI_API_KEY line, or append.
  if [ -f .env ] && grep -q '^OPENAI_API_KEY=' .env; then
    # Use a temp file because in-place sed is platform-finicky.
    awk -v k="$OPENAI_KEY" 'BEGIN{done=0} /^OPENAI_API_KEY=/{print "OPENAI_API_KEY=" k; done=1; next} {print} END{if(!done) print "OPENAI_API_KEY=" k}' .env > .env.tmp
    mv .env.tmp .env
  else
    echo "OPENAI_API_KEY=${OPENAI_KEY}" >> .env
  fi
  echo "✅ API key saved to .env"
fi

# ---------- 5. Serial port sanity check ----------
DETECTED=$(python3 -c '
try:
    from serial.tools import list_ports
    matches = [p for p in list_ports.comports()
               if "usbserial" in (p.device or "").lower()
               or "usbmodem" in (p.device or "").lower()
               or "CP210" in (p.description or "")
               or "CH340" in (p.description or "")
               or "Silicon Labs" in (p.manufacturer or "")]
    print(matches[0].device if matches else "")
except Exception:
    print("")
')
if [ -z "$DETECTED" ]; then
  echo ""
  echo "⚠️  No Maxwell USB serial port detected."
  echo "   Make sure Maxwell is plugged into the laptop, then press Enter."
  echo "   (Or Ctrl-C to cancel and try again.)"
  read -r _
fi

# ---------- 6. Launch ----------
PORT=8787
URL="http://127.0.0.1:${PORT}/admits"

# Make sure any previous Maxwell run that died without releasing the
# port gets cleared out. Without this, `python3 -m app.webapp` fails
# with "Address already in use" and the browser just sees
# "refused to connect" because nothing is listening on 8787.
if command -v lsof >/dev/null 2>&1 && lsof -nPi ":${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "⚠️  Port ${PORT} is already in use — trying to free it..."
  PIDS=$(lsof -nPi ":${PORT}" -sTCP:LISTEN -t 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    kill $PIDS 2>/dev/null || true
    sleep 1
    # Force-kill anything still hanging on.
    STILL=$(lsof -nPi ":${PORT}" -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$STILL" ]; then kill -9 $STILL 2>/dev/null || true; fi
  fi
fi

echo ""
echo "🚀 Launching web server..."
echo "   Browser will open automatically once it's ready."
echo "   Stop Maxwell with Ctrl-C in this window."
echo ""

# Wait for the server to actually bind the port before opening the
# browser. Previously we used `sleep 3`, which opened Chrome before
# the server was up on slow laptops — that's the "Safari/Chrome can't
# connect to the server" error a user might see at the fair.
( for i in $(seq 1 120); do
    if command -v nc >/dev/null 2>&1 && nc -z 127.0.0.1 "${PORT}" 2>/dev/null; then
      (command -v open >/dev/null 2>&1 && open "$URL") \
        || (command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL") \
        || true
      break
    fi
    sleep 0.25
  done ) &

# Trap any non-zero exit so the Terminal window keeps the Python
# traceback on screen instead of slamming shut when `.command` files
# finish — otherwise the operator would just see "refused to connect"
# in the browser with no clue why. This `read` block only runs on
# failure; a normal Ctrl-C exits cleanly.
set +e
python3 -m app.webapp --backend bottango
RC=$?
set -e

if [ "$RC" -ne 0 ] && [ "$RC" -ne 130 ]; then
  # 130 = SIGINT (Ctrl-C), the normal "stop Maxwell" exit path.
  echo ""
  echo "❌ Maxwell crashed (exit code ${RC})."
  echo "   The error above is the cause. Common fixes:"
  echo "   • Port ${PORT} busy — restart the laptop, or close any"
  echo "     other terminals running Maxwell."
  echo "   • Missing OPENAI_API_KEY — edit .env in this folder."
  echo "   • config.yaml unreadable — restore it from the repo."
  echo ""
  echo "Press Enter to close this window."
  read -r _
fi

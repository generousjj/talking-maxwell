// Web Serial transport for Maxwell.
//
// This is the browser-side counterpart to
// `transport/bottango_serial_backend.py`. It:
//
//   - requests a serial port from a user gesture (requestPort)
//   - opens it at 115200 baud
//   - runs the Bottango handshake (hRQ / btngoHSK)
//   - tSYN + xE + xC + rSVPin for each configured channel
//   - sends instant curves (`sCI`) at motion-tick rate
//   - reacts to a fresh `btngoHSK` line by re-registering servos
//
// Keep the feature set minimal but faithful — the goal is that a
// freshly paired ESP32 behaves identically to the Python backend
// modulo transport choice.

import { cmd, normalizedToCompressed, parseLine } from "./bottango.js";

const BAUD = 115200;

// Channel defaults mirror config.yaml exactly — these are the values
// that drive the working Python operator build. In normal use the app
// fetches /api/web/motion-config and passes overrides via the
// constructor, so any tuning in config.yaml flows to the browser with
// no JS edit. These hardcoded values are the fallback (used if the
// fetch fails, or for the MockTransport which never touches hardware).
//
// IMPORTANT: pins MUST match the ESP32 wiring. Sending to the wrong
// pin is silent — the firmware registers a servo that no hardware is
// connected to, and the real servo just sits there. That's what was
// happening before with head_lr/head_ud pinned to 10/11 instead of
// 5/6.
const DEFAULT_CHANNELS = {
  jaw:      { pin: 9, minPwm: 1450, maxPwm: 1800, maxPwmPerSec: 3000, startingPwm: 1625, inverted: true  },
  head_lr:  { pin: 5, minPwm: 1275, maxPwm: 1725, maxPwmPerSec: 1800, startingPwm: 1500, inverted: false },
  head_ud:  { pin: 6, minPwm: 850,  maxPwm: 2100, maxPwmPerSec: 1800, startingPwm: 1475, inverted: false },
  wing:     { pin: 3, minPwm: 1500, maxPwm: 2000, maxPwmPerSec: 3000, startingPwm: 1750, inverted: false },
};

// Bottango identifies effectors by pin for `sCI`.
function idFor(ch) { return ch.pin; }

export class WebSerialTransport {
  constructor({ log = () => {}, onState = () => {}, channels } = {}) {
    this.log = log;
    this.onState = onState;
    this.channels = Object.assign({}, DEFAULT_CHANNELS, channels || {});
    this.port = null;
    this.writer = null;
    this.reader = null;
    this._readerLoop = null;
    this._connected = false;
    this._writeQueue = Promise.resolve();
    this._lastSent = Object.create(null);
    this._pendingOks = [];
    this._lastHSKAt = 0;
    this._readerClosed = false;
  }

  static isSupported() { return typeof navigator !== "undefined" && "serial" in navigator; }

  async connect() {
    if (!WebSerialTransport.isSupported()) {
      throw new Error("Web Serial not supported in this browser");
    }
    if (this._connected) return;

    // MUST be called from a user gesture.
    const port = await navigator.serial.requestPort({});
    await port.open({ baudRate: BAUD });
    this.port = port;

    this.writer = port.writable.getWriter();
    this.reader = port.readable.getReader();
    this._readerClosed = false;

    this._startReaderLoop();
    this._setState({ connected: true, phase: "handshake" });

    try {
      await this._handshake();
      await this._registerAllServos("initial");
    } catch (e) {
      this.log(`connect failed: ${e.message || e}`);
      await this.disconnect();
      throw e;
    }
    this._connected = true;
    this._setState({ connected: true, phase: "ready" });
    this.log("serial connected + servos registered");

    if (port.addEventListener) {
      port.addEventListener("disconnect", () => {
        this.log("serial port disconnected by OS");
        this._connected = false;
        this._setState({ connected: false, phase: "disconnected" });
      });
    }
  }

  isConnected() { return this._connected; }

  async disconnect() {
    this._connected = false;
    try { if (this.writer) await this._write(cmd.stop()); } catch (_) {}
    try { if (this.writer) this.writer.releaseLock(); } catch (_) {}
    try { if (this.reader) await this.reader.cancel(); } catch (_) {}
    try { if (this.reader) this.reader.releaseLock(); } catch (_) {}
    this.writer = null;
    this.reader = null;
    try { if (this.port) await this.port.close(); } catch (_) {}
    this.port = null;
    this._setState({ connected: false, phase: "closed" });
  }

  // Motion tick: frame is { jaw, head_lr, head_ud, wing } in [0,1].
  // Coalesce: only send the channels that changed materially to keep
  // the serial line below ~30 lines/s total.
  async sendFrame(frame) {
    if (!this._connected) return;
    const writes = [];
    for (const [name, ch] of Object.entries(this.channels)) {
      let v = frame[name];
      if (typeof v !== "number") continue;
      if (v < 0) v = 0; if (v > 1) v = 1;
      if (ch.inverted) v = 1 - v;
      const compressed = normalizedToCompressed(v);
      const last = this._lastSent[name];
      if (last !== undefined && Math.abs(compressed - last) < 5) continue;
      this._lastSent[name] = compressed;
      writes.push(cmd.instantCurve(idFor(ch), compressed));
    }
    if (writes.length === 0) return;
    const total = writes.reduce((n, w) => n + w.length, 0);
    const merged = new Uint8Array(total);
    let off = 0;
    for (const w of writes) { merged.set(w, off); off += w.length; }
    await this._write(merged);
  }

  async wakeSweep() {
    if (!this._connected) return;
    this.log("wake sweep");
    const frames = [
      { jaw: 0.0, head_lr: 0.6, head_ud: 0.4, wing: 0.9 },
      { jaw: 0.2, head_lr: 0.4, head_ud: 0.6, wing: 0.1 },
      { jaw: 0.4, head_lr: 0.5, head_ud: 0.5, wing: 0.9 },
      { jaw: 0.0, head_lr: 0.5, head_ud: 0.5, wing: 0.1 },
    ];
    for (const f of frames) {
      await this.sendFrame(f);
      await new Promise((r) => setTimeout(r, 180));
    }
    await this.center();
  }

  async center() {
    await this.sendFrame({ jaw: 0, head_lr: 0.5, head_ud: 0.5, wing: 0 });
  }

  async safeStop() {
    try { await this._write(cmd.stop()); } catch (_) {}
  }

  // ----- internals -----

  _setState(s) { try { this.onState(s); } catch (_) {} }

  _startReaderLoop() {
    this._readerLoop = (async () => {
      let buf = "";
      const dec = new TextDecoder();
      try {
        while (this.reader) {
          const { value, done } = await this.reader.read();
          if (done) break;
          if (!value) continue;
          buf += dec.decode(value, { stream: true });
          let idx;
          while ((idx = buf.indexOf("\n")) !== -1) {
            const line = buf.slice(0, idx);
            buf = buf.slice(idx + 1);
            this._onLine(line);
          }
        }
      } catch (e) {
        this.log(`reader loop error: ${e.message || e}`);
      } finally {
        this._readerClosed = true;
      }
    })();
  }

  _onLine(line) {
    const parsed = parseLine(line);
    if (parsed.type === "hsk") {
      if (Date.now() - this._lastHSKAt > 1500) {
        this._lastHSKAt = Date.now();
        this.log(`firmware hello: version ${parsed.version} (accepting=${parsed.accepting})`);
        // Unsolicited HSK => firmware rebooted; re-register in the
        // background so we keep driving motion.
        if (this._connected) {
          this._registerAllServos("firmware-reset").catch((e) =>
            this.log(`re-register failed: ${e.message || e}`));
        }
      }
      // Resolve any waiter
      const w = this._pendingHSK;
      if (w) { this._pendingHSK = null; w.resolve(parsed); }
    } else if (parsed.type === "ok") {
      const w = this._pendingOks.shift();
      if (w) w.resolve();
    }
  }

  async _handshake() {
    for (let attempt = 0; attempt < 6; attempt++) {
      const rand = (Math.random() * 1_000_000_000) | 0;
      await this._write(cmd.handshake(rand));
      try {
        await this._waitForHSK(1200);
        return;
      } catch (_) { /* retry */ }
    }
    throw new Error("handshake timeout");
  }

  _waitForHSK(ms) {
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => { this._pendingHSK = null; reject(new Error("hsk timeout")); }, ms);
      this._pendingHSK = {
        resolve: (r) => { clearTimeout(t); resolve(r); },
      };
    });
  }

  async _registerAllServos(reason) {
    this.log(`registering servos (${reason})`);
    await this._sendAndWaitOk(cmd.timeSync(performance.now()));
    await this._sendAndWaitOk(cmd.deregisterAll());
    await this._sendAndWaitOk(cmd.clearAllCurves());
    for (const [name, ch] of Object.entries(this.channels)) {
      await this._sendAndWaitOk(cmd.registerPinServo(ch));
      this.log(`  registered ${name} -> pin ${ch.pin} (pwm ${ch.minPwm}-${ch.maxPwm}${ch.inverted ? ", INVERTED" : ""})`);
    }
    this._lastSent = Object.create(null);
  }

  async _sendAndWaitOk(bytes, timeoutMs = 1500) {
    const waiter = new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("OK timeout")), timeoutMs);
      this._pendingOks.push({
        resolve: () => { clearTimeout(t); resolve(); },
      });
    });
    await this._write(bytes);
    await waiter;
  }

  _write(bytes) {
    // Serialize writes so they hit the wire in-order; Web Serial
    // writers don't allow parallel writes anyway.
    this._writeQueue = this._writeQueue
      .then(() => this.writer && this.writer.write(bytes))
      .catch((e) => { this.log(`write failed: ${e.message || e}`); });
    return this._writeQueue;
  }
}

// ------------------------------------------------------------------
// Mock transport: satisfies the same interface without any hardware.
// Useful for developing on a laptop without Maxwell plugged in.
// ------------------------------------------------------------------

export class MockTransport {
  constructor({ log = () => {}, onState = () => {} } = {}) {
    this.log = log;
    this.onState = onState;
    this._connected = false;
    this.lastFrame = { jaw: 0, head_lr: 0.5, head_ud: 0.5, wing: 0 };
  }
  static isSupported() { return true; }
  async connect() {
    await new Promise((r) => setTimeout(r, 180));
    this._connected = true;
    this.onState({ connected: true, phase: "ready", mock: true });
    this.log("mock serial connected (no hardware)");
  }
  isConnected() { return this._connected; }
  async disconnect() {
    this._connected = false;
    this.onState({ connected: false, phase: "closed", mock: true });
  }
  async sendFrame(f) { this.lastFrame = { ...this.lastFrame, ...f }; }
  async wakeSweep() { this.log("mock wake sweep"); }
  async center() { this.lastFrame = { jaw: 0, head_lr: 0.5, head_ud: 0.5, wing: 0 }; }
  async safeStop() {}
}

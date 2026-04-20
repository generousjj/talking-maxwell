// 30 Hz motion scheduler. Drives the behavior engine and forwards
// each frame to whichever transport is currently connected.

export class MotionScheduler {
  // `speakingContextProvider` mirrors Python MotionScheduler's knob
  // of the same name: a function called each tick (when speaking)
  // that returns a SpeakingContext snapshot, which is passed through
  // to BehaviorEngine.tick. Leave null to drive the engine without
  // any speaking-mode context (IDLE-ish motion only).
  constructor({ hz = 30, behavior, transport, onFrame = () => {}, speakingContextProvider = null } = {}) {
    this.period = 1000 / hz;
    this.behavior = behavior;
    this.transport = transport;
    this.onFrame = onFrame;
    this.speakingContextProvider = speakingContextProvider;
    this._timer = null;
    this._running = false;
  }

  setTransport(t) { this.transport = t; }
  setSpeakingContextProvider(fn) { this.speakingContextProvider = fn; }

  start() {
    if (this._running) return;
    this._running = true;
    const tick = async () => {
      if (!this._running) return;
      // Only pull a speaking snapshot while in SPEAKING state, matching
      // Python's scheduler logic byte-for-byte.
      let ctx = null;
      if (this.behavior && this.behavior.state === "speaking" && this.speakingContextProvider) {
        try { ctx = this.speakingContextProvider(); } catch (_) { ctx = null; }
      }
      const frame = this.behavior.tick(ctx);
      try { this.onFrame(frame); } catch (_) {}
      if (this.transport && this.transport.isConnected()) {
        try { await this.transport.sendFrame(frame); } catch (_) {}
      }
      if (this._running) this._timer = setTimeout(tick, this.period);
    };
    tick();
  }

  stop() {
    this._running = false;
    if (this._timer) clearTimeout(this._timer);
    this._timer = null;
  }
}

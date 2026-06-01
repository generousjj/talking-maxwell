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
    // Background-tab guard: when the tab is hidden, browsers throttle
    // setTimeout to ~1 Hz. If we keep ticking, the behavior engine
    // integrates ~30 frames of motion in each delayed wake-up and the
    // servos jump in big chunks (~once a second) which looks like
    // physical jerking. Instead we freeze the scheduler entirely
    // while the page is hidden and send a single neutral "center"
    // frame so Maxwell holds a safe pose.
    this._paused = false;
    if (typeof document !== "undefined" && document.addEventListener) {
      document.addEventListener("visibilitychange", () => {
        this._handleVisibility();
      });
    }
  }

  setTransport(t) { this.transport = t; }
  setSpeakingContextProvider(fn) { this.speakingContextProvider = fn; }

  start() {
    if (this._running) return;
    this._running = true;
    const tick = async () => {
      if (!this._running) return;
      if (this._paused) {
        // Re-check soon so we wake up quickly when the user returns.
        this._timer = setTimeout(tick, 250);
        return;
      }
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

  async _handleVisibility() {
    const hidden = typeof document !== "undefined" && document.hidden;
    if (hidden && !this._paused) {
      this._paused = true;
      // Park the servos in a neutral pose so they don't sit on
      // whatever random idle frame happened to be in flight.
      if (this.transport && this.transport.isConnected && this.transport.isConnected()
          && typeof this.transport.center === "function") {
        try { await this.transport.center(); } catch (_) {}
      }
    } else if (!hidden && this._paused) {
      this._paused = false;
      // Reset the behavior engine's `_last_tick` so the first frame
      // after resume doesn't try to integrate over the entire hidden
      // duration in a single 0.5 s step.
      if (this.behavior) this.behavior._last_tick = 0;
    }
  }
}

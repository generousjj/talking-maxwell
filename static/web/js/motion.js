// 30 Hz motion scheduler. Drives the behavior engine and forwards
// each frame to whichever transport is currently connected.

export class MotionScheduler {
  constructor({ hz = 30, behavior, transport, onFrame = () => {} } = {}) {
    this.period = 1000 / hz;
    this.behavior = behavior;
    this.transport = transport;
    this.onFrame = onFrame;
    this._timer = null;
    this._running = false;
  }

  setTransport(t) { this.transport = t; }

  start() {
    if (this._running) return;
    this._running = true;
    const tick = async () => {
      if (!this._running) return;
      const frame = this.behavior.tick();
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

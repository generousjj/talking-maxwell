// Bottango wire-protocol helpers. Direct port of
// `transport/bottango_protocol.py` — every command body is a plain
// ASCII string terminated with ",h<sum>\n" where <sum> is the sum of
// the ASCII codes of every character BEFORE the ",h" separator.
//
// The same subset used by the Python serial backend is implemented
// here: handshake, time-sync, register/deregister servos, instant
// curves (`sCI`), STOP, clear curves, and clear effectors.

export const SIGNAL_MAX = 8192;

function asciiSum(body) {
  let sum = 0;
  for (let i = 0; i < body.length; i++) sum += body.charCodeAt(i);
  return sum;
}

export function frameCommand(body) {
  const line = `${body},h${asciiSum(body)}\n`;
  const bytes = new Uint8Array(line.length);
  for (let i = 0; i < line.length; i++) bytes[i] = line.charCodeAt(i);
  return bytes;
}

export const cmd = {
  handshake: (rand) => frameCommand(`hRQ,${(rand | 0)}`),
  timeSync: (ms) => frameCommand(`tSYN,${Math.round(ms)}`),
  registerPinServo: ({ pin, minPwm, maxPwm, maxPwmPerSec, startingPwm }) =>
    frameCommand(
      `rSVPin,${pin|0},${minPwm|0},${maxPwm|0},${maxPwmPerSec|0},${startingPwm|0}`,
    ),
  deregisterEffector: (id) => frameCommand(`xUE,${id}`),
  deregisterAll: () => frameCommand("xE"),
  clearAllCurves: () => frameCommand("xC"),
  stop: () => frameCommand("STOP"),
  instantCurve: (id, compressed) =>
    frameCommand(`sCI,${id},${Math.max(0, Math.min(SIGNAL_MAX, compressed | 0))}`),
};

export function normalizedToCompressed(v) {
  if (!(v >= 0)) v = 0;
  if (v > 1) v = 1;
  return Math.round(v * SIGNAL_MAX);
}

// Parses a line from the firmware. Returns one of:
//   { type: "hsk", version, randomCode, accepting }
//   { type: "ok" }
//   { type: "other", line }
export function parseLine(line) {
  const s = String(line || "").trim();
  if (s === "") return { type: "other", line: s };
  if (s === "OK") return { type: "ok" };
  if (s.startsWith("btngoHSK,")) {
    const parts = s.split(",");
    if (parts.length >= 4) {
      return {
        type: "hsk",
        version: parts[1],
        randomCode: parts[2],
        accepting: parts[3] === "1",
      };
    }
  }
  return { type: "other", line: s };
}

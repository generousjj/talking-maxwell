/**
 * Relic — browser control panel for the artifact prop (Sparkle Motion Mini).
 *
 * Phase 1: Web Serial + localStorage range editor + config.h export.
 * Hardware must be plugged into the same laptop running Chrome/Edge.
 */

const BAUD = 115200;
const STORAGE_KEY = "relic-prop-config-v1";

const DEFAULT_CONFIG = {
  totalLeds: 144,
  triggerThreshold: 2.5,
  releaseThreshold: 1.2,
  orb: [{ start: 0, end: 4 }],
  magma: [
    { start: 5, end: 10 },
    { start: 12, end: 13 },
  ],
  crystal1: [{ start: 20, end: 25 }],
  crystal2: [{ start: 30, end: 36 }],
  hidden: [],
};

/** @type {typeof DEFAULT_CONFIG} */
let config = loadConfig();

const $ = (id) => document.getElementById(id);

const els = {
  connectBtn: $("connectBtn"),
  disconnectBtn: $("disconnectBtn"),
  connPill: $("connPill"),
  serialWarn: $("serialWarn"),
  triggerBtn: $("triggerBtn"),
  resetBtn: $("resetBtn"),
  calBtn: $("calBtn"),
  printBtn: $("printBtn"),
  magX: $("magX"),
  magY: $("magY"),
  magZ: $("magZ"),
  magD: $("magD"),
  magState: $("magState"),
  magPresent: $("magPresent"),
  deltaFill: $("deltaFill"),
  triggerThresh: $("triggerThresh"),
  releaseThresh: $("releaseThresh"),
  totalLeds: $("totalLeds"),
  rangeSections: $("rangeSections"),
  serialLog: $("serialLog"),
  saveBtn: $("saveBtn"),
  exportBtn: $("exportBtn"),
  resetCfgBtn: $("resetCfgBtn"),
  clearLogBtn: $("clearLogBtn"),
  exportDialog: $("exportDialog"),
  exportText: $("exportText"),
  copyExportBtn: $("copyExportBtn"),
  logoutBtn: $("logoutBtn"),
};

/** @type {SerialPort|null} */
let port = null;
let reader = null;
let writer = null;
let readLoopAbort = null;

function loadConfig() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULT_CONFIG, ...JSON.parse(raw) };
  } catch (_) { /* ignore */ }
  return structuredClone(DEFAULT_CONFIG);
}

function persistConfig() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

function log(line, { err = false } = {}) {
  const ts = new Date().toLocaleTimeString();
  const prefix = err ? "[err]" : "";
  els.serialLog.textContent += `${ts} ${prefix}${line}\n`;
  els.serialLog.scrollTop = els.serialLog.scrollHeight;
}

function setConnected(on) {
  els.connPill.textContent = on ? "Linked" : "Unlinked";
  els.connPill.className = on ? "status-chip status-on" : "status-chip status-off";
  els.connectBtn.disabled = on;
  els.disconnectBtn.disabled = !on;
  for (const b of [els.triggerBtn, els.resetBtn, els.calBtn, els.printBtn]) {
    b.disabled = !on;
  }
}

const STATE_CLASS = {
  DORMANT: "state-dormant",
  ACTIVATING: "state-activating",
  ACTIVE: "state-active",
  DEACTIVATING: "state-deactivating",
};

function updateDeltaBar(deltaStr) {
  if (!els.deltaFill) return;
  const d = parseFloat(deltaStr);
  if (!Number.isFinite(d)) {
    els.deltaFill.style.width = "0%";
    return;
  }
  const trigger = Number(els.triggerThresh?.value) || config.triggerThreshold || 2.5;
  const pct = Math.min(100, (d / (trigger * 1.5)) * 100);
  els.deltaFill.style.width = `${pct}%`;
}

function parseMagLine(line) {
  const m = line.match(
    /X=([-\d.]+)\s+Y=([-\d.]+)\s+Z=([-\d.]+)\s+d=([-\d.]+).*state=(\w+).*magnet=(\w+)/i
  );
  if (!m) return;
  els.magX.textContent = m[1];
  els.magY.textContent = m[2];
  els.magZ.textContent = m[3];
  els.magD.textContent = `${m[4]} mT`;
  updateDeltaBar(m[4]);

  const state = m[5].toUpperCase();
  els.magState.textContent = state;
  els.magState.className = `state-badge ${STATE_CLASS[state] || "state-dormant"}`;

  const near = m[6].toUpperCase() === "YES";
  els.magPresent.textContent = near ? "Yes" : "No";
  els.magPresent.className = near ? "sigil-yes" : "sigil-no";
}

async function startReadLoop() {
  if (!port?.readable) return;
  readLoopAbort = new AbortController();
  reader = port.readable.getReader();
  const dec = new TextDecoder();
  let buf = "";

  try {
    while (!readLoopAbort.signal.aborted) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split(/\r?\n/);
      buf = parts.pop() || "";
      for (const line of parts) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        log(trimmed);
        if (trimmed.startsWith("[mag]")) parseMagLine(trimmed);
      }
    }
  } catch (e) {
    if (!readLoopAbort?.signal.aborted) {
      log(e.message || String(e), { err: true });
    }
  } finally {
    try { reader?.releaseLock(); } catch (_) {}
    reader = null;
  }
}

async function sendCmd(ch) {
  if (!writer) throw new Error("Not connected");
  const enc = new TextEncoder();
  await writer.write(enc.encode(ch));
  log(`→ sent '${ch}'`);
}

async function connect() {
  if (!("serial" in navigator)) {
    throw new Error("Web Serial not supported");
  }
  const ports = await navigator.serial.getPorts();
  port = ports[0] || (await navigator.serial.requestPort({}));
  await port.open({ baudRate: BAUD });
  writer = port.writable.getWriter();
  setConnected(true);
  log("Linked to the relic at 115200 baud");
  startReadLoop();
}

async function disconnect() {
  readLoopAbort?.abort();
  readLoopAbort = null;
  try { await writer?.close(); } catch (_) {}
  writer = null;
  try { await reader?.cancel(); } catch (_) {}
  reader = null;
  try { await port?.close(); } catch (_) {}
  port = null;
  setConnected(false);
  log("Link broken");
}

function dirLabel(start, end) {
  if (start === end) return "single";
  return start < end ? `${start}→${end}` : `${start}←${end}`;
}

function renderRangeSection(key, title, tag, desc, extraClass, showDirection) {
  const section = document.createElement("div");
  section.className = `range-section ${extraClass || ""}`;
  section.innerHTML = `
    <div class="range-section-head">
      <div>
        <h3>${title}</h3>
        <p class="range-desc">${desc}</p>
      </div>
      <span class="section-tag">${tag}</span>
    </div>
    <div class="range-rows" data-key="${key}"></div>
    <button type="button" class="add-range" data-key="${key}">+ Add segment</button>
  `;
  const rows = section.querySelector(".range-rows");
  const ranges = config[key] || [];
  ranges.forEach((r, i) => rows.appendChild(makeRangeRow(key, i, r, showDirection)));
  if (!ranges.length) rows.appendChild(makeRangeRow(key, 0, { start: 0, end: 0 }, showDirection));
  return section;
}

function makeRangeRow(key, index, range, showDirection) {
  const row = document.createElement("div");
  row.className = "range-row";
  row.dataset.key = key;
  row.dataset.index = String(index);

  const start = document.createElement("input");
  start.type = "number";
  start.min = "0";
  start.value = String(range.start);
  start.title = "Start index (inclusive, zero-based)";

  const end = document.createElement("input");
  end.type = "number";
  end.min = "0";
  end.value = String(range.end);
  end.title = showDirection
    ? "End index — first→last sets magma propagation direction"
    : "End index (inclusive)";

  const dir = document.createElement("span");
  dir.className = "dir-label";
  if (showDirection) {
    const updateDir = () => {
      dir.textContent = dirLabel(Number(start.value), Number(end.value));
    };
    start.addEventListener("input", updateDir);
    end.addEventListener("input", updateDir);
    updateDir();
  } else {
    dir.textContent = "";
  }

  const swap = document.createElement("button");
  swap.type = "button";
  swap.className = "btn-icon";
  swap.textContent = "⇄";
  swap.title = "Reverse propagation direction";
  swap.addEventListener("click", () => {
    const s = start.value;
    start.value = end.value;
    end.value = s;
    start.dispatchEvent(new Event("input"));
  });

  const del = document.createElement("button");
  del.type = "button";
  del.className = "btn-icon";
  del.textContent = "×";
  del.title = "Remove segment";
  del.addEventListener("click", () => row.remove());

  row.append(start, end, dir, swap, del);
  return row;
}

function collectRangesFromDom() {
  const out = {};
  for (const key of ["orb", "magma", "crystal1", "crystal2", "hidden"]) {
    out[key] = [];
    document.querySelectorAll(`.range-row[data-key="${key}"]`).forEach((row) => {
      const inputs = row.querySelectorAll('input[type="number"]');
      if (inputs.length < 2) return;
      const start = Number(inputs[0].value);
      const end = Number(inputs[1].value);
      if (Number.isFinite(start) && Number.isFinite(end)) {
        out[key].push({ start, end });
      }
    });
  }
  return out;
}

function syncConfigFromForm() {
  config.totalLeds = Number(els.totalLeds.value) || 144;
  config.triggerThreshold = Number(els.triggerThresh.value) || 2.5;
  config.releaseThreshold = Number(els.releaseThresh.value) || 1.2;
  Object.assign(config, collectRangesFromDom());
}

function renderForm() {
  els.totalLeds.value = String(config.totalLeds);
  els.triggerThresh.value = String(config.triggerThreshold);
  els.releaseThresh.value = String(config.releaseThreshold);

  els.rangeSections.replaceChildren(
    renderRangeSection(
      "orb", "The Orb", "Radiance",
      "Warm core glow — all pixels in a range breathe together.",
      "orb", false,
    ),
    renderRangeSection(
      "magma", "Molten Cracks", "Fire line",
      "Heat spreads from first index toward second. Reverse to change flow.",
      "magma", true,
    ),
    renderRangeSection(
      "crystal1", "Crystal Cluster I", "Amethyst",
      "First gem formation on the rock face.",
      "crystal", false,
    ),
    renderRangeSection(
      "crystal2", "Crystal Cluster II", "Amethyst",
      "Second gem formation — shimmers independently.",
      "crystal", false,
    ),
    renderRangeSection(
      "hidden", "Sealed Sections", "Dark",
      "Pixels forced off — gaps, dead wire, or unused stone.",
      "hidden", false,
    ),
  );

  els.rangeSections.querySelectorAll(".add-range").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.key;
      const rows = btn.previousElementSibling;
      const showDir = key === "magma";
      rows.appendChild(makeRangeRow(key, rows.children.length, { start: 0, end: 0 }, showDir));
    });
  });
}

function rangesToCpp(name, ranges) {
  if (!ranges.length) {
    return `static const LedRange ${name}[] = {\n  // (none)\n};`;
  }
  const lines = ranges.map((r) => `  {${r.start}, ${r.end}},`);
  return `static const LedRange ${name}[] = {\n${lines.join("\n")}\n};`;
}

function buildExportSnippet() {
  syncConfigFromForm();
  const c = config;
  return `// --- Relic export (${new Date().toISOString().slice(0, 10)}) ---
// BTF-LIGHTING WS2812B ECO · 144 LEDs/m · GRB · paste into config.h

static const uint16_t TOTAL_LEDS = ${c.totalLeds};

${rangesToCpp("ORB_RANGES", c.orb)}
static const uint8_t ORB_RANGE_COUNT = sizeof(ORB_RANGES) / sizeof(ORB_RANGES[0]);

${rangesToCpp("MAGMA_RANGES", c.magma)}
static const uint8_t MAGMA_RANGE_COUNT = sizeof(MAGMA_RANGES) / sizeof(MAGMA_RANGES[0]);

${rangesToCpp("CRYSTAL_1_RANGES", c.crystal1)}
static const uint8_t CRYSTAL_1_RANGE_COUNT =
    sizeof(CRYSTAL_1_RANGES) / sizeof(CRYSTAL_1_RANGES[0]);

${rangesToCpp("CRYSTAL_2_RANGES", c.crystal2)}
static const uint8_t CRYSTAL_2_RANGE_COUNT =
    sizeof(CRYSTAL_2_RANGES) / sizeof(CRYSTAL_2_RANGES[0]);

${rangesToCpp("HIDDEN_RANGES", c.hidden)}
static const uint8_t HIDDEN_RANGE_COUNT =
    sizeof(HIDDEN_RANGES) / sizeof(HIDDEN_RANGES[0]);

static const float MAGNET_TRIGGER_THRESHOLD = ${c.triggerThreshold}f;
static const float MAGNET_RELEASE_THRESHOLD = ${c.releaseThreshold}f;
`;
}

async function checkSession() {
  try {
    const resp = await fetch("/api/auth/me");
    const data = await resp.json();
    if (!data.ok) window.location.href = "/login";
  } catch (_) { /* offline dev */ }
}

function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

function handleShortcut(key) {
  const run = (fn) => fn().catch((e) => log(e.message, { err: true }));
  switch (key) {
    case "t":
      if (!els.triggerBtn.disabled) run(() => sendCmd("t"));
      break;
    case "r":
      if (!els.resetBtn.disabled) run(() => sendCmd("r"));
      break;
    case "c":
      if (!els.calBtn.disabled) run(() => sendCmd("c"));
      break;
    case "p":
      if (!els.printBtn.disabled) run(() => sendCmd("p"));
      break;
    default:
      break;
  }
}

function init() {
  if (!("serial" in navigator)) {
    els.serialWarn.classList.remove("hidden");
    els.connectBtn.disabled = true;
  }

  renderForm();
  checkSession();

  els.connectBtn.addEventListener("click", () => connect().catch((e) => log(e.message, { err: true })));
  els.disconnectBtn.addEventListener("click", () => disconnect().catch((e) => log(e.message, { err: true })));
  els.triggerBtn.addEventListener("click", () => sendCmd("t").catch((e) => log(e.message, { err: true })));
  els.resetBtn.addEventListener("click", () => sendCmd("r").catch((e) => log(e.message, { err: true })));
  els.calBtn.addEventListener("click", () => sendCmd("c").catch((e) => log(e.message, { err: true })));
  els.printBtn.addEventListener("click", () => sendCmd("p").catch((e) => log(e.message, { err: true })));

  els.saveBtn.addEventListener("click", () => {
    syncConfigFromForm();
    persistConfig();
    log("Map saved in this browser");
  });

  els.exportBtn.addEventListener("click", () => {
    els.exportText.value = buildExportSnippet();
    els.exportDialog.showModal();
  });

  els.copyExportBtn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(els.exportText.value);
    log("Inscription copied");
  });

  els.resetCfgBtn.addEventListener("click", () => {
    if (!confirm("Restore the default stone map?")) return;
    config = structuredClone(DEFAULT_CONFIG);
    persistConfig();
    renderForm();
    log("Default map restored");
  });

  els.clearLogBtn.addEventListener("click", () => {
    els.serialLog.textContent = "";
  });

  els.logoutBtn.addEventListener("click", async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    window.location.href = "/login";
  });

  document.addEventListener("keydown", (e) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    if (els.exportDialog.open) return;
    if (isTypingTarget(document.activeElement)) return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : "";
    if (!"trcp".includes(key)) return;
    e.preventDefault();
    handleShortcut(key);
  });
}

init();

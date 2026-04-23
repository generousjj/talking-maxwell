// Shared helpers for input (mic) + output (speaker) device pickers in
// cloud / realtime mode. Same module is used by the operator page and
// the guest admits page — each passes in its own pair of <select>
// elements and storage keys.
//
// Two friction points we work around here:
//   1. `navigator.mediaDevices.enumerateDevices()` returns entries
//      with BLANK labels until the page has been granted mic permission
//      at least once. We populate with generic names ("Microphone 1",
//      "Speaker 2") in that case and re-populate once permission is
//      granted (e.g. after the first Start click).
//   2. `HTMLMediaElement.setSinkId()` only exists on Chromium / recent
//      WebKit. We silently skip it on browsers that don't support it
//      instead of erroring out.

export function outputPickerSupported() {
  if (typeof HTMLMediaElement === "undefined") return false;
  return typeof HTMLMediaElement.prototype.setSinkId === "function";
}

export function devicesApiSupported() {
  return !!(typeof navigator !== "undefined"
    && navigator.mediaDevices
    && navigator.mediaDevices.enumerateDevices);
}

function labelFor(d, indexByKind) {
  if (d.label && d.label.trim()) return d.label;
  const n = indexByKind[d.kind] = (indexByKind[d.kind] || 0) + 1;
  const kindLabel = d.kind === "audioinput" ? "Microphone"
                  : d.kind === "audiooutput" ? "Speaker"
                  : "Device";
  return `${kindLabel} ${n}`;
}

export async function listAudioDevices() {
  if (!devicesApiSupported()) return { inputs: [], outputs: [] };
  let devices;
  try { devices = await navigator.mediaDevices.enumerateDevices(); }
  catch (_) { return { inputs: [], outputs: [] }; }
  const idx = {};
  const inputs = [];
  const outputs = [];
  for (const d of devices) {
    const entry = { deviceId: d.deviceId, label: labelFor(d, idx) };
    if (d.kind === "audioinput")  inputs.push(entry);
    else if (d.kind === "audiooutput") outputs.push(entry);
  }
  return { inputs, outputs };
}

// Populate a <select> with a list of devices. Remembers the chosen
// deviceId in localStorage under `storageKey` and re-selects it on
// subsequent renders. Returns the deviceId currently selected ("" for
// "system default").
export function renderDeviceSelect(selectEl, devices, storageKey, { includeDefaultOption = true, defaultLabel = "System default" } = {}) {
  if (!selectEl) return "";
  const prev = selectEl.value || (storageKey ? (localStorage.getItem(storageKey) || "") : "");
  selectEl.innerHTML = "";
  if (includeDefaultOption) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = defaultLabel;
    selectEl.appendChild(opt);
  }
  for (const d of devices) {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label;
    selectEl.appendChild(opt);
  }
  // Restore prior selection if still present
  const still = Array.from(selectEl.options).some((o) => o.value === prev);
  selectEl.value = still ? prev : "";
  return selectEl.value;
}

// Apply output device to an <audio>/<video> element. Returns true on
// success, false if unsupported or deviceId is invalid.
export async function applyOutputDevice(mediaEl, deviceId) {
  if (!mediaEl) return false;
  if (!outputPickerSupported()) return false;
  try {
    await mediaEl.setSinkId(deviceId || "");
    return true;
  } catch (e) {
    return false;
  }
}

// Persist a dropdown's selection to localStorage whenever it changes.
export function persistSelectTo(storageKey, selectEl, onChange) {
  if (!selectEl || !storageKey) return;
  selectEl.addEventListener("change", () => {
    try { localStorage.setItem(storageKey, selectEl.value || ""); } catch (_) {}
    if (typeof onChange === "function") onChange(selectEl.value || "");
  });
}

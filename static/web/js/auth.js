// Shared auth + fetch helpers for the hosted browser mode.
//
// All API calls rely on the SameSite=Lax, HttpOnly session cookie set
// by POST /api/auth/login — there's nothing secret stored in JS, and
// the cookie can't be read from here either (HttpOnly). We also send
// `credentials: "same-origin"` so the cookie is attached for fetches.

export async function apiJson(path, { method = "GET", body = null, signal } = {}) {
  const init = {
    method,
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body != null ? JSON.stringify(body) : undefined,
    signal,
  };
  const resp = await fetch(path, init);
  let json = null;
  try {
    json = await resp.json();
  } catch (_) {}
  if (!resp.ok) {
    const err = new Error((json && (json.error || json.message)) || `HTTP ${resp.status}`);
    err.status = resp.status;
    err.body = json;
    throw err;
  }
  return json;
}

export async function whoami() {
  try {
    return await apiJson("/api/auth/me");
  } catch (_) {
    return { ok: false, authed: false };
  }
}

export async function login(password) {
  return apiJson("/api/auth/login", { method: "POST", body: { password } });
}

export async function logout() {
  try { await apiJson("/api/auth/logout", { method: "POST" }); } catch (_) {}
  location.href = "/login";
}

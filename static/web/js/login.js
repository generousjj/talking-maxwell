import { login, whoami } from "./auth.js";

const form = document.getElementById("loginForm");
const pw = document.getElementById("password");
const err = document.getElementById("err");
const btn = document.getElementById("submitBtn");

// If already authed, kick to app immediately.
(async () => {
  const me = await whoami();
  if (me.authed) location.replace("/");
})();

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  err.textContent = "";
  btn.disabled = true;
  btn.textContent = "Signing in…";
  try {
    await login(pw.value || "");
    location.replace("/");
  } catch (e) {
    if (e.status === 429) {
      const retry = (e.body && e.body.retry_after) || 60;
      err.textContent = `Too many attempts. Try again in ${retry}s.`;
    } else {
      err.textContent = "Wrong password.";
    }
    btn.disabled = false;
    btn.textContent = "Sign in";
    pw.select();
  }
});

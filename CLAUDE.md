# CLAUDE.md

Project-specific instructions for Claude Code. See `ARCHITECTURE.md` for how the
whole system works end to end, and `README.md` for the operator quick-start.

## Keep ARCHITECTURE.md in sync

`ARCHITECTURE.md` is the canonical end-to-end reference for this project (entry
points/CLI flags, `config.yaml`, the conversation/motion/vision/transport
pipeline, the website in both deployment modes, every outbound API call,
deployment targets). When a change affects anything it documents, update the
relevant section(s) in the same turn — don't let the doc drift from the code.

This is enforced, not just requested: a `Stop` hook
(`.claude/hooks/check-architecture-doc.sh`, wired in `.claude/settings.json`)
runs before a turn is allowed to end. It checks whether any project source files
were edited more recently than `ARCHITECTURE.md` itself and, if so, blocks the
stop with a reminder listing what changed.

A few properties of the hook worth knowing so its behavior doesn't look like a
bug:

- It only flags files edited **since the hook last ran** (a timestamp baseline in
  `.git/.architecture-doc-last-check`, not tracked by git) — it does not
  retroactively nag about pre-existing uncommitted work from before the hook was
  installed.
- It won't block twice in a row for the same stale change — the baseline
  advances every time the hook runs, block or not. If you decide a change
  genuinely doesn't warrant a doc update (a typo fix, a test-only tweak, a
  comment), it's fine to say so briefly and move on; you won't get stuck in a
  loop.
- It ignores doc-about-doc/tooling churn (`README.md`, `CLAUDE.md`, `.claude/`,
  `ARCHITECTURE.md` itself, `.git*`, `.venv/`, caches, `data/`, `.env`) so those
  never trigger a false positive.

If you're editing the hook script itself, re-test it directly before trusting
it — a hook that silently does nothing is worse than no hook:

```bash
echo '{}' | bash .claude/hooks/check-architecture-doc.sh
```

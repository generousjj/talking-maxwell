#!/usr/bin/env bash
# Stop hook for talking-maxwell: nudges Claude to update ARCHITECTURE.md
# when project source files have changed more recently than the doc was
# last touched. See CLAUDE.md for the policy this enforces.
#
# State: .git/.architecture-doc-last-check stores the epoch timestamp of
# the last time this hook ran. Only files edited AFTER that timestamp
# count as "new" - this means:
#   (a) the very first run just seeds the baseline instead of blocking,
#       so it never retroactively nags about work that predates the
#       hook's installation, and
#   (b) the same stale change can't trigger a block forever if Claude or
#       the user decides not to update the doc for it - the baseline
#       advances every run regardless of outcome.

set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
cd "$repo_root" || exit 0

state_file=".git/.architecture-doc-last-check"
now="$(date +%s)"

mtime_of() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null; }

# First run ever: seed the baseline to now and exit quietly. Don't
# retroactively nag about work that predates the hook's installation.
if [ ! -f "$state_file" ]; then
  echo "$now" > "$state_file"
  exit 0
fi

last_check="$(cat "$state_file" 2>/dev/null || echo 0)"
case "$last_check" in ''|*[!0-9]*) last_check=0 ;; esac

# Files git currently considers changed (staged, unstaged, or untracked
# and not gitignored) - the working set worth caring about at all.
changed="$(git status --porcelain 2>/dev/null)"

# Always advance the baseline before deciding, so a stale flag can never
# repeat on a later run even if nothing gets edited in response.
echo "$now" > "$state_file"

[ -z "$changed" ] && exit 0

# Paths that don't represent "project source" for doc-sync purposes:
# docs-about-docs, Claude Code's own config, build/venv/cache noise.
exclude_re='^(ARCHITECTURE\.md|README\.md|CLAUDE\.md|\.claude/|\.git|\.venv/|__pycache__/|\.pytest_cache/|data/|\.env|.*\.DS_Store$|.*\.csv$)'

stale=""
while IFS= read -r path; do
  [ -z "$path" ] && continue
  echo "$path" | grep -qE "$exclude_re" && continue
  [ -e "$path" ] || continue   # skip deletions, nothing to stat
  mt="$(mtime_of "$path")"
  [ -n "$mt" ] && [ "$mt" -gt "$last_check" ] && stale="$stale
$path"
done <<< "$(echo "$changed" | awk '{print substr($0,4)}')"

stale="$(echo "$stale" | sed '/^$/d')"
[ -z "$stale" ] && exit 0

reason="Project files changed since ARCHITECTURE.md was last reviewed, and ARCHITECTURE.md itself hasn't been touched to match:
$stale

Before finishing this turn: check whether these changes affect anything ARCHITECTURE.md documents (entry points/CLI flags, config.yaml fields, the conversation/motion/vision/transport pipeline, website routes and pages, outbound API calls, deployment). If so, update the relevant section(s) now. If the changes are cosmetic or don't affect the documented architecture (a typo fix, a test-only tweak, a comment), it's fine to say so briefly and finish without editing the doc - just don't skip the check silently."

jq -n --arg reason "$reason" \
  '{decision: "block", reason: $reason, systemMessage: "ARCHITECTURE.md check: some changed files are newer than the doc — asking Claude to review before finishing."}'
exit 0

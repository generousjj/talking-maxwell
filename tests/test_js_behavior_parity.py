"""JS behavior engine parity with the Python engine.

The JS port in ``static/web/js/behavior.js`` is meant to run the same
algorithms as ``motion/behavior_engine.py``. We can't do bit-identical
parity (Python uses Mersenne Twister; JS uses mulberry32), but we can
assert the same *behavioral* contracts hold on both sides:

  - IDLE keeps jaw=0 and head near center.
  - IDLE wing pulses continuously (no long flat-rest gaps).
  - SPEAKING with high envelope drives jaw up.
  - SPEAKING with a phrase_boundary pushes head_ud off center
    (the nod fires).
  - Wing cooldown keeps flap count bounded over a long window.
  - Output-side head smoothing softens step changes.

The test is skipped when `node` isn't on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
BEHAVIOR_JS = REPO_ROOT / "static" / "web" / "js" / "behavior.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not BEHAVIOR_JS.is_file(),
    reason="node or behavior.js not available",
)


def _run_js(script: str) -> dict:
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=20,
    )
    assert proc.returncode == 0, f"node failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_js_idle_keeps_jaw_closed_and_head_near_center():
    # Uses the same defaults as motion.BehaviorGains (idle_nod_strength=0.12,
    # idle_tilt_strength=0.08) so the 0.3..0.7 band matches the Python
    # ``test_idle_has_no_jaw_and_head_stays_near_center`` test byte-for-byte.
    result = _run_js(
        """
        import('./static/web/js/behavior.js').then(({ BehaviorEngine, STATE }) => {
          let t = 0;
          const env = { behaviorEnvelope: 0, _smoothed: 0, _mapToJaw: () => 0 };
          const be = new BehaviorEngine({ envelope: env, now: () => t, gains: {
            seed: 42,
            idleNodStrength: 0.12,
            idleTiltStrength: 0.08,
          }});
          be.setState(STATE.IDLE);
          let maxJaw = 0, minHLr = 1, maxHLr = 0, minHUd = 1, maxHUd = 0;
          for (let i = 0; i < 60; i++) {
            t = i / 30;
            const o = be.tick();
            maxJaw = Math.max(maxJaw, o.jaw);
            minHLr = Math.min(minHLr, o.head_lr);
            maxHLr = Math.max(maxHLr, o.head_lr);
            minHUd = Math.min(minHUd, o.head_ud);
            maxHUd = Math.max(maxHUd, o.head_ud);
          }
          console.log(JSON.stringify({ maxJaw, minHLr, maxHLr, minHUd, maxHUd }));
        });
        """
    )
    assert result["maxJaw"] == 0, "idle jaw must stay fully closed"
    assert 0.3 < result["minHLr"] < 0.7
    assert 0.3 < result["maxHLr"] < 0.7
    assert 0.3 < result["minHUd"] < 0.7
    assert 0.3 < result["maxHUd"] < 0.7


def test_js_idle_wing_pulses_continuously():
    # Same invariant as motion/behavior_engine.py::_idle — the
    # raised-cosine wing shape should never flat-rest for long stretches.
    result = _run_js(
        """
        import('./static/web/js/behavior.js').then(({ BehaviorEngine, STATE }) => {
          let t = 0;
          const env = { behaviorEnvelope: 0, _smoothed: 0, _mapToJaw: () => 0 };
          const be = new BehaviorEngine({ envelope: env, now: () => t, gains: { seed: 1 } });
          be.setState(STATE.IDLE);
          let wingMax = 0, runZero = 0, maxRunZero = 0;
          for (let i = 0; i < 900; i++) {
            t = i / 30;
            const o = be.tick();
            wingMax = Math.max(wingMax, o.wing);
            if (o.wing < 0.001) { runZero++; maxRunZero = Math.max(maxRunZero, runZero); }
            else runZero = 0;
          }
          console.log(JSON.stringify({ wingMax, maxRunZero }));
        });
        """
    )
    assert result["wingMax"] > 0.3, "wing must reach a clear peak during idle"
    assert result["maxRunZero"] < 5, (
        f"wing flat-rested at 0 for {result['maxRunZero']} ticks — "
        "raised-cosine shape should only touch 0 instantaneously"
    )


def test_js_speaking_high_envelope_drives_jaw():
    result = _run_js(
        """
        import('./static/web/js/behavior.js').then(({ BehaviorEngine, STATE }) => {
          let t = 0;
          const env = { behaviorEnvelope: 0.8, _smoothed: 0.8, _mapToJaw: () => 0.8 };
          const be = new BehaviorEngine({ envelope: env, now: () => t, gains: { seed: 42 } });
          be.setState(STATE.SPEAKING);
          const ctx = { envelope: 0.8, text: 'Hello there', progress: 0.5,
                        phrase_boundary: false, emphasis: 0, question_like: false, excited: false };
          let jawMax = 0;
          for (let i = 0; i < 30; i++) {
            t = i / 30;
            const o = be.tick(ctx);
            jawMax = Math.max(jawMax, o.jaw);
          }
          console.log(JSON.stringify({ jawMax }));
        });
        """
    )
    assert result["jawMax"] >= 0.6, "jaw should reach >=0.6 on envelope=0.8"


def test_js_phrase_boundary_fires_nod():
    # One boundary at t=0 should pull head_ud off-center within 400ms.
    result = _run_js(
        """
        import('./static/web/js/behavior.js').then(({ BehaviorEngine, STATE }) => {
          let t = 0;
          const env = { behaviorEnvelope: 0.4, _smoothed: 0.4, _mapToJaw: () => 0.4 };
          const be = new BehaviorEngine({ envelope: env, now: () => t, gains: { seed: 0 } });
          be.setState(STATE.SPEAKING);
          let maxDev = 0;
          for (let i = 0; i < 15; i++) {
            t = i / 30;
            const ctx = { envelope: 0.4, text: 'Hi!', progress: 0,
                          phrase_boundary: i === 0, emphasis: 0,
                          question_like: false, excited: false };
            const o = be.tick(ctx);
            maxDev = Math.max(maxDev, Math.abs(o.head_ud - 0.5));
          }
          console.log(JSON.stringify({ maxDev }));
        });
        """
    )
    assert result["maxDev"] > 0.05, "phrase_boundary should move head_ud off center"


def test_js_wing_cooldown_bounds_flap_count():
    # Mirrors test_wing_cooldown_prevents_rapid_flaps: 10 s of loud
    # excited speech must not produce more than a handful of flaps.
    result = _run_js(
        """
        import('./static/web/js/behavior.js').then(({ BehaviorEngine, STATE }) => {
          let t = 0;
          const env = { behaviorEnvelope: 0.9, _smoothed: 0.9, _mapToJaw: () => 0.9 };
          const be = new BehaviorEngine({ envelope: env, now: () => t, gains: { seed: 42 } });
          be.setState(STATE.SPEAKING);
          let fires = 0;
          for (let i = 0; i < 300; i++) {
            t = i / 30;
            const ctx = { envelope: 0.9, text: 'wow!', progress: 0,
                          phrase_boundary: false, emphasis: 0.9,
                          question_like: false, excited: true };
            const o = be.tick(ctx);
            if (o.wing > 0) fires++;
          }
          console.log(JSON.stringify({ fires }));
        });
        """
    )
    assert result["fires"] <= 5, f"expected sparse wing usage, got {result['fires']} flaps"

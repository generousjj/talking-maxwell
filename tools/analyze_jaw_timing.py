"""Simulate the full jaw pipeline against a WAV and report timing + PWM.

Same math the live code runs. Outputs a CSV and a Markdown summary so we
can see per-20ms how much the jaw actually moves in microseconds.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
import motion.envelope as env_mod
from motion.envelope import EnvelopeFollower
from motion.models import JawCalibration


def main(wav_path: str) -> None:
    w = wave.open(wav_path)
    sr = w.getframerate()
    buf = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()

    frame_samples = int(sr * 0.020)
    rms = np.array(
        [
            np.sqrt(np.mean(buf[i : i + frame_samples] ** 2))
            for i in range(0, len(buf) - frame_samples, frame_samples)
        ]
    )

    # Deterministic clock so peak-hold behaves like a 20ms scheduler would.
    t = {"now": 0.0}
    env_mod.time.monotonic = lambda: t["now"]

    config = load_config("config.example.yaml")
    cal = config.motion.jaw
    follower = EnvelopeFollower(sample_rate=sr, calibration=cal, frame_ms=20.0)

    jaw_min = int(config.bottango.serial.jaw.min_pwm)
    jaw_max = int(config.bottango.serial.jaw.max_pwm)
    jaw_span = jaw_max - jaw_min
    print(
        f"clip: {wav_path}  sr={sr}  frames={len(rms)}  dur={len(rms)*0.020:.2f}s"
    )
    print(
        f"jaw PWM hardware range: {jaw_min}..{jaw_max}μs  span={jaw_span}μs  "
        f"(typical hobby servo full range is 500-2500μs = 2000μs, so this "
        f"mouth uses {jaw_span/2000*100:.1f}% of a servo's full travel)"
    )
    print(
        f"envelope config: gain={cal.gain} attack={cal.attack} release={cal.release} "
        f"floor={cal.floor} ceiling={cal.ceiling}\n"
    )

    rows = []
    prev_pwm = None
    abs_changes = []
    for i, r in enumerate(rms):
        t["now"] = i * 0.020
        jaw_in_range = follower.process_rms(float(r))  # in [floor, ceiling]
        # The pipeline un-maps this back to [0,1] before the behavior engine
        # multiplies / sends it. Replicate that:
        if cal.ceiling - cal.floor > 1e-6:
            jaw_norm = max(
                0.0, min(1.0, (jaw_in_range - cal.floor) / (cal.ceiling - cal.floor))
            )
        else:
            jaw_norm = jaw_in_range
        compressed = int(round(jaw_norm * 8192))
        pwm = jaw_min + (compressed / 8192.0) * jaw_span
        rows.append((i * 0.020, float(r), jaw_norm, int(round(pwm))))
        if prev_pwm is not None:
            abs_changes.append(abs(pwm - prev_pwm))
        prev_pwm = pwm

    t_arr = np.array([r[0] for r in rows])
    rms_arr = np.array([r[1] for r in rows])
    jaw_arr = np.array([r[2] for r in rows])
    pwm_arr = np.array([r[3] for r in rows])

    # Print a sampled table every 100ms so we can eyeball sync.
    print(
        f"{'t(s)':>6} {'audio_rms':>9} {'jaw_norm':>9} {'pwm(μs)':>9} {'~pct_open':>10}"
    )
    for i in range(0, len(rows), 5):  # 5 frames = 100ms
        t_s, r, jn, p = rows[i]
        pct = (p - jaw_min) / jaw_span * 100 if jaw_span > 0 else 0
        bar = "#" * int(pct / 5)
        print(f"{t_s:6.2f} {r:9.3f} {jn:9.3f} {p:9d} {pct:9.1f}%  {bar}")

    print()
    print("=== summary ===")
    print(
        f"audio RMS:   min={rms_arr.min():.3f}  p50={np.percentile(rms_arr,50):.3f}  "
        f"p95={np.percentile(rms_arr,95):.3f}  max={rms_arr.max():.3f}"
    )
    print(
        f"jaw norm:    min={jaw_arr.min():.3f}  p50={np.percentile(jaw_arr,50):.3f}  "
        f"p95={np.percentile(jaw_arr,95):.3f}  max={jaw_arr.max():.3f}"
    )
    print(
        f"PWM (μs):    min={pwm_arr.min()}  p50={int(np.percentile(pwm_arr,50))}  "
        f"p95={int(np.percentile(pwm_arr,95))}  max={pwm_arr.max()}   "
        f"peak-to-peak swing={pwm_arr.max()-pwm_arr.min()}μs"
    )
    if abs_changes:
        mean_change = np.mean(abs_changes)
        print(
            f"mean |ΔPWM| per 20ms tick: {mean_change:.1f}μs  "
            f"(small = little motion)"
        )

    out = ROOT / "tools" / "jaw_trace.csv"
    with out.open("w") as fh:
        fh.write("t_s,rms,jaw_norm,pwm_us,pct_open\n")
        for t_s, r, jn, p in rows:
            pct = (p - jaw_min) / jaw_span * 100 if jaw_span > 0 else 0
            fh.write(f"{t_s:.3f},{r:.4f},{jn:.4f},{p},{pct:.2f}\n")
    print(f"\nFull per-20ms trace written to {out}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/maxwell_short.wav"
    main(path)

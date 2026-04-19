"""CLI entry point.

Usage:

    python -m app.cli --mode typed --backend mock
    python -m app.cli --mode typed --backend bottango --config config.yaml
    python -m app.cli --mode live --backend mock
    python -m app.cli --mode replay --wav some_audio.wav --backend mock
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from conversation.llm import build_llm_provider, StubLLM
from conversation.stt import build_stt_provider, LocalStubSTT
from conversation.tts import build_tts_provider, SineStubTTS
from motion.state_machine import ConversationStateMachine
from transport.bottango_backend import BottangoBackend, BottangoIdentifiers
from transport.bottango_serial_backend import (
    BottangoSerialBackend,
    BottangoServoConfig,
)
from transport.mock_backend import MockBackend

from .config import AppConfig, load_config
from .pipeline import ConversationPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maxwell",
        description="Maxwell animatronic chatbot prototype.",
    )
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--env-file", default=None, help="Path to .env file.")
    parser.add_argument(
        "--mode",
        choices=("typed", "live", "replay"),
        default=None,
        help="typed=typed-text, live=mic conversation, replay=drive motion from a WAV file",
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "bottango", "bottango_serial", "bottango_http"),
        default=None,
        help=(
            "Motion backend to use. `bottango` honors config.bottango.transport "
            "(defaults to `serial`). `bottango_serial` forces the USB-serial "
            "backend, `bottango_http` forces the legacy HTTP transport."
        ),
    )
    parser.add_argument(
        "--serial-port",
        default=None,
        help="Override auto-detected serial port (e.g. /dev/cu.usbmodem1101).",
    )
    parser.add_argument(
        "--playback-device",
        default=None,
        help=(
            "Audio output device. Int index or name substring "
            "(e.g. 'MacBook Pro Speakers'). Defaults to config audio.playback_device."
        ),
    )
    parser.add_argument("--wav", default=None, help="WAV file for replay mode.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process a single turn and exit (useful for scripting).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render a matplotlib plot of motion channels at exit (mock backend only).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path to write motion channels for offline inspection.",
    )
    parser.add_argument(
        "--safe-providers",
        action="store_true",
        help="Force stub providers (no network calls). Useful for quick local smoke tests.",
    )
    parser.add_argument("--text", default=None, help="Speak this text and exit (typed mode).")
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_backend(config: AppConfig, args: argparse.Namespace):
    backend_name = args.backend or config.backend

    use_bottango = backend_name in ("bottango", "bottango_serial", "bottango_http")
    if not use_bottango and config.bottango.enabled:
        use_bottango = True

    if use_bottango:
        if backend_name == "bottango_http":
            transport = "http"
        elif backend_name == "bottango_serial":
            transport = "serial"
        else:
            transport = (config.bottango.transport or "serial").lower()
        if transport == "serial":
            s = config.bottango.serial
            return BottangoSerialBackend(
                port=args.serial_port or s.port,
                baud=s.baud,
                allow_auto_detect=s.auto_detect,
                handshake_timeout_s=s.handshake_timeout_s,
                command_timeout_s=s.command_timeout_s,
                compressed_signal_max=s.compressed_signal_max,
                min_delta_for_send=s.min_delta_for_send,
                jaw_min_delta=s.jaw_min_delta,
                jaw_min_send_interval_s=s.jaw_min_send_interval_s,
                jaw=_servo(s.jaw),
                head_lr=_servo(s.head_lr),
                head_ud=_servo(s.head_ud),
                wing=_servo(s.wing),
            )
        if transport == "http":
            ids = BottangoIdentifiers(
                jaw=config.bottango.jaw_identifier,
                head_lr=config.bottango.head_lr_identifier,
                head_ud=config.bottango.head_ud_identifier,
                wing=config.bottango.wing_identifier,
            )
            return BottangoBackend(
                base_url=config.bottango.base_url,
                path_template=config.bottango.path_template,
                identifiers=ids,
                request_timeout_s=config.bottango.request_timeout_s,
                value_scale=config.bottango.value_scale,
                health_path=config.bottango.health_path,
            )
        raise ValueError(
            f"Unknown bottango.transport={transport!r}; use 'serial' or 'http'."
        )

    csv_path = Path(args.csv) if args.csv else (
        Path(config.logging.motion_csv_path) if config.logging.motion_csv_path else None
    )
    return MockBackend(
        csv_path=csv_path,
        log_every_n=config.logging.log_every_n_motion,
        plot_after=args.plot or config.logging.plot_after,
    )


def _servo(cfg) -> BottangoServoConfig:
    return BottangoServoConfig(
        pin=int(cfg.pin),
        min_pwm=int(cfg.min_pwm),
        max_pwm=int(cfg.max_pwm),
        max_pwm_per_sec=int(cfg.max_pwm_per_sec),
        starting_pwm=cfg.starting_pwm,
        invert=bool(getattr(cfg, "invert", False)),
    )


def _build_providers(config: AppConfig, args: argparse.Namespace):
    if args.safe_providers:
        return LocalStubSTT(), StubLLM(), SineStubTTS()
    try:
        stt = build_stt_provider(config.providers.stt, model=config.providers.stt_model)
    except Exception as exc:  # noqa: BLE001
        logging.warning("falling back to LocalStubSTT: %s", exc)
        stt = LocalStubSTT()
    try:
        llm = build_llm_provider(config.providers.llm, model=config.providers.llm_model)
    except Exception as exc:  # noqa: BLE001
        logging.warning("falling back to StubLLM: %s", exc)
        llm = StubLLM()
    try:
        tts = build_tts_provider(
            config.providers.tts,
            model=config.providers.tts_model,
            voice=config.providers.tts_voice,
            instructions=config.providers.tts_instructions,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("falling back to SineStubTTS: %s", exc)
        tts = SineStubTTS()
    return stt, llm, tts


async def _run(config: AppConfig, args: argparse.Namespace) -> int:
    mode = args.mode or config.mode
    backend = _build_backend(config, args)
    stt, llm, tts = _build_providers(config, args)
    state_machine = ConversationStateMachine()

    pipeline = ConversationPipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        backend=backend,
        state_machine=state_machine,
        jaw_calibration=config.motion.jaw,
        behavior_gains=config.motion.behavior,
        rate_hz=config.motion.rate_hz,
        personality=config.personality,
        audio_frame_ms=config.audio.frame_ms,
        audio_input_sample_rate=config.audio.input_sample_rate,
        playback_device=args.playback_device or config.audio.playback_device,
        mic_max_s=config.audio.max_utterance_s,
        mic_silence_threshold=config.audio.vad_threshold,
        mic_silence_hangover_s=config.audio.vad_silence_hangover_s,
    )

    async with pipeline:
        if mode == "replay":
            if not args.wav:
                print("replay mode requires --wav <path>", file=sys.stderr)
                return 2
            await pipeline.speak_wav_file(args.wav, text="(replay)")
            return 0

        if mode == "typed":
            if args.text:
                await pipeline.handle_typed_turn(args.text)
                return 0
            print("Typed-text mode. Type a message and hit enter. Ctrl-C to exit.")
            while True:
                try:
                    line = await asyncio.to_thread(sys.stdin.readline)
                except (KeyboardInterrupt, EOFError):
                    break
                if not line:
                    break
                user_text = line.strip()
                if not user_text:
                    continue
                if user_text.lower() in {"quit", "exit"}:
                    break
                reply = await pipeline.handle_typed_turn(user_text)
                print(f"Maxwell: {reply}")
                if args.once:
                    break
            return 0

        if mode == "live":
            print(
                "Live conversation mode. Speak into the mic; the bird replies.\n"
                "Press Ctrl-C to exit."
            )
            while True:
                try:
                    user_text, reply = await pipeline.handle_live_turn()
                except KeyboardInterrupt:
                    break
                if user_text:
                    print(f"You: {user_text}")
                if reply:
                    print(f"Maxwell: {reply}")
                if args.once:
                    break
            return 0

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config, env_file=args.env_file)
    _configure_logging(config.logging.level)
    try:
        return asyncio.run(_run(config, args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

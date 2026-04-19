"""Text-to-speech providers.

The preferred provider is OpenAI's ``audio.speech`` endpoint, which sounds
pleasant and returns clean WAV bytes we can immediately analyze for the jaw
envelope. A macOS-friendly fallback shells out to the built-in ``say``
command so the prototype runs fully offline.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .audio import AudioBuffer, decode_wav_bytes, load_wav_file

log = logging.getLogger(__name__)


class TTSProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    async def synthesize(self, text: str) -> AudioBuffer:
        """Synthesize the given text to a mono AudioBuffer."""


@dataclass
class OpenAITTS(TTSProvider):
    """OpenAI TTS provider using the ``audio.speech`` endpoint.

    ``model`` defaults to ``gpt-4o-mini-tts`` (fast, pleasant). ``voice`` can
    be any voice the endpoint supports. ``instructions`` is an optional
    free-form style hint that steers delivery (pitch, cadence, mood) — the
    typical way to coax a "parrot character" out of the neutral voices.
    Both ``voice`` and ``instructions`` are intentionally mutable so the
    web UI can swap them live without reconnecting.
    """

    name: str = "openai"
    model: str = "gpt-4o-mini-tts"
    voice: str = "ballad"
    instructions: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    response_format: str = "wav"
    style_prefix: str = ""

    async def synthesize(self, text: str) -> AudioBuffer:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is not installed. Run `pip install openai` "
                "or switch to the `macos_say` / `stub` TTS provider."
            ) from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set; "
                "cannot call OpenAI TTS."
            )

        client = OpenAI(api_key=api_key)
        input_text = f"{self.style_prefix}{text}" if self.style_prefix else text

        def _run() -> bytes:
            kwargs = {
                "model": self.model,
                "voice": self.voice,
                "input": input_text,
                "response_format": self.response_format,
            }
            if self.instructions:
                kwargs["instructions"] = self.instructions
            resp = client.audio.speech.create(**kwargs)
            if hasattr(resp, "read"):
                return resp.read()  # type: ignore[no-any-return]
            if hasattr(resp, "content"):
                return resp.content  # type: ignore[no-any-return]
            return bytes(resp)  # type: ignore[arg-type]

        wav_bytes = await asyncio.to_thread(_run)
        return decode_wav_bytes(wav_bytes)


@dataclass
class MacOSSayTTS(TTSProvider):
    """macOS-native fallback that shells out to ``say -o out.wav``."""

    name: str = "macos_say"
    voice: Optional[str] = None
    rate_wpm: int = 190
    sample_rate: int = 22050

    async def synthesize(self, text: str) -> AudioBuffer:
        if shutil.which("say") is None:
            raise RuntimeError(
                "The macOS `say` command is unavailable. Install it or pick a "
                "different TTS provider."
            )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "speech.wav"
            cmd = [
                "say",
                "-o",
                str(out),
                "--file-format=WAVE",
                f"--data-format=LEF32@{self.sample_rate}",
                "-r",
                str(self.rate_wpm),
            ]
            if self.voice:
                cmd.extend(["-v", self.voice])
            cmd.append(text)

            def _run() -> None:
                subprocess.run(cmd, check=True)

            await asyncio.to_thread(_run)
            return load_wav_file(str(out))


@dataclass
class SineStubTTS(TTSProvider):
    """Placeholder TTS that generates a short modulated tone per word.

    Useful in CI and on machines without `say` or OpenAI available. The jaw
    envelope still "moves" so motion pipelines can be exercised end-to-end.
    """

    name: str = "sine_stub"
    sample_rate: int = 22050
    words_per_second: float = 2.6

    async def synthesize(self, text: str) -> AudioBuffer:
        import numpy as np

        duration = max(0.6, len(text.split()) / max(0.5, self.words_per_second))
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        carrier = np.sin(2 * np.pi * 180 * t)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
        samples = (0.35 * carrier * envelope).astype(np.float32)
        return AudioBuffer(samples=samples, sample_rate=self.sample_rate)


def build_tts_provider(
    name: str,
    *,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    instructions: Optional[str] = None,
) -> TTSProvider:
    name = (name or "").lower()
    if name in ("openai", "gpt"):
        return OpenAITTS(
            model=model or "gpt-4o-mini-tts",
            voice=voice or "ballad",
            instructions=instructions or "",
        )
    if name in ("macos_say", "say", "macos"):
        return MacOSSayTTS(voice=voice)
    if name in ("sine", "stub", "sine_stub", "none"):
        return SineStubTTS()
    raise ValueError(f"Unknown TTS provider: {name}")

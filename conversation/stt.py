"""Speech-to-text providers.

The default provider uses OpenAI's Whisper transcription API. A local stub is
provided for development without network/credentials; it simply tells the
caller to either switch to typed-text mode or configure a real STT provider.
"""

from __future__ import annotations

import abc
import io
import logging
import os
from dataclasses import dataclass
from typing import Optional

from .audio import AudioBuffer, encode_wav_bytes

log = logging.getLogger(__name__)


class STTProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    async def transcribe(self, audio: AudioBuffer, *, language: Optional[str] = None) -> str:
        """Transcribe the given audio buffer to text."""


@dataclass
class OpenAIWhisperSTT(STTProvider):
    """OpenAI Whisper API-based STT provider.

    Requires ``OPENAI_API_KEY`` in the environment. Uses the modern
    ``openai`` Python client if available.
    """

    name: str = "openai_whisper"
    model: str = "whisper-1"
    api_key_env: str = "OPENAI_API_KEY"

    async def transcribe(self, audio: AudioBuffer, *, language: Optional[str] = None) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is not installed. Run `pip install openai` "
                "or select a different STT provider."
            ) from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set; "
                "cannot call OpenAI Whisper."
            )

        client = OpenAI(api_key=api_key)
        wav_bytes = encode_wav_bytes(audio)
        buf = io.BytesIO(wav_bytes)
        buf.name = "speech.wav"

        # TODO(provider-tuning): expose prompt/temperature once needed.
        kwargs = {"model": self.model, "file": buf}
        if language:
            kwargs["language"] = language

        import asyncio

        def _run() -> str:
            resp = client.audio.transcriptions.create(**kwargs)
            return getattr(resp, "text", "").strip()

        return await asyncio.to_thread(_run)


@dataclass
class LocalStubSTT(STTProvider):
    """Explicit stub that fails loudly; useful for mock mode without keys."""

    name: str = "local_stub"

    async def transcribe(self, audio: AudioBuffer, *, language: Optional[str] = None) -> str:
        raise RuntimeError(
            "LocalStubSTT cannot actually transcribe audio. "
            "Switch to typed-text mode (`--mode typed`) or configure an STT "
            "provider such as `openai_whisper` in config.yaml."
        )


def build_stt_provider(name: str, *, model: Optional[str] = None) -> STTProvider:
    name = (name or "").lower()
    if name in ("openai", "openai_whisper", "whisper"):
        return OpenAIWhisperSTT(model=model or "whisper-1")
    if name in ("stub", "local", "local_stub", "none"):
        return LocalStubSTT()
    raise ValueError(f"Unknown STT provider: {name}")

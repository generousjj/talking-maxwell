"""LLM providers for generating parrot chatbot replies."""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


DEFAULT_PERSONALITY = (
    "You are Maxwell, a cheerful animatronic parrot. "
    "Respond in one or two short sentences, playful and warm, with a bird-like flair. "
    "Occasionally add a short 'squawk!', 'polly!' or 'awk!' interjection, "
    "but do not overdo it. Avoid emojis and avoid explaining that you are an AI."
)


class LLMProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    async def reply(self, user_text: str, *, personality: str = DEFAULT_PERSONALITY) -> str:
        """Generate a short chatbot reply to the given user utterance."""


@dataclass
class OpenAILLM(LLMProvider):
    """OpenAI chat completions provider.

    Defaults to ``gpt-4o-mini`` which is the cheapest fast model suitable for
    short replies. Configure a different model in config.yaml if desired.
    """

    name: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    max_output_tokens: int = 160
    temperature: float = 0.8

    async def reply(self, user_text: str, *, personality: str = DEFAULT_PERSONALITY) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is not installed. Run `pip install openai` "
                "or select a different LLM provider."
            ) from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set; "
                "cannot call OpenAI."
            )

        client = OpenAI(api_key=api_key)

        def _run() -> str:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": personality},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
            )
            return resp.choices[0].message.content.strip()

        return await asyncio.to_thread(_run)


@dataclass
class StubLLM(LLMProvider):
    """Fully offline parrot responder used as a fallback."""

    name: str = "stub"
    seed: Optional[int] = None

    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    async def reply(self, user_text: str, *, personality: str = DEFAULT_PERSONALITY) -> str:
        lines = [
            f"Squawk! {user_text[:60]}? Polly wants a cracker!",
            "Awk! That sounds fun. Tell me more!",
            "Polly says: hello, friend! Feathers fluffed and ready.",
            "Squawk squawk! I'm listening with both ears.",
            "Polly approves! Very interesting indeed.",
        ]
        return self._rng.choice(lines)


def build_llm_provider(
    name: str,
    *,
    model: Optional[str] = None,
) -> LLMProvider:
    name = (name or "").lower()
    if name in ("openai", "gpt", "chat"):
        return OpenAILLM(model=model or "gpt-4o-mini")
    if name in ("stub", "local", "offline", "none"):
        return StubLLM()
    raise ValueError(f"Unknown LLM provider: {name}")

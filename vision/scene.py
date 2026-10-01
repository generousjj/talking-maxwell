"""Scene understanding: a single camera frame in, a spoken-style description out.

This is the **slow, paid, on-demand** vision pipeline — the counterpart to the free
local face tracker. It sends one JPEG frame to a vision-capable chat model and gets
back a short sentence Maxwell can say aloud. It is only ever called when triggered
(operator button or a spoken request), so it has no idle cost.

Mirrors ``conversation.llm``: a ``VisionProvider`` ABC, an ``OpenAIVisionProvider``
(reusing the ``openai`` SDK + ``OPENAI_API_KEY`` already configured, defaulting to
the cheapest vision-capable model), a ``StubVisionProvider`` for offline tests, and
a ``build_vision_provider`` factory. ``cv2`` is imported lazily for JPEG encoding.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def encode_frame_jpeg(frame: np.ndarray, *, quality: int = 80) -> bytes:
    """Encode a BGR frame to JPEG bytes. Raises if opencv isn't available."""
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "opencv-python is required to encode frames for scene understanding."
        ) from exc
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to JPEG-encode frame")
    return buf.tobytes()


class VisionProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    async def describe(self, frame: np.ndarray, *, prompt: str) -> str:
        """Return a short natural-language description of the frame."""


@dataclass
class OpenAIVisionProvider(VisionProvider):
    """OpenAI vision via chat completions with an inline image.

    Defaults to ``gpt-4o-mini``, which accepts image input and is the cheapest
    model suitable for a one-sentence description — matching the cost-minded model
    choice in :class:`conversation.llm.OpenAILLM`. ``max_output_tokens`` caps reply
    length as a further cost rail.
    """

    name: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    max_output_tokens: int = 120
    temperature: float = 0.7

    async def describe(self, frame: np.ndarray, *, prompt: str) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is not installed. Run `pip install openai` "
                "or select scene_provider: stub."
            ) from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set; "
                "cannot call OpenAI vision."
            )

        jpeg = encode_frame_jpeg(frame)
        b64 = base64.b64encode(jpeg).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64}"
        client = OpenAI(api_key=api_key)

        def _run() -> str:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
            )
            return (resp.choices[0].message.content or "").strip()

        return await asyncio.to_thread(_run)


@dataclass
class StubVisionProvider(VisionProvider):
    """Offline stand-in: reports frame dimensions instead of calling any API."""

    name: str = "stub"

    async def describe(self, frame: np.ndarray, *, prompt: str) -> str:
        h, w = frame.shape[:2]
        return f"(stub vision) I see a {w}x{h} frame but can't really look right now."


def build_vision_provider(
    name: str,
    *,
    model: Optional[str] = None,
    max_output_tokens: int = 120,
) -> VisionProvider:
    name = (name or "").lower()
    if name in ("openai", "gpt", "vision"):
        return OpenAIVisionProvider(
            model=model or "gpt-4o-mini", max_output_tokens=max_output_tokens
        )
    if name in ("stub", "offline", "none"):
        return StubVisionProvider()
    raise ValueError(f"Unknown vision provider: {name}")

"""The contract every provider adapter implements.

Adapters translate. They do not retry, do not fall back, do not price and do
not aggregate: that is the gateway's job, and keeping it in one place is what
stops each provider from growing its own subtly different policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from llm_gateway.audio import ProviderTranscriptionResponse, TranscriptionRequest
from llm_gateway.contracts import LLMRequest
from llm_gateway.media import (
    ImageRequest,
    ProviderImageResponse,
    ProviderVideoResponse,
    VideoRequest,
)
from llm_gateway.tools import ProviderToolCall
from llm_gateway.usage import TokenUsage


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """One provider reply, already normalised but not yet interpreted."""

    output_text: str | None
    usage: TokenUsage
    finish_reason: str | None = None
    model_used: str | None = None
    """Set when the provider reports a different model than the one requested."""
    tool_calls: tuple[ProviderToolCall, ...] = ()
    """Calls the model made, still carrying the arguments exactly as sent."""


@runtime_checkable
class ProviderAdapter(Protocol):
    """Implemented once per provider."""

    name: str

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        """Perform exactly one call. Raise a typed error; never return one."""
        ...


@runtime_checkable
class ImageProviderAdapter(Protocol):
    """Provider adapter capable of generating or editing images."""

    name: str

    async def generate_image(self, request: ImageRequest, *, model: str) -> ProviderImageResponse:
        """Perform exactly one image call. Raise a typed error; never return one."""
        ...


@runtime_checkable
class VideoProviderAdapter(Protocol):
    """Provider adapter capable of generating video."""

    name: str

    async def generate_video(self, request: VideoRequest, *, model: str) -> ProviderVideoResponse:
        """Perform exactly one video call. Raise a typed error; never return one."""
        ...


@runtime_checkable
class AudioProviderAdapter(Protocol):
    """Provider adapter capable of speech-to-text."""

    name: str

    async def transcribe(
        self, request: TranscriptionRequest, *, model: str
    ) -> ProviderTranscriptionResponse:
        """Perform exactly one transcription call."""
        ...

"""The contract every provider adapter implements.

Adapters translate. They do not retry, do not fall back, do not price and do
not aggregate: that is the gateway's job, and keeping it in one place is what
stops each provider from growing its own subtly different policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from llm_gateway.contracts import LLMRequest
from llm_gateway.usage import TokenUsage


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """One provider reply, already normalised but not yet interpreted."""

    output_text: str | None
    usage: TokenUsage
    finish_reason: str | None = None
    model_used: str | None = None
    """Set when the provider reports a different model than the one requested."""


@runtime_checkable
class ProviderAdapter(Protocol):
    """Implemented once per provider."""

    name: str

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        """Perform exactly one call. Raise a typed error; never return one."""
        ...

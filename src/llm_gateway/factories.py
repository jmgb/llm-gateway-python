"""Convenience constructors. Every one of them is optional.

An application may build its own SDK clients and register adapters by hand;
these helpers exist so that the common case is not seven copies of the same
twenty lines. They still take the credential as an explicit argument — the
package never reads an environment variable, so a key is always something the
caller decided to hand over.

Model routing lives here rather than in the adapters: which families exist is
knowledge that ages, and it should age in one obvious place.
"""

from __future__ import annotations

from typing import Any

from llm_gateway.errors import ProviderNotInstalled
from llm_gateway.providers.gemini import GeminiAdapter
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openai import OpenAIAdapter
from llm_gateway.registry import ProviderRegistry

OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
GEMINI_MODEL_PREFIXES = ("gemini", "models/gemini")
GROQ_MODEL_PREFIXES = ("llama", "mixtral", "gemma", "qwen", "kimi", "groq/")


def _require_key(api_key: str) -> None:
    if not api_key or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")


def create_openai_client(*, api_key: str, base_url: str | None = None) -> Any:
    """Build an ``AsyncOpenAI``. Also serves OpenRouter, via ``base_url``."""
    _require_key(api_key)
    try:
        from openai import AsyncOpenAI
    except ImportError as error:
        raise ProviderNotInstalled.for_provider("openai") from error
    if base_url:
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    return AsyncOpenAI(api_key=api_key)


def create_gemini_client(*, api_key: str) -> Any:
    """Build a ``google.genai`` client."""
    _require_key(api_key)
    try:
        from google import genai
    except ImportError as error:
        raise ProviderNotInstalled.for_provider("gemini") from error
    return genai.Client(api_key=api_key)


def create_groq_client(*, api_key: str) -> Any:
    """Build an ``AsyncGroq``."""
    _require_key(api_key)
    try:
        from groq import AsyncGroq
    except ImportError as error:
        raise ProviderNotInstalled.for_provider("groq") from error
    return AsyncGroq(api_key=api_key)


def build_registry(
    *,
    openai_client: Any | None = None,
    gemini_client: Any | None = None,
    groq_client: Any | None = None,
    extra_openai_prefixes: tuple[str, ...] = (),
) -> ProviderRegistry:
    """Register the adapters for the clients the application supplies."""
    registry = ProviderRegistry()
    if openai_client is not None:
        registry.register(
            OpenAIAdapter(openai_client),
            model_prefixes=OPENAI_MODEL_PREFIXES + extra_openai_prefixes,
        )
    if gemini_client is not None:
        registry.register(GeminiAdapter(gemini_client), model_prefixes=GEMINI_MODEL_PREFIXES)
    if groq_client is not None:
        registry.register(GroqAdapter(groq_client), model_prefixes=GROQ_MODEL_PREFIXES)

    if not registry.provider_names:
        raise ValueError("build_registry needs at least one client")
    return registry

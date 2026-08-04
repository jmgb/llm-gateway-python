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
from llm_gateway.providers.assemblyai import AssemblyAIAdapter, AssemblyAIHttpClient
from llm_gateway.providers.gemini import GeminiAdapter
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openai import OpenAIAdapter
from llm_gateway.providers.openrouter import OpenRouterAdapter
from llm_gateway.registry import ProviderRegistry

OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
GEMINI_MODEL_PREFIXES = ("gemini-3", "models/gemini-3")
GROQ_MODEL_PREFIXES = (
    "llama",
    "mixtral",
    "gemma",
    "qwen",
    "kimi",
    "groq/",
    "whisper-",
    "distil-whisper-",
)
OPENROUTER_MODEL_PREFIXES = ("openrouter/",)
"""Deliberately short. OpenRouter's catalogued models route by their declared
provider, while uncatalogued vendor namespaces are rejected. Only OpenRouter's
own provider-prefixed ids need a fallback prefix."""
ASSEMBLYAI_MODEL_PREFIXES = ("assemblyai-",)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _require_key(api_key: str) -> None:
    if not api_key or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")


def create_openai_client(*, api_key: str, base_url: str | None = None) -> Any:
    """Build an ``AsyncOpenAI``.

    ``base_url`` points the same SDK at any OpenAI-compatible endpoint — Azure,
    vLLM, a self-hosted gateway. For OpenRouter use
    :func:`create_openrouter_client`, which is a provider in its own right.
    """
    _require_key(api_key)
    try:
        from openai import AsyncOpenAI
    except ImportError as error:
        raise ProviderNotInstalled.for_provider("openai") from error
    if base_url:
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    return AsyncOpenAI(api_key=api_key)


def create_openrouter_client(*, api_key: str, base_url: str = OPENROUTER_BASE_URL) -> Any:
    """Build an ``AsyncOpenAI`` pointed at OpenRouter.

    OpenRouter ships no SDK of its own: it speaks the OpenAI wire format, so
    the ``openai`` extra is what this needs installed. That is a fact about the
    transport, not about the provider — the adapter, the capabilities and the
    prices are OpenRouter's own.
    """
    _require_key(api_key)
    try:
        from openai import AsyncOpenAI
    except ImportError as error:
        raise ProviderNotInstalled.for_provider("openrouter") from error
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


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


def create_assemblyai_client(*, api_key: str) -> Any:
    """Build the small REST client used by the AssemblyAI adapter."""
    _require_key(api_key)
    try:
        __import__("httpx")
    except ImportError as error:
        raise ProviderNotInstalled.for_provider("assemblyai") from error
    return AssemblyAIHttpClient(api_key=api_key)


def build_registry(
    *,
    openai_client: Any | None = None,
    gemini_client: Any | None = None,
    groq_client: Any | None = None,
    openrouter_client: Any | None = None,
    assemblyai_client: Any | None = None,
    extra_openai_prefixes: tuple[str, ...] = (),
) -> ProviderRegistry:
    """Register the adapters for the clients the application supplies.

    ``extra_openai_prefixes`` widens the OpenAI adapter to model ids it does
    not know — an Azure deployment name, a self-hosted id. OpenRouter does not
    need it: pass ``openrouter_client`` and its models route by themselves.
    """
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
    if openrouter_client is not None:
        registry.register(
            OpenRouterAdapter(openrouter_client), model_prefixes=OPENROUTER_MODEL_PREFIXES
        )
    if assemblyai_client is not None:
        registry.register(
            AssemblyAIAdapter(assemblyai_client), model_prefixes=ASSEMBLYAI_MODEL_PREFIXES
        )

    if not registry.provider_names:
        raise ValueError("build_registry needs at least one client")
    return registry

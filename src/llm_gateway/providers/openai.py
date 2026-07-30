"""OpenAI Responses API adapter.

The client is injected. This module imports no SDK, so ``llm_gateway`` remains
importable with no extra installed; the application builds its own
``AsyncOpenAI`` (or anything shaped like it) and keeps ownership of the key.

OpenRouter is served by this same adapter: it exposes an OpenAI-compatible API,
so it is a ``base_url`` on the injected client rather than a fourth provider.
"""

from __future__ import annotations

from typing import Any

from llm_gateway.capabilities import ProviderCapabilities
from llm_gateway.contracts import LLMRequest, ResponseFormat
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.usage import TokenUsage

CAPABILITIES = ProviderCapabilities(
    structured_outputs=True,
    json_mode=True,
    function_calling=True,
    inline_files=True,
    remote_files=True,
    reasoning_effort=True,
    conversation_history=True,
    reports_token_usage=True,
)


class OpenAIAdapter:
    """Translates the neutral contract to the Responses API."""

    name = "openai"
    capabilities = CAPABILITIES

    def __init__(self, client: Any) -> None:
        self._client = client

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        kwargs = self._build_kwargs(request, model=model)
        try:
            raw = await self._client.responses.create(**kwargs)
        except Exception as error:
            raise classify_provider_error(error) from None

        return ProviderResponse(
            output_text=getattr(raw, "output_text", None),
            usage=_usage(getattr(raw, "usage", None)),
            finish_reason=getattr(raw, "status", None),
            model_used=getattr(raw, "model", None),
        )

    def _build_kwargs(self, request: LLMRequest, *, model: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system_prompt:
            kwargs["instructions"] = request.system_prompt
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": request.reasoning_effort}

        if request.response_format is ResponseFormat.JSON_OBJECT:
            kwargs["text"] = {"format": {"type": "json_object"}}
        elif request.response_format is ResponseFormat.JSON_SCHEMA:
            schema = request.response_schema
            assert schema is not None  # guaranteed by LLMRequest validation
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                }
            }
        return kwargs


def _usage(raw: Any) -> TokenUsage:
    if raw is None:
        return TokenUsage.unknown()
    details = getattr(raw, "output_tokens_details", None)
    return TokenUsage(
        input_tokens=getattr(raw, "input_tokens", None),
        output_tokens=getattr(raw, "output_tokens", None),
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
        cached_input_tokens=getattr(
            getattr(raw, "input_tokens_details", None), "cached_tokens", None
        ),
    )

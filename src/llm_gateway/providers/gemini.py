"""Google Gen AI adapter.

Targets the ``google-genai`` SDK (the ``client.aio`` async surface), not the
retired ``google-generativeai`` package. The client is injected, so this module
imports no SDK.

File Search / Interactions is deliberately out of scope for this version: it is
a distinct capability with its own cost model, and pretending it is the same
call would hide that retrieved documents are billed as input context.
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
    reasoning_effort=False,
    conversation_history=True,
    reports_token_usage=True,
)


class GeminiAdapter:
    """Translates the neutral contract to ``generate_content``."""

    name = "gemini"
    capabilities = CAPABILITIES

    def __init__(self, client: Any) -> None:
        self._client = client

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        try:
            raw = await self._client.aio.models.generate_content(
                model=model,
                contents=[
                    {"role": _role(m.role), "parts": [{"text": m.content}]}
                    for m in request.messages
                ],
                config=self._build_config(request),
            )
        except Exception as error:
            raise classify_provider_error(error) from None

        return ProviderResponse(
            output_text=getattr(raw, "text", None),
            usage=_usage(getattr(raw, "usage_metadata", None)),
            finish_reason=_finish_reason(raw),
            model_used=getattr(raw, "model_version", None),
        )

    def _build_config(self, request: LLMRequest) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if request.system_prompt:
            config["system_instruction"] = request.system_prompt
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            config["max_output_tokens"] = request.max_output_tokens

        if request.response_format is ResponseFormat.JSON_OBJECT:
            config["response_mime_type"] = "application/json"
        elif request.response_format is ResponseFormat.JSON_SCHEMA:
            schema = request.response_schema
            assert schema is not None  # guaranteed by LLMRequest validation
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = schema.model_json_schema()
        return config


def _role(role: str) -> str:
    """Gemini calls the assistant turn "model"."""
    return "model" if role == "assistant" else "user"


def _usage(raw: Any) -> TokenUsage:
    if raw is None:
        return TokenUsage.unknown()
    # Gemini reports thoughts *outside* candidates_token_count, unlike the
    # providers whose output count already contains them. They are billed at
    # the output rate, so folding them in here is what keeps output_tokens
    # meaning the same thing in every adapter. Without a candidate count there
    # is no full output to complete, so thoughts alone stay unknown.
    candidates = getattr(raw, "candidates_token_count", None)
    thoughts = getattr(raw, "thoughts_token_count", None)
    return TokenUsage(
        input_tokens=getattr(raw, "prompt_token_count", None),
        output_tokens=None if candidates is None else candidates + (thoughts or 0),
        reasoning_tokens=thoughts,
        cached_input_tokens=getattr(raw, "cached_content_token_count", None),
    )


def _finish_reason(raw: Any) -> str | None:
    candidates = getattr(raw, "candidates", None) or ()
    for candidate in candidates:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            return str(reason)
    return None

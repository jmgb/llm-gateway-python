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
from llm_gateway.errors import ConfigurationError, ProviderError
from llm_gateway.media import GeneratedImage, ImageRequest, ProviderImageResponse
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.providers.validation import reject_file_attachments, reject_tools
from llm_gateway.usage import ImageUsage, TokenUsage

CAPABILITIES = ProviderCapabilities(
    structured_outputs=True,
    json_mode=True,
    image_generation=True,
    image_editing=True,
    # Gemini serves all three; the neutral request cannot express any of them
    # yet, and a capability no caller can reach is a promise that answers
    # nothing. See tests/contract/test_capability_honesty.py.
    function_calling=False,
    inline_files=False,
    remote_files=False,
    audio_transcription=False,
    reasoning_effort=True,
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
        reject_file_attachments(request, provider=self.name)
        reject_tools(request, provider=self.name)
        try:
            raw = await self._client.aio.models.generate_content(
                model=model,
                contents=[
                    {"role": _role(m.role), "parts": [{"text": m.content}]}
                    for m in request.messages
                ],
                config=self._build_config(request, model=model),
            )
        except Exception as error:
            raise classify_provider_error(error) from None

        return ProviderResponse(
            output_text=getattr(raw, "text", None),
            usage=_usage(getattr(raw, "usage_metadata", None)),
            finish_reason=_finish_reason(raw),
            model_used=getattr(raw, "model_version", None),
        )

    async def generate_image(self, request: ImageRequest, *, model: str) -> ProviderImageResponse:
        """Translate to ``generate_content`` asking for the image modality.

        Gemini answers a text description often enough that a reply without
        inline data is treated as a provider failure: returning it as an empty
        success is how a caller ends up showing the user nothing.
        """
        parts: list[dict[str, Any]] = [{"text": request.prompt}]
        if request.image is not None:
            if request.image.data is None:
                raise ConfigurationError(
                    "Gemini needs the source image as bytes; it does not fetch a URL"
                )
            parts.append(
                {
                    "inline_data": {
                        "mime_type": request.image.mime_type or "image/png",
                        "data": request.image.data,
                    }
                }
            )

        config: dict[str, Any] = {"response_modalities": ["IMAGE"]}
        if request.aspect_ratio is not None:
            config["image_config"] = {"aspect_ratio": request.aspect_ratio}

        try:
            raw = await self._client.aio.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": parts}],
                config=config,
            )
        except Exception as error:
            raise classify_provider_error(error) from None

        return _image_response(raw, model=model)

    def _build_config(self, request: LLMRequest, *, model: str) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if request.system_prompt:
            config["system_instruction"] = request.system_prompt
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            config["max_output_tokens"] = request.max_output_tokens
        if request.reasoning_effort is not None and _is_gemini_3(model):
            config["thinking_config"] = {"thinking_level": request.reasoning_effort}

        if request.response_format is ResponseFormat.JSON_OBJECT:
            config["response_mime_type"] = "application/json"
        elif request.response_format is ResponseFormat.JSON_SCHEMA:
            schema = request.response_schema
            assert schema is not None  # guaranteed by LLMRequest validation
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = schema.model_json_schema()
        return config


_BLOCKING_FINISH_REASONS = ("SAFETY", "PROHIBITED", "RECITATION", "BLOCKLIST", "SPII")


def _image_response(raw: Any, *, model: str) -> ProviderImageResponse:
    candidates = getattr(raw, "candidates", None) or ()
    candidate = candidates[0] if candidates else None
    if candidate is None:
        raise ProviderError("Gemini returned no candidate for the image request")

    reason = str(getattr(candidate, "finish_reason", "") or "")
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or ()
    images = tuple(
        GeneratedImage(
            data=part.inline_data.data,
            mime_type=getattr(part.inline_data, "mime_type", None),
        )
        for part in parts
        if getattr(part, "inline_data", None) is not None
    )
    if not images:
        if any(blocked in reason.upper() for blocked in _BLOCKING_FINISH_REASONS):
            raise ProviderError(f"Gemini blocked the image generation: {reason}")
        raise ProviderError(f"Gemini returned no image (finish reason: {reason or 'unknown'})")

    tokens = _usage(getattr(raw, "usage_metadata", None))
    return ProviderImageResponse(
        images=images,
        usage=ImageUsage(images=len(images), tokens=tokens),
        model_used=getattr(raw, "model_version", None) or model,
    )


def _is_gemini_3(model: str) -> bool:
    return model.startswith(("gemini-3", "models/gemini-3"))


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

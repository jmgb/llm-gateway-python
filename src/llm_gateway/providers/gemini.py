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
from llm_gateway.providers.base import ProviderResponse, reported_count, reported_usage
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.providers.validation import (
    reject_file_attachments,
    reject_tools,
    reject_unsendable_image_options,
)
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
        # Built before the call, and refused as configuration: a schema that
        # fails to render is a local bug, and classified as a provider error it
        # would be retried, counted as a billable attempt and handed to the
        # fallback, none of which can fix it.
        contents = [
            {"role": _role(m.role), "parts": [{"text": m.content}]} for m in request.messages
        ]
        try:
            config = self._build_config(request, model=model)
        except Exception as error:
            raise ConfigurationError(
                f"the request could not be built for Gemini ({type(error).__name__})"
            ) from error
        try:
            raw = await self._client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as error:
            raise classify_provider_error(error) from error

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
        reject_unsendable_image_options(request, provider=self.name)
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
            raise classify_provider_error(error) from error

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

    reason = _reason_text(getattr(candidate, "finish_reason", None)) or ""
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


_GEMINI_3_ALIASES = ("gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest")
"""Floating ids the catalogue prices, and declares efforts for, as generation 3.

Without them here, an alias accepted an effort in the catalogue and lost it at
this boundary: the call went out without thinking and nothing said so.
"""


def _is_gemini_3(model: str) -> bool:
    bare = model.removeprefix("models/")
    return bare.startswith("gemini-3") or bare in _GEMINI_3_ALIASES


def _role(role: str) -> str:
    """Gemini calls the assistant turn "model"."""
    return "model" if role == "assistant" else "user"


def _usage(raw: Any) -> TokenUsage:
    if raw is None:
        return TokenUsage.unknown()
    # Gemini reports thoughts *outside* candidates_token_count, unlike the
    # providers whose output count already contains them. They are billed at
    # the output rate, so folding them in here is what keeps output_tokens
    # meaning the same thing in every adapter.
    prompt = reported_count(getattr(raw, "prompt_token_count", None))
    candidates = reported_count(getattr(raw, "candidates_token_count", None))
    thoughts = reported_count(getattr(raw, "thoughts_token_count", None))
    if candidates is None:
        candidates = _visible_from_total(raw, prompt=prompt, thoughts=thoughts)
    return reported_usage(
        input_tokens=prompt,
        output_tokens=None if candidates is None else candidates + (thoughts or 0),
        reasoning_tokens=thoughts,
        cached_input_tokens=getattr(raw, "cached_content_token_count", None),
    )


def _visible_from_total(raw: Any, *, prompt: int | None, thoughts: int | None) -> int | None:
    """The candidate count Gemini leaves out when it is zero.

    That is exactly the reply where thinking spent ``max_output_tokens`` before
    any text was written, so reading the gap as unknown output dropped every
    billed thought. The total still accounts for it; anything it cannot
    reconcile — a total smaller than its parts — stays unknown rather than
    guessed.
    """
    total = reported_count(getattr(raw, "total_token_count", None))
    if total is None or prompt is None:
        return None
    tool_prompt = reported_count(getattr(raw, "tool_use_prompt_token_count", None)) or 0
    visible = total - prompt - (thoughts or 0) - tool_prompt
    return visible if visible >= 0 else None


def _finish_reason(raw: Any) -> str | None:
    candidates = getattr(raw, "candidates", None) or ()
    for candidate in candidates:
        reason = _reason_text(getattr(candidate, "finish_reason", None))
        if reason is not None:
            return reason
    return None


def _reason_text(reason: Any) -> str | None:
    """``STOP``, as other providers report theirs, not ``FinishReason.STOP``.

    The SDK hands back an enum, and ``str()`` of one names its class.
    """
    if reason is None:
        return None
    return str(getattr(reason, "value", reason))

"""Groq adapter.

Chat Completions shape. The client is injected, so this module imports no SDK.
"""

from __future__ import annotations

from typing import Any

from llm_gateway.audio import (
    ProviderTranscriptionResponse,
    TranscriptionRequest,
    normalize_provider_transcription,
)
from llm_gateway.capabilities import ProviderCapabilities
from llm_gateway.contracts import LLMRequest, ResponseFormat
from llm_gateway.errors import ConfigurationError
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.providers.schema_prompt import system_prompt_for
from llm_gateway.providers.validation import reject_file_attachments
from llm_gateway.usage import TokenUsage

CAPABILITIES = ProviderCapabilities(
    structured_outputs=False,
    json_mode=True,
    # No request field asks for tools, so declaring them promises nothing.
    function_calling=False,
    inline_files=False,
    remote_files=False,
    audio_transcription=True,
    reasoning_effort=True,
    conversation_history=True,
    reports_token_usage=True,
)


class GroqAdapter:
    """Translates the neutral contract to Chat Completions."""

    name = "groq"
    capabilities = CAPABILITIES

    def __init__(self, client: Any) -> None:
        self._client = client

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        reject_file_attachments(request, provider=self.name)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(request),
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.reasoning_effort is not None and model.startswith("openai/gpt-oss-"):
            kwargs["reasoning_effort"] = request.reasoning_effort
        if request.response_format in (ResponseFormat.JSON_OBJECT, ResponseFormat.JSON_SCHEMA):
            # Groq does not enforce a schema; asking for JSON is the most this
            # provider can honestly promise, and the gateway validates after.
            kwargs["response_format"] = {"type": "json_object"}

        try:
            raw = await self._client.chat.completions.create(**kwargs)
        except Exception as error:
            raise classify_provider_error(error) from None

        choice = _first_choice(raw)
        return ProviderResponse(
            output_text=getattr(getattr(choice, "message", None), "content", None),
            usage=_usage(getattr(raw, "usage", None)),
            finish_reason=getattr(choice, "finish_reason", None),
            model_used=getattr(raw, "model", None),
        )

    async def transcribe(
        self, request: TranscriptionRequest, *, model: str
    ) -> ProviderTranscriptionResponse:
        if request.speaker_labels:
            raise ConfigurationError("Groq transcription does not support speaker labels")
        kwargs: dict[str, Any] = {
            "model": model,
            "response_format": "verbose_json",
            "temperature": 0.0,
        }
        if request.audio.url is not None:
            kwargs["url"] = request.audio.url
        elif request.audio.data is not None:
            kwargs["file"] = (
                (request.audio.filename, request.audio.data, request.audio.mime_type)
                if request.audio.mime_type is not None
                else (request.audio.filename, request.audio.data)
            )
        else:
            raise ConfigurationError("Groq transcription requires audio bytes or a URL")
        if request.language is not None:
            kwargs["language"] = request.language
        if request.prompt:
            kwargs["prompt"] = request.prompt
        try:
            raw = await self._client.audio.transcriptions.create(**kwargs)
        except Exception as error:
            raise classify_provider_error(error) from None
        return normalize_provider_transcription(raw, request=request, model=model)

    def _build_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        """Carry whatever the requested format needs the model to be told.

        Groq enforces no schema, so a structured call that does not describe
        one leaves the model to invent its field names — a valid-JSON answer
        that fails validation, is billed, and hands the call to the fallback.
        It also rejects `json_object` outright when the messages never say
        "json", so both JSON formats need something said here.
        """
        messages: list[dict[str, str]] = []
        system_prompt = system_prompt_for(request)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        return messages


def _first_choice(raw: Any) -> Any:
    choices = getattr(raw, "choices", None) or ()
    return choices[0] if choices else None


def _usage(raw: Any) -> TokenUsage:
    if raw is None:
        return TokenUsage.unknown()
    return TokenUsage(
        input_tokens=getattr(raw, "prompt_tokens", None),
        # Chat Completions counts reasoning inside completion_tokens, so the
        # breakdown changes no amount. It is read anyway: without it a thinking
        # model looks like it returned every token it was billed for. Models
        # that do not think report no details, and the breakdown stays unknown.
        output_tokens=getattr(raw, "completion_tokens", None),
        reasoning_tokens=getattr(
            getattr(raw, "completion_tokens_details", None), "reasoning_tokens", None
        ),
    )

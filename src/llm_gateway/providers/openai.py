"""OpenAI Responses API adapter.

The client is injected. This module imports no SDK, so ``llm_gateway`` remains
importable with no extra installed; the application builds its own
``AsyncOpenAI`` (or anything shaped like it) and keeps ownership of the key.

Pointing the injected client at another OpenAI-compatible endpoint — Azure,
vLLM, a self-hosted gateway — is the application's call and needs nothing here.
OpenRouter is *not* one of those: it has its own adapter, because it speaks
Chat Completions and cannot promise this one's capabilities.
"""

from __future__ import annotations

import json
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
from llm_gateway.providers.strict_schema import strict_json_schema
from llm_gateway.tools import FunctionTool, ProviderToolCall, RequiredTool, ToolChoice
from llm_gateway.usage import TokenUsage

CAPABILITIES = ProviderCapabilities(
    structured_outputs=True,
    json_mode=True,
    function_calling=True,
    # The provider does this too; the request contract has no way to ask for
    # it, and a capability a caller cannot exercise reads as available while
    # answering nothing. Declared when the contract grows.
    inline_files=False,
    remote_files=True,
    audio_transcription=True,
    reasoning_effort=True,
    verbosity=True,
    upstream_routing=False,
    conversation_history=True,
    reports_token_usage=True,
)

VERBOSITY_MODEL_PREFIXES = ("gpt-5",)
"""Families that document the dial.

Sending it to a model that predates it is a 400 for the whole call, and a
fallback inherits the request that asked for it — so the gate is here as well
as in the catalogue, the same arrangement reasoning effort already uses.
"""


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
            tool_calls=_tool_calls(raw),
        )

    async def transcribe(
        self, request: TranscriptionRequest, *, model: str
    ) -> ProviderTranscriptionResponse:
        if request.audio.data is None:
            raise ConfigurationError("OpenAI transcription requires audio bytes")
        if request.speaker_labels:
            raise ConfigurationError("OpenAI gpt-transcribe does not support speaker labels")
        kwargs: dict[str, Any] = {
            "model": model,
            "file": (
                (request.audio.filename, request.audio.data, request.audio.mime_type)
                if request.audio.mime_type is not None
                else (request.audio.filename, request.audio.data)
            ),
            # The current gpt-transcribe endpoint is an API-backed model
            # alias and accepts json/text, not the Whisper-only verbose_json
            # format. Keep verbose_json for other OpenAI transcription ids.
            "response_format": "json" if model == "gpt-transcribe" else "verbose_json",
        }
        if request.language is not None:
            if model == "gpt-transcribe":
                kwargs["languages"] = [request.language]
            else:
                kwargs["language"] = request.language
        if request.prompt:
            kwargs["prompt"] = request.prompt
        try:
            raw = await self._client.audio.transcriptions.create(**kwargs)
        except Exception as error:
            raise classify_provider_error(error) from None
        return normalize_provider_transcription(raw, request=request, model=model)

    def _build_kwargs(self, request: LLMRequest, *, model: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model, "input": self._build_input(request)}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": request.reasoning_effort}

        # Format and verbosity share one field, so they are collected before it
        # is set: assigning either on its own would discard the other.
        text: dict[str, Any] = {}
        if request.response_format is ResponseFormat.JSON_OBJECT:
            text["format"] = {"type": "json_object"}
        elif request.response_format is ResponseFormat.JSON_SCHEMA:
            schema = request.response_schema
            assert schema is not None  # guaranteed by LLMRequest validation
            text["format"] = {
                "type": "json_schema",
                "name": schema.__name__,
                # Pydantic's schema is not the subset strict mode accepts;
                # sending it unchanged is a 400 on every structured call.
                "schema": strict_json_schema(schema),
                "strict": True,
            }
        if request.verbosity is not None and model.startswith(VERBOSITY_MODEL_PREFIXES):
            text["verbosity"] = request.verbosity
        if text:
            kwargs["text"] = text

        if request.tools:
            kwargs["tools"] = [_function_tool(tool) for tool in request.tools]
            # Stating nothing and stating "auto" are the same instruction here,
            # and sending it makes the request self-describing in a provider log.
            kwargs["tool_choice"] = _tool_choice(request.tool_choice)
        return kwargs

    def _build_input(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Carry the system prompt as a message rather than as ``instructions``.

        ``json_object`` mode is rejected unless the word "json" appears in the
        input, and ``instructions`` is not part of the input. A system prompt
        that asks for JSON would therefore be invisible to that check. Sending
        it as a message also matches the arrangement Chat Completions used.
        """
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        if request.attachments:
            for message in reversed(messages):
                if message["role"] != "user":
                    continue
                text = message["content"]
                message["content"] = [{"type": "input_text", "text": text}]
                message["content"].extend(
                    {"type": "input_file", "file_id": attachment.file_id}
                    for attachment in request.attachments
                )
                break

        # After the attachment pass, which walks the list looking for a role:
        # these items have none, and they must stay at the end anyway. The
        # model's own call is replayed before its output, because the provider
        # rejects an output that answers nothing it can see.
        for result in request.tool_results:
            messages.append(
                {
                    "type": "function_call",
                    "call_id": result.call.id,
                    "name": result.call.name,
                    "arguments": json.dumps(result.call.arguments, ensure_ascii=False),
                }
            )
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": result.call.id,
                    "output": result.output,
                }
            )
        return messages


def _function_tool(tool: FunctionTool) -> dict[str, Any]:
    """The flat shape the Responses API takes, unlike Chat Completions' nested one.

    ``additionalProperties`` is filled in when the caller left it out: the API
    rejects a function schema without it, which is a 400 for the whole call and
    not just for the tool.
    """
    parameters = dict(tool.parameters)
    parameters.setdefault("additionalProperties", False)
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": parameters,
    }


def _tool_choice(choice: ToolChoice | RequiredTool | None) -> Any:
    if isinstance(choice, RequiredTool):
        return {"type": "function", "name": choice.name}
    return (choice or ToolChoice.AUTO).value


def _tool_calls(raw: Any) -> tuple[ProviderToolCall, ...]:
    """Read the calls out of an output list that also carries reasoning items."""
    items = getattr(raw, "output", None) or ()
    calls: list[ProviderToolCall] = []
    for index, item in enumerate(items):
        if getattr(item, "type", None) != "function_call":
            continue
        arguments = getattr(item, "arguments", "")
        calls.append(
            ProviderToolCall(
                # `call_id` is the one the continuation must quote; `id` is the
                # output item's own and is not interchangeable with it.
                id=getattr(item, "call_id", None) or getattr(item, "id", None) or f"call_{index}",
                name=getattr(item, "name", ""),
                arguments=(
                    json.dumps(arguments, ensure_ascii=False)
                    if isinstance(arguments, dict | list)
                    else str(arguments)
                ),
            )
        )
    return tuple(calls)


def _usage(raw: Any) -> TokenUsage:
    if raw is None:
        return TokenUsage.unknown()
    details = getattr(raw, "output_tokens_details", None)
    return TokenUsage(
        input_tokens=getattr(raw, "input_tokens", None),
        # The Responses API counts reasoning inside output_tokens: input plus
        # output reconciles to total_tokens. It is reported here only as a
        # breakdown, so it must not be added to the billable output.
        output_tokens=getattr(raw, "output_tokens", None),
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
        cached_input_tokens=getattr(
            getattr(raw, "input_tokens_details", None), "cached_tokens", None
        ),
    )

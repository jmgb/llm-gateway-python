"""Groq adapter.

Chat Completions shape. The client is injected, so this module imports no SDK.
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
from llm_gateway.providers.chat_completions import base_messages, first_choice
from llm_gateway.providers.chat_completions import token_usage as _usage
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.providers.validation import reject_file_attachments
from llm_gateway.tools import FunctionTool, ProviderToolCall, RequiredTool, ToolChoice

CAPABILITIES = ProviderCapabilities(
    structured_outputs=False,
    json_mode=True,
    function_calling=True,
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
        if request.tools:
            kwargs["tools"] = [_function_tool(tool) for tool in request.tools]
            kwargs["tool_choice"] = _tool_choice(request.tool_choice)
        elif request.response_format in (ResponseFormat.JSON_OBJECT, ResponseFormat.JSON_SCHEMA):
            # Tools and the JSON format are mutually exclusive here: both
            # applications that ship this send one or the other, never the
            # pair. Nothing is lost by the omission — Groq enforces no schema
            # anyway, the system prompt still describes the shape, and the
            # gateway still validates whatever comes back.
            kwargs["response_format"] = {"type": "json_object"}

        try:
            raw = await self._client.chat.completions.create(**kwargs)
        except Exception as error:
            raise classify_provider_error(error) from error

        choice = first_choice(raw)
        return ProviderResponse(
            output_text=getattr(getattr(choice, "message", None), "content", None),
            usage=_usage(getattr(raw, "usage", None)),
            finish_reason=getattr(choice, "finish_reason", None),
            model_used=getattr(raw, "model", None),
            tool_calls=_tool_calls(choice),
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
            raise classify_provider_error(error) from error
        return normalize_provider_transcription(raw, request=request, model=model)

    def _build_messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Carry whatever the requested format needs the model to be told.

        Groq enforces no schema, so a structured call that does not describe
        one leaves the model to invent its field names — a valid-JSON answer
        that fails validation, is billed, and hands the call to the fallback.
        It also rejects `json_object` outright when the messages never say
        "json", so both JSON formats need something said here.
        """
        messages = base_messages(request)

        # One assistant turn holding every call it made, then one message per
        # result. A `tool` message whose id names no call above it is a 400.
        if request.tool_results:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": result.call.id,
                            "type": "function",
                            "function": {
                                "name": result.call.name,
                                "arguments": json.dumps(result.call.arguments, ensure_ascii=False),
                            },
                        }
                        for result in request.tool_results
                    ],
                }
            )
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": result.call.id,
                    "content": result.output,
                }
                for result in request.tool_results
            )
        return messages


def _function_tool(tool: FunctionTool) -> dict[str, Any]:
    """The nested shape Chat Completions takes, unlike the Responses API's flat one.

    ``additionalProperties`` is *not* filled in, unlike the OpenAI adapter:
    nothing here rejects a schema without it, and adding it would narrow a
    caller's schema behind their back.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": dict(tool.parameters),
        },
    }


def _tool_choice(choice: ToolChoice | RequiredTool | None) -> Any:
    if isinstance(choice, RequiredTool):
        return {"type": "function", "function": {"name": choice.name}}
    return (choice or ToolChoice.AUTO).value


def _tool_calls(choice: Any) -> tuple[ProviderToolCall, ...]:
    raw_calls = getattr(getattr(choice, "message", None), "tool_calls", None) or ()
    calls: list[ProviderToolCall] = []
    for raw in raw_calls:
        function = getattr(raw, "function", None)
        call_id = getattr(raw, "id", None)
        calls.append(
            ProviderToolCall(
                id=str(call_id) if call_id else "",
                # Kept even when empty, as the OpenAI adapter does: skipping it
                # would pass a malformed reply off as an answer with no calls,
                # where the gateway would otherwise refuse it and count it.
                name=getattr(function, "name", None) or "",
                arguments=getattr(function, "arguments", "") or "",
            )
        )
    return tuple(calls)

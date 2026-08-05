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
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.providers.schema_prompt import system_prompt_for
from llm_gateway.providers.validation import reject_file_attachments
from llm_gateway.tools import FunctionTool, ProviderToolCall, RequiredTool, ToolChoice
from llm_gateway.usage import TokenUsage

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
            raise classify_provider_error(error) from None

        choice = _first_choice(raw)
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
            raise classify_provider_error(error) from None
        return normalize_provider_transcription(raw, request=request, model=model)

    def _build_messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Carry whatever the requested format needs the model to be told.

        Groq enforces no schema, so a structured call that does not describe
        one leaves the model to invent its field names — a valid-JSON answer
        that fails validation, is billed, and hands the call to the fallback.
        It also rejects `json_object` outright when the messages never say
        "json", so both JSON formats need something said here.
        """
        messages: list[dict[str, Any]] = []
        system_prompt = system_prompt_for(request)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)

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
    for index, raw in enumerate(raw_calls):
        function = getattr(raw, "function", None)
        name = getattr(function, "name", None)
        if not name:
            continue
        calls.append(
            ProviderToolCall(
                id=getattr(raw, "id", None) or f"call_{index}",
                name=name,
                arguments=getattr(function, "arguments", "") or "",
            )
        )
    return tuple(calls)


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

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

import base64
import json
from typing import Any

from llm_gateway.audio import (
    ProviderTranscriptionResponse,
    TranscriptionRequest,
    normalize_provider_transcription,
)
from llm_gateway.capabilities import ProviderCapabilities
from llm_gateway.contracts import LLMRequest, ResponseFormat
from llm_gateway.errors import ConfigurationError, LLMGatewayError, ProviderError
from llm_gateway.media import GeneratedImage, ImageRequest, ProviderImageResponse
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.providers.schema_prompt import system_prompt_for
from llm_gateway.providers.strict_schema import strict_json_schema
from llm_gateway.tools import FunctionTool, ProviderToolCall, RequiredTool, ToolChoice
from llm_gateway.usage import ImageUsage, TokenUsage

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
    # The gpt-image family, reached through the Images API rather than
    # Responses. The catalogue keeps the two apart: an image model is
    # modality="image", so generate() refuses it and the image gateway is the
    # only way in.
    image_generation=True,
    image_editing=True,
    reasoning_effort=True,
    verbosity=True,
    upstream_routing=False,
    conversation_history=True,
    reports_token_usage=True,
)

VERBOSITY_MODEL_PREFIXES = ("gpt-5", "gpt-6")
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

    async def generate_image(self, request: ImageRequest, *, model: str) -> ProviderImageResponse:
        """Translate to the Images API, which is not the Responses one.

        Editing and generating are two endpoints rather than one call with an
        optional field, so the presence of a source image decides which is
        used. Neither accepts an aspect ratio, and squaring a portrait the
        caller asked for is a bug that only shows up in the finished picture.
        """
        if request.aspect_ratio is not None:
            raise ConfigurationError(
                "OpenAI sizes images as WIDTHxHEIGHT; it takes no aspect ratio"
            )

        kwargs: dict[str, Any] = {"model": model, "prompt": request.prompt}
        if request.size is not None:
            kwargs["size"] = request.size
        # Left out when unstated: OpenAI's own default applies, and the reply
        # reports the output tokens it produced, so the bill stays priceable
        # either way.
        if request.quality is not None:
            kwargs["quality"] = request.quality

        try:
            if request.image is None:
                raw = await self._client.images.generate(**kwargs)
            else:
                if request.image.data is None:
                    raise ConfigurationError(
                        "OpenAI needs the source image as bytes; it does not fetch a URL"
                    )
                mime_type = request.image.mime_type or "image/png"
                kwargs["image"] = (
                    f"image.{mime_type.rsplit('/', 1)[-1]}",
                    request.image.data,
                    mime_type,
                )
                raw = await self._client.images.edit(**kwargs)
        except LLMGatewayError:
            raise
        except Exception as error:
            raise classify_provider_error(error) from None

        return _image_response(raw, model=model)

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

        Placing the prompt where the check can see it is only half of the rule;
        the word still has to be there. ``system_prompt_for`` adds it when the
        caller's prompt does not, which is the same debt Groq and OpenRouter
        already settle here. Structured outputs are declared, so the schema
        itself is not repeated in the conversation.
        """
        messages: list[dict[str, Any]] = []
        system_prompt = system_prompt_for(request, structured_outputs=True)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
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
    for item in items:
        if getattr(item, "type", None) != "function_call":
            continue
        arguments = getattr(item, "arguments", "")
        call_id = getattr(item, "call_id", None)
        calls.append(
            ProviderToolCall(
                # `call_id` is the one the continuation must quote; `id` is the
                # output item's own and is not interchangeable with it. Keep a
                # missing id empty so the gateway can reject and account for it.
                id=str(call_id) if call_id else "",
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


def _image_response(raw: Any, *, model: str) -> ProviderImageResponse:
    """Read the pictures and the three-way token split out of one reply."""
    images: list[GeneratedImage] = []
    mime_type = _image_mime_type(raw)
    for item in getattr(raw, "data", None) or ():
        encoded = getattr(item, "b64_json", None)
        if encoded:
            images.append(GeneratedImage(data=base64.b64decode(encoded), mime_type=mime_type))
            continue
        url = getattr(item, "url", None)
        if url:
            images.append(GeneratedImage(url=str(url), mime_type=mime_type))
    if not images:
        raise ProviderError(f"OpenAI returned no image for {model}")

    usage = getattr(raw, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    return ProviderImageResponse(
        images=tuple(images),
        usage=ImageUsage(
            images=len(images),
            tokens=_image_usage(usage),
            # Absent rather than zero when the provider reported no breakdown:
            # the image price catalogue would otherwise charge a reference
            # photo at the cheaper text rate and understate the invoice.
            input_image_tokens=getattr(details, "image_tokens", None),
        ),
        model_used=model,
    )


def _image_mime_type(raw: Any) -> str:
    """PNG unless the reply says otherwise, which is the API's own default."""
    output_format = getattr(raw, "output_format", None)
    return f"image/{output_format}" if output_format else "image/png"


def _image_usage(raw: Any) -> TokenUsage:
    """The Images API reports no reasoning, so only the two totals are read."""
    if raw is None:
        return TokenUsage.unknown()
    return TokenUsage(
        input_tokens=getattr(raw, "input_tokens", None),
        output_tokens=getattr(raw, "output_tokens", None),
        cached_input_tokens=getattr(
            getattr(raw, "input_tokens_details", None), "cached_tokens", None
        ),
    )

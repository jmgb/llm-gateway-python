"""Venice adapter: Chat Completions for text, queue-and-retrieve for video.

Venice speaks the OpenAI wire format for chat, but it is a provider of its own
for the reasons ``CONTRIBUTING.md`` lists: it has its own prices, its own
catalogue names, and a video API no OpenAI client knows. One small REST client
carries both, so the adapter needs ``httpx2`` and no SDK.

Two facts about Venice shape every request:

* It prepends a system prompt of its own unless told not to — around 1500
  tokens the caller did not write, billed on every call and steering the
  answer. This package ships no prompts, so it switches that one off.
* Cloudflare in front of the API refuses the default User-Agent of Python HTTP
  clients, so the client sends a neutral one instead.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

from llm_gateway.capabilities import ProviderCapabilities
from llm_gateway.contracts import LLMRequest, ResponseFormat
from llm_gateway.errors import ConfigurationError, LLMGatewayError, ProviderError
from llm_gateway.media import GeneratedVideo, ProviderVideoResponse, VideoRequest
from llm_gateway.providers._polling import read_status
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.chat_completions import token_usage as _usage
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.providers.schema_prompt import system_prompt_for
from llm_gateway.providers.strict_schema import strict_json_schema
from llm_gateway.providers.validation import (
    reject_file_attachments,
    reject_tools,
    reject_visual_inputs,
)
from llm_gateway.usage import VideoUsage

VENICE_BASE_URL = "https://api.venice.ai/api/v1"

CAPABILITIES = ProviderCapabilities(
    # Every catalogued Venice text model declares `supportsResponseSchema`,
    # and its strict grammar is what binds the answer to the shape.
    structured_outputs=True,
    json_mode=True,
    # Venice's models do declare tool support, but the round trip has not been
    # tested here, so a request with tools is refused rather than answered in
    # prose where a call was expected.
    function_calling=False,
    video_generation=True,
    # Only the text-to-video variant is catalogued; a first frame is refused.
    video_from_image=False,
    conversation_history=True,
    reports_token_usage=True,
)

# Keys Venice's structured-output grammar answers HTTP 400 "Unimplemented
# keys" for. Pydantic emits `uniqueItems` for every `set` field. Removing it
# only from what is sent loses nothing: the gateway validates the answer
# against the caller's full model afterwards.
_UNSUPPORTED_SCHEMA_KEYS = frozenset({"uniqueItems"})


@dataclass(frozen=True, slots=True)
class _VideoShape:
    """What a video model accepts, read from Venice's `/models?type=video`."""

    resolution: str
    """The only tier, spelt as the API demands: `768P`, never `768p`."""
    durations: range
    """Whole seconds. Venice requires one on every request and bills by it."""


_VIDEO_SHAPES = {
    "minimax-h3-chili-text-to-video": _VideoShape(resolution="768P", durations=range(4, 16)),
}

# The cheapest tier each video model offers, in the gateway's lower-case
# spelling, which is the one the price tables and `VideoUsage` use.
_LOWEST_RESOLUTION = {model: shape.resolution.lower() for model, shape in _VIDEO_SHAPES.items()}


class VeniceHttpClient:
    """Small injected REST client; it owns the key, never the gateway."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = VENICE_BASE_URL,
        user_agent: str = "Mozilla/5.0 (compatible; neutral-llm-gateway)",
        timeout_seconds: float = 600.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "User-Agent": user_agent}
        self._timeout_seconds = timeout_seconds

    async def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any] | bytes:
        """The decoded JSON body, or the raw bytes of a finished clip."""
        import httpx2

        async with httpx2.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                f"{self._base_url}/{path.lstrip('/')}", headers=self._headers, json=json
            )
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("video/"):
                return bytes(response.content)
            return cast(dict[str, Any], response.json())


class VeniceAdapter:
    """Translates text and video requests into Venice's REST API."""

    name = "venice"
    capabilities = CAPABILITIES

    def __init__(
        self,
        client: Any,
        *,
        max_poll_attempts: int = 180,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        if max_poll_attempts < 1:
            raise ValueError("max_poll_attempts must be positive")
        if poll_interval_seconds < 0:
            raise ValueError("poll interval cannot be negative")
        self._client = client
        self._max_poll_attempts = max_poll_attempts
        self._poll_interval_seconds = poll_interval_seconds

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        reject_visual_inputs(request, provider=self.name)
        reject_file_attachments(request, provider=self.name)
        reject_tools(request, provider=self.name)
        body: dict[str, Any] = {
            "model": model,
            "messages": _messages(request),
            "venice_parameters": {"include_venice_system_prompt": False},
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            body["max_tokens"] = request.max_output_tokens
        if request.response_format is ResponseFormat.JSON_OBJECT:
            body["response_format"] = {"type": "json_object"}
        elif request.response_format is ResponseFormat.JSON_SCHEMA:
            schema = request.response_schema
            assert schema is not None  # guaranteed by LLMRequest validation
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": _without_unsupported_keys(strict_json_schema(schema)),
                },
            }

        try:
            raw = await self._client.post("/chat/completions", json=body)
        except Exception as error:
            raise classify_provider_error(error) from error
        if not isinstance(raw, dict):
            raise ProviderError("Venice answered a chat call with something other than JSON")

        choices = raw.get("choices") or ()
        choice = choices[0] if choices else {}
        return ProviderResponse(
            output_text=(choice.get("message") or {}).get("content"),
            usage=_usage(_namespace(raw.get("usage"))),
            finish_reason=choice.get("finish_reason"),
            model_used=raw.get("model"),
        )

    async def generate_video(self, request: VideoRequest, *, model: str) -> ProviderVideoResponse:
        """Queue the clip, then retrieve until Venice hands back the MP4."""
        shape = _VIDEO_SHAPES.get(model)
        if shape is None:
            raise ConfigurationError(f"Venice video model {model!r} has no known constraints")
        if request.webhook_url is not None:
            raise ConfigurationError(
                "Venice video is awaited, not announced; it cannot register a webhook"
            )
        if request.image is not None:
            raise ConfigurationError(f"{model} is text-to-video and takes no first frame")
        if request.resolution is not None and request.resolution.upper() != shape.resolution:
            raise ConfigurationError(
                f"{model} offers resolution {shape.resolution.lower()}, not {request.resolution!r}"
            )
        # Venice refuses a request without a duration, and the price climbs
        # with it — USD 0.12 at 4s, 0.60 at 15s on MiniMax H3 Chili — so an
        # unstated one is the shortest clip, not a length nobody asked for.
        seconds = request.duration_seconds or shape.durations.start
        if seconds not in shape.durations:
            raise ConfigurationError(
                f"{model} accepts a duration of {shape.durations.start} to "
                f"{shape.durations.stop - 1} seconds, not {seconds}"
            )
        payload = {
            "model": model,
            "prompt": request.prompt,
            "duration": f"{seconds}s",
            "resolution": shape.resolution,
        }

        try:
            queued = await self._client.post("/video/queue", json=payload)
            if not isinstance(queued, dict) or not queued.get("queue_id"):
                raise ProviderError("Venice returned no queue id")
            video = await self._retrieve(
                model, str(queued["queue_id"]), download_url=queued.get("download_url")
            )
        except LLMGatewayError:
            raise
        except Exception as error:
            raise classify_provider_error(error) from error

        return ProviderVideoResponse(
            videos=(video,),
            # Exact rather than estimated: Venice takes whole seconds only and
            # prices the clip by the duration requested, not by what it renders.
            usage=VideoUsage(seconds=float(seconds), videos=1, resolution=shape.resolution.lower()),
            model_used=model,
        )

    async def _retrieve(self, model: str, queue_id: str, *, download_url: Any) -> GeneratedVideo:
        body = {"model": model, "queue_id": queue_id}

        async def post(path: str) -> dict[str, Any] | bytes:
            return cast("dict[str, Any] | bytes", await self._client.post(path, json=body))

        for attempt in range(self._max_poll_attempts):
            result = await read_status(
                post, "/video/retrieve", interval_seconds=self._poll_interval_seconds
            )
            if isinstance(result, bytes):
                return GeneratedVideo(data=result, mime_type="video/mp4")
            # Venice documents PROCESSING and COMPLETED only; a failed job comes
            # back as an HTTP error. Any other status is polled again, since
            # giving up on a job that is still running would not stop the bill.
            if result.get("status") == "COMPLETED":
                if isinstance(download_url, str) and download_url:
                    return GeneratedVideo(url=download_url, mime_type="video/mp4")
                raise ProviderError(f"Venice finished {model} but returned no video")
            if attempt + 1 < self._max_poll_attempts:
                await asyncio.sleep(self._poll_interval_seconds)

        raise TimeoutError("Venice video polling timed out")


def _messages(request: LLMRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system_prompt = system_prompt_for(request, structured_outputs=True)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend({"role": m.role, "content": m.content} for m in request.messages)
    return messages


def _without_unsupported_keys(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: _without_unsupported_keys(value)
            for key, value in node.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(node, list):
        return [_without_unsupported_keys(item) for item in node]
    return node


def _namespace(node: Any) -> Any:
    """The JSON usage block in the attribute shape the shared mapper reads."""
    if isinstance(node, dict):
        return SimpleNamespace(**{key: _namespace(value) for key, value in node.items()})
    return node


__all__ = ["CAPABILITIES", "VENICE_BASE_URL", "VeniceAdapter", "VeniceHttpClient"]

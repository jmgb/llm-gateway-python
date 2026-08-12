"""WaveSpeed image and video adapter, over its submit-and-poll REST API.

WaveSpeed ships no SDK, so the transport is the same small injected REST
client shape the AssemblyAI adapter uses. Options the API does not accept are
refused rather than dropped: an aspect ratio that silently became a square is
a bug the caller cannot see.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, cast

from llm_gateway.capabilities import ProviderCapabilities
from llm_gateway.contracts import LLMRequest
from llm_gateway.errors import ConfigurationError, LLMGatewayError, ProviderError
from llm_gateway.media import (
    GeneratedImage,
    GeneratedVideo,
    ImageInput,
    ImageRequest,
    ProviderImageResponse,
    ProviderVideoResponse,
    VideoRequest,
)
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.usage import ImageUsage, VideoUsage

CAPABILITIES = ProviderCapabilities(
    image_generation=True,
    image_editing=False,
    video_generation=True,
    video_from_image=True,
    conversation_history=False,
    reports_token_usage=False,
)

# The cheapest tier each video model offers, used when the caller states none.
# A model absent here is one whose tiers nobody read from WaveSpeed's docs, and
# it is left to the provider's own default rather than given an invented floor.
_LOWEST_RESOLUTION = {"wavespeed-ai/minimax-h3/image-to-video": "480p"}


class WaveSpeedHttpClient:
    """Small injected REST client; it owns the key, never the gateway."""

    def __init__(self, *, api_key: str, base_url: str = "https://api.wavespeed.ai") -> None:
        if not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", path, json=json)

    async def get(self, path: str) -> dict[str, Any]:
        return await self._request("GET", path)

    async def _request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        import httpx2

        async with httpx2.AsyncClient(timeout=120.0) as client:
            response = await client.request(
                method,
                f"{self._base_url}/{path.lstrip('/')}",
                headers=self._headers,
                json=json,
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())


class WaveSpeedAdapter:
    """Translates image and video requests into WaveSpeed's polling workflow.

    Every model lives at ``/api/v3/<model id>`` and answers with a task id, so
    the model id in the catalogue is also the path. Video takes minutes, which
    is why the polling budget defaults high enough to outlast a short clip.
    """

    name = "wavespeed"
    capabilities = CAPABILITIES

    def __init__(
        self,
        client: Any,
        *,
        max_poll_attempts: int = 300,
        poll_interval_seconds: float = 2.0,
    ) -> None:
        if max_poll_attempts < 1:
            raise ValueError("max_poll_attempts must be positive")
        if poll_interval_seconds < 0:
            raise ValueError("poll interval cannot be negative")
        self._client = client
        self._max_poll_attempts = max_poll_attempts
        self._poll_interval_seconds = poll_interval_seconds

    async def generate_image(self, request: ImageRequest, *, model: str) -> ProviderImageResponse:
        if request.image is not None:
            raise ConfigurationError("this WaveSpeed model does not support image editing")
        if request.aspect_ratio is not None:
            raise ConfigurationError("WaveSpeed sizes images explicitly, not by aspect ratio")

        try:
            submitted = await self._client.post(f"/api/v3/{model}", json={"prompt": request.prompt})
            _require_success(submitted)
            task_id = (submitted.get("data") or {}).get("id")
            if not isinstance(task_id, str) or not task_id:
                raise ProviderError("WaveSpeed returned no task id")

            outputs = await self._poll(task_id)
        except LLMGatewayError:
            raise
        except Exception as error:
            raise classify_provider_error(error) from None

        images = tuple(GeneratedImage(url=str(url)) for url in outputs if url)
        if not images:
            raise ProviderError(f"WaveSpeed returned no image for {model}")
        return ProviderImageResponse(
            images=images,
            usage=ImageUsage(images=len(images)),
            model_used=model,
        )

    async def generate_video(self, request: VideoRequest, *, model: str) -> ProviderVideoResponse:
        """Send prompt, first frame and clip settings, then poll for the MP4."""
        if model.endswith("/image-to-video") and request.image is None:
            raise ConfigurationError(f"{model} requires a first frame")
        payload: dict[str, Any] = {"prompt": request.prompt}
        if request.image is not None:
            payload["image"] = _image_reference(request.image)
        # An unstated resolution is the model's cheapest tier. Sending nothing
        # would let WaveSpeed pick, and 768p on MiniMax H3 costs twice 480p —
        # a bill the caller never asked for, and one the usage below could not
        # even price, since the resolution would come back unknown.
        resolution = request.resolution or _LOWEST_RESOLUTION.get(model)
        if resolution is not None:
            payload["resolution"] = resolution
        if request.duration_seconds is not None:
            payload["duration"] = request.duration_seconds

        try:
            submitted = await self._client.post(f"/api/v3/{model}", json=payload)
            _require_success(submitted)
            task_id = (submitted.get("data") or {}).get("id")
            if not isinstance(task_id, str) or not task_id:
                raise ProviderError("WaveSpeed returned no task id")

            outputs = await self._poll(task_id)
        except LLMGatewayError:
            raise
        except Exception as error:
            raise classify_provider_error(error) from None

        videos = tuple(
            GeneratedVideo(url=str(url), mime_type="video/mp4") for url in outputs if url
        )
        if not videos:
            raise ProviderError(f"WaveSpeed returned no video for {model}")
        return ProviderVideoResponse(
            videos=videos,
            usage=VideoUsage(
                seconds=float(request.duration_seconds) if request.duration_seconds else None,
                videos=len(videos),
                resolution=resolution,
                # WaveSpeed reports no clip length, and it snaps the output to
                # the model's frame grid, so the requested duration is an
                # estimate rather than a measurement. Cost degrades to
                # ESTIMATED instead of claiming a figure nobody measured.
                partial_aggregate=True,
            ),
            model_used=model,
        )

    async def _poll(self, task_id: str) -> list[Any]:
        for attempt in range(self._max_poll_attempts):
            result = await self._client.get(f"/api/v3/predictions/{task_id}/result")
            _require_success(result)
            data = result.get("data") or {}
            status = data.get("status")
            if status == "completed":
                return list(data.get("outputs") or [])
            if status in {"failed", "cancelled", "timeout"}:
                raise ProviderError(f"WaveSpeed generation ended with status {status}")
            if attempt + 1 < self._max_poll_attempts:
                await asyncio.sleep(self._poll_interval_seconds)

        raise TimeoutError("WaveSpeed generation polling timed out")

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise ConfigurationError("WaveSpeed is registered for media generation only")


def _image_reference(image: ImageInput) -> str:
    """A URL as given, or bytes as a data URI.

    WaveSpeed fetches a URL and also accepts an inline data URI, which is what
    makes "generate with one provider, animate with another" work without the
    caller having to host the intermediate frame anywhere.
    """
    if image.url is not None:
        return image.url
    assert image.data is not None  # guaranteed by ImageInput validation
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.mime_type or 'image/png'};base64,{encoded}"


def _require_success(response: dict[str, Any]) -> None:
    code = response.get("code")
    if code is not None and code != 200:
        raise ProviderError(f"WaveSpeed returned API code {code}")


__all__ = ["CAPABILITIES", "WaveSpeedAdapter", "WaveSpeedHttpClient"]

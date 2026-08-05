"""Replicate image adapter.

Replicate runs one model per call and answers with a URL, or with an SDK
``FileOutput`` wrapping one. Hosting the source image of an edit is the
application's job: Replicate fetches a URL and this adapter will not upload
bytes on the caller's behalf.
"""

from __future__ import annotations

from typing import Any

from llm_gateway.capabilities import ProviderCapabilities
from llm_gateway.contracts import LLMRequest
from llm_gateway.errors import ConfigurationError, LLMGatewayError, ProviderError
from llm_gateway.media import GeneratedImage, ImageRequest, ProviderImageResponse
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.usage import ImageUsage

CAPABILITIES = ProviderCapabilities(
    image_generation=True,
    image_editing=True,
    conversation_history=False,
    reports_token_usage=False,
)


class ReplicateAdapter:
    """Translates an image request to ``client.async_run``."""

    name = "replicate"
    capabilities = CAPABILITIES

    def __init__(self, client: Any) -> None:
        self._client = client

    async def generate_image(self, request: ImageRequest, *, model: str) -> ProviderImageResponse:
        payload: dict[str, Any] = {"prompt": request.prompt}
        if request.image is not None:
            if request.image.url is None:
                raise ConfigurationError(
                    "Replicate needs the source image as a URL it can fetch, not as bytes"
                )
            payload["input_image"] = request.image.url
        if request.aspect_ratio is not None:
            payload["aspect_ratio"] = request.aspect_ratio

        try:
            output = await self._client.async_run(model, input=payload)
        except LLMGatewayError:
            raise
        except Exception as error:
            raise classify_provider_error(error) from None

        images = _images(output)
        if not images:
            raise ProviderError(f"Replicate returned no image for {model}")
        return ProviderImageResponse(
            images=images,
            usage=ImageUsage(images=len(images)),
            model_used=model,
        )

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise ConfigurationError("Replicate is registered for image generation only")


def _images(output: Any) -> tuple[GeneratedImage, ...]:
    """Normalise the three shapes the SDK returns: object, list, or plain string."""
    if output is None:
        return ()
    candidates = output if isinstance(output, list | tuple) else [output]
    urls = [url for url in (_url(candidate) for candidate in candidates) if url]
    return tuple(GeneratedImage(url=url) for url in urls)


def _url(candidate: Any) -> str | None:
    if isinstance(candidate, str):
        return candidate or None
    url = getattr(candidate, "url", None)
    return str(url) if url else None


__all__ = ["CAPABILITIES", "ReplicateAdapter"]

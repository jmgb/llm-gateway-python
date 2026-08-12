"""Replicate image and video adapter.

Replicate runs one model per call and answers with a URL, or with an SDK
``FileOutput`` wrapping one. Hosting the source image of an edit is the
application's job: Replicate fetches a URL and this adapter will not upload
bytes on the caller's behalf.

Image and video use different halves of the API for a reason that is not
stylistic. An image finishes inside one ``async_run``; a video is a
*prediction* that keeps running for minutes after the call returns, so it goes
through ``predictions.async_create`` and is read back later — by a poll or by
the webhook Replicate calls when it finishes.

The ``async_`` prefixes are load-bearing. The SDK exposes both halves under
similar names — ``predictions.create`` is a blocking call that returns a
``Prediction``, ``predictions.async_create`` is the coroutine — and awaiting
the synchronous one raises ``TypeError`` at runtime while looking correct on
the page.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from llm_gateway.capabilities import ProviderCapabilities
from llm_gateway.contracts import LLMRequest
from llm_gateway.errors import ConfigurationError, LLMGatewayError, ProviderError
from llm_gateway.media import (
    GeneratedImage,
    GeneratedVideo,
    ImageInput,
    ImageRequest,
    ProviderImageResponse,
    ProviderVideoJobUpdate,
    VideoJob,
    VideoJobStatus,
    VideoRequest,
)
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.usage import ImageUsage, VideoUsage

CAPABILITIES = ProviderCapabilities(
    image_generation=True,
    image_editing=True,
    video_generation=True,
    video_from_image=True,
    video_webhooks=True,
    conversation_history=False,
    reports_token_usage=False,
)

# Replicate's prediction states, mapped onto the neutral five. Absent on
# purpose is any default: a state this table does not know is a state whose
# meaning nobody verified, and reading it as "still working" polls a dead job
# until the application gives up on its own.
_JOB_STATUS = {
    "starting": VideoJobStatus.QUEUED,
    "processing": VideoJobStatus.RUNNING,
    "succeeded": VideoJobStatus.SUCCEEDED,
    "failed": VideoJobStatus.FAILED,
    "canceled": VideoJobStatus.CANCELLED,
}


@dataclass(frozen=True, slots=True)
class _VideoShape:
    """How one Replicate model spells the options this package can express.

    Replicate is a host, not an API: every model publishes its own input
    schema, and they disagree about all three options here. The wrong key is
    the expensive kind of wrong — Replicate ignores what it does not recognise,
    generates the default, and bills for it, so nothing surfaces until someone
    compares the clip to what was asked for.

    Read from each model's published schema, not inferred from the others.
    """

    image_field: str | None = "image"
    duration_field: str | None = None
    resolution_field: str | None = "resolution"
    resolutions: Mapping[str, str] | None = None
    """Neutral resolution to the model's own spelling. ``None`` passes it through."""
    default_resolution: str | None = None
    """Sent when the caller states none: always the cheapest tier the model has.

    Not the provider's own default, which on every model here is dearer —
    720p for Wan and Seedance, ``pro`` (1080p) for Kling. Silence should not
    buy the expensive one.
    """


_VIDEO_SHAPES = {
    # Sizes a clip in frames at a frame rate, so it has no seconds field.
    "wan-video/wan-2.2-5b-fast": _VideoShape(
        resolutions={"480p": "480p", "720p": "720p"},
        default_resolution="480p",
    ),
    # Has no resolution field at all: 720p/1080p/4K are `mode` values.
    "kwaivgi/kling-v3-video": _VideoShape(
        image_field="start_image",
        duration_field="duration",
        resolution_field="mode",
        resolutions={"720p": "standard", "1080p": "pro", "4k": "4k"},
        default_resolution="720p",
    ),
    # 2.5 dropped the 1080p and 4K tiers its 2.0 predecessor offered.
    "bytedance/seedance-2.5": _VideoShape(
        duration_field="duration",
        resolutions={"480p": "480p", "720p": "720p"},
        default_resolution="480p",
    ),
}

# What every video model on Replicate has been observed to share. A model
# nobody verified gets only this, because sending an unverified key is how an
# option gets silently dropped — and it gets no default resolution either,
# since a floor nobody read from a schema is a guess.
_DEFAULT_VIDEO_SHAPE = _VideoShape(resolution_field=None)


class ReplicateAdapter:
    """Translates an image request to ``client.async_run``, and video to a prediction."""

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

    async def submit_video(self, request: VideoRequest, *, model: str) -> VideoJob:
        """Create a prediction and hand back its id, without waiting for the clip."""
        shape = _VIDEO_SHAPES.get(model, _DEFAULT_VIDEO_SHAPE)
        payload: dict[str, Any] = {"prompt": request.prompt}
        if request.image is not None:
            if shape.image_field is None:
                raise ConfigurationError(f"{model} does not animate a first frame")
            payload[shape.image_field] = _frame_reference(request.image)
        if request.duration_seconds is not None:
            if shape.duration_field is None:
                # Wan sizes a clip in frames at a frame rate, so seconds are
                # two numbers this adapter would have to invent. Sending
                # nothing would quietly generate the default length instead.
                raise ConfigurationError(f"{model} takes a frame count, not a duration in seconds")
            payload[shape.duration_field] = request.duration_seconds
        # An unstated resolution is the model's cheapest tier, not the
        # provider's default, which is dearer on every model catalogued here.
        resolution = request.resolution or shape.default_resolution
        if resolution is not None:
            if shape.resolution_field is None:
                raise ConfigurationError(f"{model} has no verified resolution option")
            payload[shape.resolution_field] = _resolution_for(shape, resolution, model)

        create: dict[str, Any] = {"model": model, "input": payload}
        if request.webhook_url is not None:
            create["webhook"] = request.webhook_url
            # Only the terminal event. Every intermediate one costs the
            # application a request it would answer by doing nothing.
            create["webhook_events_filter"] = ["completed"]

        try:
            prediction = await self._client.predictions.async_create(**create)
        except LLMGatewayError:
            raise
        except Exception as error:
            raise classify_provider_error(error) from None

        job_id = str(getattr(prediction, "id", "") or "")
        if not job_id:
            # The prediction may well be running and billable; it is simply
            # unreachable, which is worth an error rather than a silent leak.
            raise ProviderError(f"Replicate returned no prediction id for {model}")
        return VideoJob(
            id=job_id,
            model=model,
            provider=self.name,
            status=_status(getattr(prediction, "status", None)),
        )

    async def poll_video(self, job: VideoJob) -> ProviderVideoJobUpdate:
        """Read one prediction back. Raises only if the read itself failed."""
        try:
            prediction = await self._client.predictions.async_get(job.id)
        except LLMGatewayError:
            raise
        except Exception as error:
            raise classify_provider_error(error) from None

        status = _status(getattr(prediction, "status", None))
        if status is not VideoJobStatus.SUCCEEDED:
            error_text = getattr(prediction, "error", None)
            return ProviderVideoJobUpdate(
                status=status,
                error=str(error_text) if error_text else None,
            )

        videos = _videos(getattr(prediction, "output", None))
        metrics = getattr(prediction, "metrics", None) or {}
        return ProviderVideoJobUpdate(
            status=status,
            videos=videos,
            usage=VideoUsage(
                # Measured by Replicate on the clip it just produced, so this
                # is actual usage rather than the duration someone requested.
                # Absent it stays unknown — never zero, which would price a
                # real invoice at nothing.
                seconds=_measured_seconds(metrics),
                videos=len(videos),
                resolution=_measured_resolution(metrics, job.model),
            ),
        )

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise ConfigurationError("Replicate is registered for media generation only")


def _measured_seconds(metrics: Any) -> float | None:
    """The clip length Replicate measured, or ``None`` when it reported none.

    Anything unusable — missing, negative, not a number — is unknown rather
    than coerced, because a wrong length prices a real invoice wrongly and a
    missing one only leaves it unpriced.
    """
    if not isinstance(metrics, dict):
        return None
    try:
        seconds = float(metrics.get("video_output_duration_seconds"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 and math.isfinite(seconds) else None


def _measured_resolution(metrics: Any, model: str) -> str | None:
    """The tier that actually ran, back in the package's own spelling.

    Kling reports ``model_variant`` — its own name for the tier, the same one
    the request sent as ``mode``. Translating it back is what lets a
    resolution-keyed price table find the rate that was really charged. A
    variant the model does not declare stays unknown rather than being mapped
    to the nearest guess.
    """
    if not isinstance(metrics, dict):
        return None
    variant = metrics.get("model_variant")
    shape = _VIDEO_SHAPES.get(model)
    if not isinstance(variant, str) or shape is None or shape.resolutions is None:
        return None
    for neutral, provider_name in shape.resolutions.items():
        if provider_name == variant:
            return neutral
    return None


def _frame_reference(image: ImageInput) -> str:
    """A URL as given, or bytes as a data URI.

    Replicate fetches a URL and also accepts an inline data URI, which is what
    makes "generate the frame with one provider, animate it with another" work
    without the caller hosting the intermediate image anywhere. The image
    *editing* path above still refuses bytes; that endpoint takes a URL it
    fetches, and nothing has asked it to change.
    """
    if image.url is not None:
        return image.url
    assert image.data is not None  # guaranteed by ImageInput validation
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.mime_type or 'image/png'};base64,{encoded}"


def _resolution_for(shape: _VideoShape, resolution: str, model: str) -> str:
    """The model's own spelling of a neutral resolution.

    Refused rather than defaulted when the model has no such size: silently
    generating 1080p for a caller who asked for 360p bills the expensive rate
    for something nobody chose.
    """
    if shape.resolutions is None:
        return resolution
    translated = shape.resolutions.get(resolution)
    if translated is None:
        raise ConfigurationError(
            f"{model} does not offer resolution {resolution!r}; it has {sorted(shape.resolutions)}"
        )
    return translated


def _status(raw: Any) -> VideoJobStatus:
    status = _JOB_STATUS.get(str(raw))
    if status is None:
        raise ProviderError(f"Replicate reported unknown prediction status {raw!r}")
    return status


def _videos(output: Any) -> tuple[GeneratedVideo, ...]:
    if output is None:
        return ()
    candidates = output if isinstance(output, list | tuple) else [output]
    urls = [url for url in (_url(candidate) for candidate in candidates) if url]
    return tuple(GeneratedVideo(url=url, mime_type="video/mp4") for url in urls)


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

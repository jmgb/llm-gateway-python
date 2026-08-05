"""Provider-neutral image and video generation contracts.

An image is not a token message, so it does not travel through ``LLMRequest``:
its output is bytes or a URL, and its price is per image for some providers and
per token for others. Modelling it as a text call would force the result type
to lie about one of the two.

What stays with the application, exactly as it does for audio: downloading a
URL, re-hosting it, watermarking, per-user quotas, moderation policy and the
prompt engineering that turns a user's idea into a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_gateway.contracts import AttemptOutcome, FailurePhase
from llm_gateway.policies import FallbackPolicy, RetryPolicy, TimeoutPolicy
from llm_gateway.pricing import ImageCost, VideoCost
from llm_gateway.usage import ImageUsage, VideoUsage


@dataclass(frozen=True, slots=True)
class ImageInput:
    """The starting image of an edit, as bytes or as a fetchable URL.

    Both exist because providers accept different ones and neither converts
    the other for free: Gemini takes inline bytes, Replicate takes a URL it
    fetches itself. An adapter rejects the form it cannot use rather than
    downloading or hosting on the caller's behalf.
    """

    data: bytes | None = None
    url: str | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if self.data is None and self.url is None:
            raise ValueError("image data or url is required")
        if self.data is not None and not self.data:
            raise ValueError("image data must be non-empty")
        if self.url is not None and not self.url.strip():
            raise ValueError("image url must be non-empty")


@dataclass(frozen=True, slots=True)
class ImageRequest:
    """One image generation or edit, separate from ``LLMRequest``."""

    model: str
    prompt: str
    image: ImageInput | None = None
    """Present for an edit, absent for a generation. Not every provider edits."""
    aspect_ratio: str | None = None
    timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.disabled)
    fallback_policy: FallbackPolicy = field(default_factory=FallbackPolicy.disabled)
    request_id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("an image model identifier is required")
        if not self.prompt.strip():
            raise ValueError("an image prompt must be non-empty")
        if self.aspect_ratio is not None and not self.aspect_ratio.strip():
            raise ValueError("aspect ratio must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """One returned image. Bytes from some providers, a URL from others."""

    data: bytes | None = None
    url: str | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if self.data is None and self.url is None:
            raise ValueError("a generated image needs data or url")
        if self.data is not None and not self.data:
            raise ValueError("generated image data must be non-empty")
        if self.url is not None and not self.url.strip():
            raise ValueError("generated image url must be non-empty")


@dataclass(frozen=True, slots=True)
class ProviderImageResponse:
    """One provider response before gateway accounting and orchestration."""

    images: tuple[GeneratedImage, ...]
    usage: ImageUsage
    model_used: str | None = None


@dataclass(frozen=True, slots=True)
class ImageAttempt:
    """One billable or configuration-failed image attempt."""

    index: int
    model: str
    provider: str
    outcome: AttemptOutcome
    usage: ImageUsage
    cost: ImageCost
    latency_ms: int
    error_type: str | None = None
    billable: bool = True
    failure_phase: FailurePhase | None = None


@dataclass(frozen=True, slots=True)
class ImageExecution:
    """What happened during an image call."""

    requested_model: str
    model_used: str
    provider: str
    attempts: tuple[ImageAttempt, ...]
    latency_ms: int

    @property
    def fallback_used(self) -> bool:
        return bool(self.attempts and self.attempts[-1].model != self.requested_model)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


@dataclass(frozen=True, slots=True)
class ImageResult:
    """Images and image accounting, kept separate from token results."""

    images: tuple[GeneratedImage, ...]
    usage: ImageUsage
    execution: ImageExecution
    cost: ImageCost

    @property
    def output(self) -> tuple[GeneratedImage, ...]:
        return self.images


@dataclass(frozen=True, slots=True)
class VideoRequest:
    """One video generation, from a prompt and optionally a first frame.

    Deliberately synchronous from the caller's side: the adapter owns the
    provider's polling loop, exactly as the transcription adapters do, and
    ``timeout_policy.total_seconds`` is the whole budget. Providers that
    answer through a webhook — Replicate, Sora — will need a two-phase submit
    and poll contract instead, and that is a separate task.
    """

    model: str
    prompt: str
    image: ImageInput | None = None
    """The first frame. Present for image-to-video, absent for text-to-video."""
    duration_seconds: int | None = None
    resolution: str | None = None
    timeout_policy: TimeoutPolicy = field(
        default_factory=lambda: TimeoutPolicy(total_seconds=900.0)
    )
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.disabled)
    fallback_policy: FallbackPolicy = field(default_factory=FallbackPolicy.disabled)
    request_id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("a video model identifier is required")
        if not self.prompt.strip():
            raise ValueError("a video prompt must be non-empty")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("video duration must be positive")
        if self.resolution is not None and not self.resolution.strip():
            raise ValueError("resolution must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class GeneratedVideo:
    """One returned video. A URL from every provider seen so far."""

    url: str | None = None
    data: bytes | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if self.data is None and self.url is None:
            raise ValueError("a generated video needs data or url")
        if self.data is not None and not self.data:
            raise ValueError("generated video data must be non-empty")
        if self.url is not None and not self.url.strip():
            raise ValueError("generated video url must be non-empty")


@dataclass(frozen=True, slots=True)
class ProviderVideoResponse:
    """One provider response before gateway accounting and orchestration."""

    videos: tuple[GeneratedVideo, ...]
    usage: VideoUsage
    model_used: str | None = None


@dataclass(frozen=True, slots=True)
class VideoAttempt:
    """One billable or configuration-failed video attempt."""

    index: int
    model: str
    provider: str
    outcome: AttemptOutcome
    usage: VideoUsage
    cost: VideoCost
    latency_ms: int
    error_type: str | None = None
    billable: bool = True
    failure_phase: FailurePhase | None = None


@dataclass(frozen=True, slots=True)
class VideoExecution:
    """What happened during a video call."""

    requested_model: str
    model_used: str
    provider: str
    attempts: tuple[VideoAttempt, ...]
    latency_ms: int

    @property
    def fallback_used(self) -> bool:
        return bool(self.attempts and self.attempts[-1].model != self.requested_model)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


@dataclass(frozen=True, slots=True)
class VideoResult:
    """Videos and video accounting, kept separate from token results."""

    videos: tuple[GeneratedVideo, ...]
    usage: VideoUsage
    execution: VideoExecution
    cost: VideoCost

    @property
    def output(self) -> tuple[GeneratedVideo, ...]:
        return self.videos

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
from enum import Enum

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

    Serves both shapes of provider. Where the adapter can own the provider's
    polling loop, ``LLMGateway.generate_video()`` waits and
    ``timeout_policy.total_seconds`` is the whole budget. Where the provider
    answers with a job id minutes before the clip exists, the same request
    goes through ``submit_video()`` and ``poll_video()`` instead, and the
    timeout policy then bounds only the submission.
    """

    model: str
    prompt: str
    image: ImageInput | None = None
    """The first frame. Present for image-to-video, absent for text-to-video."""
    duration_seconds: int | None = None
    resolution: str | None = None
    """Left unset, the adapter sends the **cheapest tier its model offers**.

    Not the provider's own default, which is dearer on every catalogued video
    model: 720p for Wan and Seedance, ``pro`` (1080p) for Kling, and on
    WaveSpeed a 768p clip that costs twice its 480p one. Resolution is the
    single biggest lever on a video bill, so silence buys the floor and
    anything above it has to be asked for by name.

    A model whose tiers this package has not read from a published schema gets
    no default at all — an invented floor is a guess, and a rejected request.
    """
    webhook_url: str | None = None
    """Where a job-shaped provider should announce that it finished.

    Only the registration belongs here. Receiving the callback, verifying its
    signature and deciding what it triggers are the application's, and an
    adapter whose provider has no webhook refuses the field rather than
    dropping it — a caller who thinks it registered one waits forever.
    """
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
        if self.webhook_url is not None and not self.webhook_url.strip():
            raise ValueError("webhook url must be non-empty when provided")


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


class VideoJobStatus(Enum):
    """Where a submitted generation is, normalised across providers.

    Deliberately smaller than any provider's vocabulary. An application
    branches on "can I read the clip yet", and every extra state is one more
    branch each consumer has to get right for a distinction none of them use.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether polling again can still change the answer."""
        return self in {
            VideoJobStatus.SUCCEEDED,
            VideoJobStatus.FAILED,
            VideoJobStatus.CANCELLED,
        }


@dataclass(frozen=True, slots=True)
class VideoJob:
    """A submitted generation whose clip does not exist yet.

    Plain, storable data on purpose. Minutes pass between submission and
    result, so the process that polls is usually not the one that submitted:
    a worker reading a database row, or a webhook handler holding nothing but
    what it saved. Everything needed to poll fits in four strings.

    ``model`` and ``provider`` are the ones that *hold* the job, which after a
    fallback is not the model that was requested. Polling the requested one
    would ask the wrong provider for an id it never issued.
    """

    id: str
    model: str
    provider: str
    status: VideoJobStatus = VideoJobStatus.QUEUED

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("a video job id is required to poll it later")
        if not self.model.strip():
            raise ValueError("a video job records the model that holds it")
        if not self.provider.strip():
            raise ValueError("a video job records the provider that holds it")

    def with_status(self, status: VideoJobStatus) -> VideoJob:
        return VideoJob(id=self.id, model=self.model, provider=self.provider, status=status)


@dataclass(frozen=True, slots=True)
class ProviderVideoJobUpdate:
    """One reading of a job's state, before gateway pricing and accounting."""

    status: VideoJobStatus = VideoJobStatus.QUEUED
    videos: tuple[GeneratedVideo, ...] = ()
    usage: VideoUsage = field(default_factory=VideoUsage)
    error: str | None = None
    """Why the provider gave up, when it says. Never sent to a sink."""


@dataclass(frozen=True, slots=True)
class VideoJobResult:
    """What one poll found: a status, and a clip once there is one."""

    job: VideoJob
    videos: tuple[GeneratedVideo, ...] = ()
    usage: VideoUsage = field(default_factory=VideoUsage)
    cost: VideoCost = field(default_factory=VideoCost.unavailable)
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.job.status.is_terminal

    @property
    def output(self) -> tuple[GeneratedVideo, ...]:
        return self.videos

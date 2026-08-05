"""Token accounting.

The central invariant: absence of usage is never reported as zero usage. A
provider that returns nothing produces ``TokenUsage.unknown()``, whose
``complete`` flag is ``False``; a provider that genuinely reports zero produces
a complete measurement. Cost estimation depends on that distinction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


def _add(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Tokens reported by a provider. ``None`` means "not reported"."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    """A breakdown of ``output_tokens``, never an addition to it."""
    retrieved_document_tokens: int | None = None
    cached_input_tokens: int | None = None
    partial_aggregate: bool = False
    """Set when this total absorbed an attempt that reported nothing."""

    def __post_init__(self) -> None:
        counts = (
            self.input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.retrieved_document_tokens,
            self.cached_input_tokens,
        )
        if any(count is not None and count < 0 for count in counts):
            raise ValueError("token counts must be non-negative")
        if (
            self.output_tokens is not None
            and self.reasoning_tokens is not None
            and self.reasoning_tokens > self.output_tokens
        ):
            raise ValueError("reasoning tokens cannot exceed output tokens")

    @classmethod
    def unknown(cls) -> TokenUsage:
        """Usage the provider did not report at all."""
        return cls()

    @property
    def complete(self) -> bool:
        """True only when every billable dimension of every attempt was reported."""
        if self.partial_aggregate:
            return False
        return self.input_tokens is not None and self.output_tokens is not None

    @property
    def billable_input_tokens(self) -> int | None:
        """Input plus retrieved documents, which are billed as input context."""
        if self.input_tokens is None:
            return None
        return self.input_tokens + (self.retrieved_document_tokens or 0)

    @property
    def billable_output_tokens(self) -> int | None:
        """The output count, which already contains the reasoning tokens.

        Providers disagree about where thinking is counted, so adapters
        normalise it at the boundary: whatever is billed at the output rate is
        inside ``output_tokens`` by the time it reaches this class. Adding
        ``reasoning_tokens`` here would bill them a second time.
        """
        return self.output_tokens

    @property
    def visible_output_tokens(self) -> int | None:
        """The part of the output the caller actually received back."""
        if self.output_tokens is None:
            return None
        return self.output_tokens - (self.reasoning_tokens or 0)

    def merge(self, other: TokenUsage) -> TokenUsage:
        """Aggregate two attempts. An unknown operand taints the total."""
        return replace(
            self,
            input_tokens=_add(self.input_tokens, other.input_tokens),
            output_tokens=_add(self.output_tokens, other.output_tokens),
            reasoning_tokens=_add(self.reasoning_tokens, other.reasoning_tokens),
            retrieved_document_tokens=_add(
                self.retrieved_document_tokens, other.retrieved_document_tokens
            ),
            cached_input_tokens=_add(self.cached_input_tokens, other.cached_input_tokens),
            partial_aggregate=not (self.complete and other.complete),
        )


@dataclass(frozen=True, slots=True)
class ImageUsage:
    """What an image generation produced, in whichever unit bills it.

    Providers disagree about the unit: Replicate and WaveSpeed charge per
    image, Gemini charges the same tokens as a text call. Both are reported,
    and the price catalogue decides which one applies to the model — so an
    adapter never has to know how its provider is billed.
    """

    images: int | None = None
    tokens: TokenUsage | None = None
    """Only providers that bill image generation as tokens report this."""
    partial_aggregate: bool = False

    def __post_init__(self) -> None:
        if self.images is not None and self.images < 0:
            raise ValueError("image count must be non-negative")

    @classmethod
    def unknown(cls) -> ImageUsage:
        return cls()

    @property
    def complete(self) -> bool:
        if self.partial_aggregate or self.images is None:
            return False
        return self.tokens is None or self.tokens.complete

    def merge(self, other: ImageUsage) -> ImageUsage:
        images = _add(self.images, other.images)
        if self.tokens is None:
            tokens = other.tokens
        elif other.tokens is None:
            tokens = self.tokens
        else:
            tokens = self.tokens.merge(other.tokens)
        return ImageUsage(
            images=images,
            tokens=tokens,
            partial_aggregate=not (self.complete and other.complete),
        )


@dataclass(frozen=True, slots=True)
class VideoUsage:
    """What a video generation produced, in the unit that bills it.

    Seconds are the billable dimension for every provider seen so far, and
    ``resolution`` is part of usage rather than of the request because the
    rate depends on it: MiniMax H3 costs twice as much at 768p as at 480p, and
    a provider may snap the clip to its own frame grid.
    """

    seconds: float | None = None
    videos: int | None = None
    resolution: str | None = None
    partial_aggregate: bool = False

    def __post_init__(self) -> None:
        if self.seconds is not None and (self.seconds < 0 or not math.isfinite(self.seconds)):
            raise ValueError("video seconds must be non-negative and finite")
        if self.videos is not None and self.videos < 0:
            raise ValueError("video count must be non-negative")

    @classmethod
    def unknown(cls) -> VideoUsage:
        return cls()

    @property
    def complete(self) -> bool:
        return self.seconds is not None and not self.partial_aggregate

    def merge(self, other: VideoUsage) -> VideoUsage:
        if self.seconds is None and other.seconds is None:
            seconds = None
        else:
            seconds = (self.seconds or 0.0) + (other.seconds or 0.0)
        resolutions_conflict = (
            self.resolution is not None
            and other.resolution is not None
            and self.resolution != other.resolution
        )
        return VideoUsage(
            seconds=seconds,
            videos=_add(self.videos, other.videos),
            resolution=(self.resolution if self.resolution == other.resolution else None),
            partial_aggregate=not (self.complete and other.complete) or resolutions_conflict,
        )


@dataclass(frozen=True, slots=True)
class AudioUsage:
    """Audio duration reported by a transcription provider.

    ``None`` means the provider did not report duration. It is intentionally a
    different type from ``TokenUsage`` so a transcription cannot enter token
    pricing by accident.
    """

    duration_seconds: float | None = None
    partial_aggregate: bool = False

    def __post_init__(self) -> None:
        if self.duration_seconds is not None and (
            self.duration_seconds < 0 or not math.isfinite(self.duration_seconds)
        ):
            raise ValueError("audio duration must be non-negative and finite")

    @classmethod
    def unknown(cls) -> AudioUsage:
        return cls()

    @property
    def complete(self) -> bool:
        return self.duration_seconds is not None and not self.partial_aggregate

    def merge(self, other: AudioUsage) -> AudioUsage:
        if self.duration_seconds is None and other.duration_seconds is None:
            duration = None
        else:
            duration = (self.duration_seconds or 0.0) + (other.duration_seconds or 0.0)
        return AudioUsage(
            duration_seconds=duration,
            partial_aggregate=not (self.complete and other.complete),
        )

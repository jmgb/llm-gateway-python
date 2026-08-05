"""Cost accounting.

Three rules the rest of the package depends on:

* an unknown cost is ``UNAVAILABLE``, never ``USD 0``;
* usage that is missing a billable dimension yields an ``ESTIMATED`` lower
  bound, never a silent ``ACTUAL``;
* arithmetic happens in whole microdollars so that totals reconcile exactly.

The gateway uses the package's versioned catalogue by default. Applications
can inject a different catalogue for negotiated or dynamic rates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from llm_gateway.usage import AudioUsage, ImageUsage, TokenUsage, VideoUsage

_MICRO = Decimal(1_000_000)


class CostMeasurement(Enum):
    """How much confidence the amount deserves."""

    ACTUAL = "ACTUAL"
    """Every billable dimension was reported and priced."""

    ESTIMATED = "ESTIMATED"
    """A lower bound: something billable was missing or unpriced."""

    UNAVAILABLE = "UNAVAILABLE"
    """No amount could be computed. Not the same as free."""


@dataclass(frozen=True, slots=True)
class ModelRate:
    """Price per token, expressed in microdollars to avoid float drift."""

    input_microusd_per_token: Decimal
    output_microusd_per_token: Decimal

    def __post_init__(self) -> None:
        rates = (self.input_microusd_per_token, self.output_microusd_per_token)
        if any(not rate.is_finite() or rate < 0 for rate in rates):
            raise ValueError("token rates must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class AudioRate:
    """Price per audio minute, with provider-specific minimum billing."""

    usd_per_minute: Decimal
    minimum_billable_seconds: int = 0

    def __post_init__(self) -> None:
        if not self.usd_per_minute.is_finite():
            raise ValueError("audio price must be finite")
        if self.usd_per_minute < 0:
            raise ValueError("audio price cannot be negative")
        if self.minimum_billable_seconds < 0:
            raise ValueError("audio minimum cannot be negative")


@dataclass(frozen=True, slots=True)
class Cost:
    """An amount plus the honesty about how it was obtained."""

    measurement: CostMeasurement
    microusd: int | None = None
    pricing_version: str | None = None

    @classmethod
    def unavailable(cls, *, pricing_version: str | None = None) -> Cost:
        return cls(measurement=CostMeasurement.UNAVAILABLE, pricing_version=pricing_version)

    @property
    def amount_usd(self) -> Decimal | None:
        """Serialised with six decimals; ``None`` when unavailable."""
        if self.microusd is None:
            return None
        return (Decimal(self.microusd) / _MICRO).quantize(Decimal("0.000001"))

    def merge(self, other: Cost) -> Cost:
        """Aggregate billable attempts, keeping the weakest measurement."""
        amounts = [c.microusd for c in (self, other) if c.microusd is not None]
        total = sum(amounts) if amounts else None
        if self.measurement is other.measurement:
            measurement = self.measurement
        elif total is None:
            measurement = CostMeasurement.UNAVAILABLE
        else:
            measurement = CostMeasurement.ESTIMATED
        return Cost(
            measurement=measurement,
            microusd=total,
            pricing_version=self.pricing_version or other.pricing_version,
        )


@dataclass(frozen=True, slots=True)
class AudioCost:
    """Audio amount kept separate from token ``Cost``."""

    measurement: CostMeasurement
    microusd: int | None = None
    pricing_version: str | None = None

    @classmethod
    def unavailable(cls, *, pricing_version: str | None = None) -> AudioCost:
        return cls(measurement=CostMeasurement.UNAVAILABLE, pricing_version=pricing_version)

    @property
    def amount_usd(self) -> Decimal | None:
        if self.microusd is None:
            return None
        return (Decimal(self.microusd) / _MICRO).quantize(Decimal("0.000001"))

    def merge(self, other: AudioCost) -> AudioCost:
        amounts = [c.microusd for c in (self, other) if c.microusd is not None]
        total = sum(amounts) if amounts else None
        if self.measurement is other.measurement:
            measurement = self.measurement
        elif total is None:
            measurement = CostMeasurement.UNAVAILABLE
        else:
            measurement = CostMeasurement.ESTIMATED
        return AudioCost(
            measurement=measurement,
            microusd=total,
            pricing_version=self.pricing_version or other.pricing_version,
        )


@dataclass(frozen=True, slots=True)
class ImageRate:
    """Price for one generated image, in whichever unit the provider bills.

    Exactly one of the two applies per model, and which one is a fact about
    the provider: a per-image charge for Replicate and WaveSpeed, the ordinary
    token rate for Gemini, whose image models bill the picture as output
    tokens. A rate that declares neither would price every image at nothing.
    """

    usd_per_image: Decimal | None = None
    token_rate: ModelRate | None = None

    def __post_init__(self) -> None:
        if (self.usd_per_image is None) == (self.token_rate is None):
            raise ValueError("an image rate needs exactly one per-image price or token rate")
        if self.usd_per_image is not None and not self.usd_per_image.is_finite():
            raise ValueError("image price must be finite")
        if self.usd_per_image is not None and self.usd_per_image < 0:
            raise ValueError("image price cannot be negative")


@dataclass(frozen=True, slots=True)
class ImageCost:
    """Image amount kept separate from token and audio cost."""

    measurement: CostMeasurement
    microusd: int | None = None
    pricing_version: str | None = None

    @classmethod
    def unavailable(cls, *, pricing_version: str | None = None) -> ImageCost:
        return cls(measurement=CostMeasurement.UNAVAILABLE, pricing_version=pricing_version)

    @property
    def amount_usd(self) -> Decimal | None:
        if self.microusd is None:
            return None
        return (Decimal(self.microusd) / _MICRO).quantize(Decimal("0.000001"))

    def merge(self, other: ImageCost) -> ImageCost:
        amounts = [c.microusd for c in (self, other) if c.microusd is not None]
        total = sum(amounts) if amounts else None
        if self.measurement is other.measurement:
            measurement = self.measurement
        elif total is None:
            measurement = CostMeasurement.UNAVAILABLE
        else:
            measurement = CostMeasurement.ESTIMATED
        return ImageCost(
            measurement=measurement,
            microusd=total,
            pricing_version=self.pricing_version or other.pricing_version,
        )


@dataclass(frozen=True, slots=True)
class VideoRate:
    """Price per second of generated video, which resolution can change.

    ``usd_per_second_by_resolution`` wins when the provider reports one. A
    reported resolution the table does not know yields no amount rather than
    the default rate: MiniMax H3 costs double at 768p, so guessing the cheaper
    number would halve a real invoice.
    """

    usd_per_second: Decimal | None = None
    usd_per_second_by_resolution: Mapping[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.usd_per_second is None and not self.usd_per_second_by_resolution:
            raise ValueError("a video rate needs a per-second price")
        if self.usd_per_second is not None and not self.usd_per_second.is_finite():
            raise ValueError("video price must be finite")
        if self.usd_per_second is not None and self.usd_per_second < 0:
            raise ValueError("video price cannot be negative")
        rates = dict(self.usd_per_second_by_resolution)
        if any(not resolution.strip() for resolution in rates):
            raise ValueError("video resolution names must be non-empty")
        if any(not price.is_finite() for price in rates.values()):
            raise ValueError("video resolution price must be finite")
        if any(price < 0 for price in rates.values()):
            raise ValueError("video resolution price cannot be negative")
        object.__setattr__(self, "usd_per_second_by_resolution", MappingProxyType(rates))

    def for_resolution(self, resolution: str | None) -> Decimal | None:
        if not self.usd_per_second_by_resolution:
            return self.usd_per_second
        if resolution is None:
            return self.usd_per_second
        return self.usd_per_second_by_resolution.get(resolution)


@dataclass(frozen=True, slots=True)
class VideoCost:
    """Video amount kept separate from token, audio and image cost."""

    measurement: CostMeasurement
    microusd: int | None = None
    pricing_version: str | None = None

    @classmethod
    def unavailable(cls, *, pricing_version: str | None = None) -> VideoCost:
        return cls(measurement=CostMeasurement.UNAVAILABLE, pricing_version=pricing_version)

    @property
    def amount_usd(self) -> Decimal | None:
        if self.microusd is None:
            return None
        return (Decimal(self.microusd) / _MICRO).quantize(Decimal("0.000001"))

    def merge(self, other: VideoCost) -> VideoCost:
        amounts = [c.microusd for c in (self, other) if c.microusd is not None]
        total = sum(amounts) if amounts else None
        if self.measurement is other.measurement:
            measurement = self.measurement
        elif total is None:
            measurement = CostMeasurement.UNAVAILABLE
        else:
            measurement = CostMeasurement.ESTIMATED
        return VideoCost(
            measurement=measurement,
            microusd=total,
            pricing_version=self.pricing_version or other.pricing_version,
        )


@runtime_checkable
class PriceCatalog(Protocol):
    """Port implemented by the consuming application."""

    @property
    def version(self) -> str:
        """Identifier that lets an amount be recomputed later."""
        ...

    def estimate(self, model: str, usage: TokenUsage) -> Cost: ...


@runtime_checkable
class AudioPriceCatalog(Protocol):
    """Port for duration-based transcription pricing."""

    @property
    def version(self) -> str: ...

    def estimate(self, model: str, usage: AudioUsage) -> AudioCost: ...


@runtime_checkable
class ImagePriceCatalog(Protocol):
    """Port for image generation pricing."""

    @property
    def version(self) -> str: ...

    def estimate(self, model: str, usage: ImageUsage) -> ImageCost: ...


@runtime_checkable
class VideoPriceCatalog(Protocol):
    """Port for video generation pricing."""

    @property
    def version(self) -> str: ...

    def estimate(self, model: str, usage: VideoUsage) -> VideoCost: ...


class StaticPriceCatalog:
    """A frozen table of rates, identified by a version string."""

    def __init__(self, *, version: str, rates: dict[str, ModelRate]) -> None:
        if not version.strip():
            raise ValueError("a pricing version is required so amounts can be audited later")
        self._version = version
        self._rates = dict(rates)

    @property
    def version(self) -> str:
        return self._version

    def estimate(self, model: str, usage: TokenUsage) -> Cost:
        rate = self._rates.get(model)
        if rate is None:
            return Cost.unavailable(pricing_version=self._version)

        billable_input = usage.billable_input_tokens
        billable_output = usage.billable_output_tokens
        if billable_input is None and billable_output is None:
            return Cost.unavailable(pricing_version=self._version)

        raw = (
            Decimal(billable_input or 0) * rate.input_microusd_per_token
            + Decimal(billable_output or 0) * rate.output_microusd_per_token
        )
        microusd = int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        measurement = CostMeasurement.ACTUAL if usage.complete else CostMeasurement.ESTIMATED
        return Cost(measurement=measurement, microusd=microusd, pricing_version=self._version)


class StaticAudioPriceCatalog:
    """A frozen duration-price table identified by a version string."""

    def __init__(self, *, version: str, rates: dict[str, AudioRate]) -> None:
        if not version.strip():
            raise ValueError("a pricing version is required so amounts are auditable")
        self._version = version
        self._rates = dict(rates)

    @property
    def version(self) -> str:
        return self._version

    def estimate(self, model: str, usage: AudioUsage) -> AudioCost:
        rate = self._rates.get(model)
        if rate is None or usage.duration_seconds is None:
            return AudioCost.unavailable(pricing_version=self._version)

        billed_seconds = max(usage.duration_seconds, rate.minimum_billable_seconds)
        raw = Decimal(str(billed_seconds)) / Decimal(60) * rate.usd_per_minute * _MICRO
        microusd = int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        measurement = CostMeasurement.ACTUAL if usage.complete else CostMeasurement.ESTIMATED
        return AudioCost(
            measurement=measurement,
            microusd=microusd,
            pricing_version=self._version,
        )


class StaticImagePriceCatalog:
    """A frozen image-price table identified by a version string."""

    def __init__(self, *, version: str, rates: dict[str, ImageRate]) -> None:
        if not version.strip():
            raise ValueError("a pricing version is required so amounts are auditable")
        self._version = version
        self._rates = dict(rates)

    @property
    def version(self) -> str:
        return self._version

    def estimate(self, model: str, usage: ImageUsage) -> ImageCost:
        rate = self._rates.get(model)
        if rate is None:
            return ImageCost.unavailable(pricing_version=self._version)

        raw = self._raw_microusd(rate, usage)
        if raw is None:
            return ImageCost.unavailable(pricing_version=self._version)

        measurement = CostMeasurement.ACTUAL if usage.complete else CostMeasurement.ESTIMATED
        return ImageCost(
            measurement=measurement,
            microusd=int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
            pricing_version=self._version,
        )

    def _raw_microusd(self, rate: ImageRate, usage: ImageUsage) -> Decimal | None:
        """The model's own billing unit decides, never the operation's."""
        if rate.token_rate is not None:
            tokens = usage.tokens
            if tokens is None:
                return None
            billable_input = tokens.billable_input_tokens
            billable_output = tokens.billable_output_tokens
            if billable_input is None and billable_output is None:
                return None
            return (
                Decimal(billable_input or 0) * rate.token_rate.input_microusd_per_token
                + Decimal(billable_output or 0) * rate.token_rate.output_microusd_per_token
            )
        if usage.images is None:
            return None
        assert rate.usd_per_image is not None  # guaranteed by ImageRate validation
        return Decimal(usage.images) * rate.usd_per_image * _MICRO


class StaticVideoPriceCatalog:
    """A frozen video-price table identified by a version string."""

    def __init__(self, *, version: str, rates: dict[str, VideoRate]) -> None:
        if not version.strip():
            raise ValueError("a pricing version is required so amounts are auditable")
        self._version = version
        self._rates = dict(rates)

    @property
    def version(self) -> str:
        return self._version

    def estimate(self, model: str, usage: VideoUsage) -> VideoCost:
        rate = self._rates.get(model)
        if rate is None or usage.seconds is None:
            return VideoCost.unavailable(pricing_version=self._version)

        per_second = rate.for_resolution(usage.resolution)
        if per_second is None:
            return VideoCost.unavailable(pricing_version=self._version)

        raw = Decimal(str(usage.seconds)) * per_second * _MICRO
        measurement = CostMeasurement.ACTUAL if usage.complete else CostMeasurement.ESTIMATED
        return VideoCost(
            measurement=measurement,
            microusd=int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
            pricing_version=self._version,
        )


class NullPriceCatalog:
    """Default catalogue: reports that cost is unknown, never that it is zero."""

    @property
    def version(self) -> str:
        return "none"

    def estimate(self, model: str, usage: TokenUsage) -> Cost:
        return Cost.unavailable()


class NullAudioPriceCatalog:
    """Default audio catalogue: unknown duration/cost, never free."""

    @property
    def version(self) -> str:
        return "none"

    def estimate(self, model: str, usage: AudioUsage) -> AudioCost:
        return AudioCost.unavailable()


class NullImagePriceCatalog:
    """Default image catalogue: unknown cost, never free."""

    @property
    def version(self) -> str:
        return "none"

    def estimate(self, model: str, usage: ImageUsage) -> ImageCost:
        return ImageCost.unavailable()


class NullVideoPriceCatalog:
    """Default video catalogue: unknown cost, never free."""

    @property
    def version(self) -> str:
        return "none"

    def estimate(self, model: str, usage: VideoUsage) -> VideoCost:
        return VideoCost.unavailable()

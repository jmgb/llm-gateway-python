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

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from llm_gateway.usage import AudioUsage, TokenUsage

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


@dataclass(frozen=True, slots=True)
class AudioRate:
    """Price per audio minute, with provider-specific minimum billing."""

    usd_per_minute: Decimal
    minimum_billable_seconds: int = 0

    def __post_init__(self) -> None:
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

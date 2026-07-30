"""Cost accounting.

Three rules the rest of the package depends on:

* an unknown cost is ``UNAVAILABLE``, never ``USD 0``;
* usage that is missing a billable dimension yields an ``ESTIMATED`` lower
  bound, never a silent ``ACTUAL``;
* arithmetic happens in whole microdollars so that totals reconcile exactly.

Catalogues are injected. The package ships no prices of its own, because the
authority on what a model costs is the consuming application, which already
has to reconcile it against a provider invoice.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from llm_gateway.usage import TokenUsage

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


@runtime_checkable
class PriceCatalog(Protocol):
    """Port implemented by the consuming application."""

    @property
    def version(self) -> str:
        """Identifier that lets an amount be recomputed later."""
        ...

    def estimate(self, model: str, usage: TokenUsage) -> Cost: ...


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


class NullPriceCatalog:
    """Default catalogue: reports that cost is unknown, never that it is zero."""

    @property
    def version(self) -> str:
        return "none"

    def estimate(self, model: str, usage: TokenUsage) -> Cost:
        return Cost.unavailable()

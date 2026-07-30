"""Retry, fallback and timeout as explicit, inspectable values.

Fallback defaults to disabled. A gateway that silently answers with a
different model than the one requested corrupts any A/B comparison and any
cost attribution, so switching models is always an opt-in decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_gateway.errors import LLMGatewayError


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times to re-attempt the *same* model, and when."""

    max_attempts: int = 1
    base_delay_seconds: float = 0.0
    retry_transient_only: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("a retry policy must allow at least one attempt")
        if self.base_delay_seconds < 0:
            raise ValueError("base delay cannot be negative")

    @classmethod
    def disabled(cls) -> RetryPolicy:
        """One attempt, no retries."""
        return cls(max_attempts=1)

    @classmethod
    def transient(cls, *, max_attempts: int = 2, base_delay_seconds: float = 0.5) -> RetryPolicy:
        """Retry only errors that may succeed later: rate limits, outages, timeouts."""
        return cls(max_attempts=max_attempts, base_delay_seconds=base_delay_seconds)

    def should_retry(self, error: LLMGatewayError, *, attempt_number: int) -> bool:
        if attempt_number >= self.max_attempts:
            return False
        if self.retry_transient_only:
            return error.transient
        return True

    def delay_before(self, *, attempt_number: int) -> float:
        """Exponential backoff, counted from the attempt that just failed."""
        return float(self.base_delay_seconds * (2 ** (attempt_number - 1)))


@dataclass(frozen=True, slots=True)
class FallbackPolicy:
    """Alternative models to try after the requested one is exhausted."""

    models: tuple[str, ...] = field(default=())

    @classmethod
    def disabled(cls) -> FallbackPolicy:
        return cls(models=())

    @classmethod
    def models_in_order(cls, *models: str) -> FallbackPolicy:
        return cls(models=tuple(models))

    @property
    def enabled(self) -> bool:
        return bool(self.models)


@dataclass(frozen=True, slots=True)
class TimeoutPolicy:
    """Seconds allowed per attempt and in total."""

    total_seconds: float = 120.0
    per_attempt_seconds_override: float | None = None

    def __post_init__(self) -> None:
        if self.total_seconds <= 0:
            raise ValueError("timeout must be positive")
        if self.per_attempt_seconds_override is not None and self.per_attempt_seconds_override <= 0:
            raise ValueError("per-attempt timeout must be positive")

    @property
    def per_attempt_seconds(self) -> float:
        return self.per_attempt_seconds_override or self.total_seconds

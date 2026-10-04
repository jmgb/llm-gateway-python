"""What the gateways share about a single attempt.

Four gateways, four attempt types, one set of rules: how an attempt is
stamped, which ones reach the bill, how a failure is classified and what a
fallback alert says about it. These are plain functions over structural types
rather than a generic gateway, because the orchestration around them differs
exactly where it matters — what counts as a usable reply, and when a video job
is billed — and a shared base class would hide those differences instead of
removing them.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol, Self, TypeVar

from llm_gateway.contracts import AttemptOutcome, FailurePhase
from llm_gateway.errors import (
    ConfigurationError,
    LLMGatewayError,
    OutputError,
    ProviderTimeoutError,
    SchemaValidationError,
)


class _Mergeable(Protocol):
    def merge(self, other: Self) -> Self: ...


_U = TypeVar("_U", bound=_Mergeable)
_C = TypeVar("_C", bound=_Mergeable)
_U_co = TypeVar("_U_co", bound=_Mergeable, covariant=True)
_C_co = TypeVar("_C_co", bound=_Mergeable, covariant=True)
_A = TypeVar("_A")


class RecordedAttempt(Protocol[_U_co, _C_co]):
    """The fields every media attempt type carries, whatever its unit."""

    @property
    def index(self) -> int: ...
    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...
    @property
    def outcome(self) -> AttemptOutcome: ...
    @property
    def usage(self) -> _U_co: ...
    @property
    def cost(self) -> _C_co: ...
    @property
    def error_type(self) -> str | None: ...
    @property
    def error_message(self) -> str | None: ...
    @property
    def billable(self) -> bool: ...
    @property
    def failure_phase(self) -> FailurePhase | None: ...


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def new_attempt(
    attempt_type: Callable[..., _A],
    *,
    index: int,
    model: str,
    provider: str,
    outcome: AttemptOutcome,
    usage: object,
    cost: object,
    started: float,
    error_type: str | None = None,
    error_message: str | None = None,
    billable: bool = True,
    failure_phase: FailurePhase | None = None,
) -> _A:
    """One attempt of ``attempt_type``, its latency measured from ``started``."""
    return attempt_type(
        index=index,
        model=model,
        provider=provider,
        outcome=outcome,
        usage=usage,
        cost=cost,
        latency_ms=elapsed_ms(started),
        error_type=error_type,
        error_message=error_message,
        billable=billable,
        failure_phase=failure_phase,
    )


def phase_of(failure: LLMGatewayError) -> FailurePhase:
    """Classify structurally, so a consumer never has to parse a message."""
    if isinstance(failure, ConfigurationError):
        return FailurePhase.CONFIGURATION
    if isinstance(failure, SchemaValidationError):
        return FailurePhase.SCHEMA_VALIDATION
    if isinstance(failure, OutputError):
        return FailurePhase.OUTPUT_PARSING
    if isinstance(failure, ProviderTimeoutError):
        return FailurePhase.TIMEOUT
    return FailurePhase.PROVIDER


def aggregate(
    attempts: Sequence[RecordedAttempt[_U, _C]],
    *,
    unknown_usage: _U,
    unavailable_cost: _C,
) -> tuple[_U, _C]:
    """Sum every billable attempt, so a retry is never invisible in the total."""
    billable = [attempt for attempt in attempts if attempt.billable]
    if not billable:
        return unknown_usage, unavailable_cost
    usage = billable[0].usage
    cost = billable[0].cost
    for attempt in billable[1:]:
        usage = usage.merge(attempt.usage)
        cost = cost.merge(attempt.cost)
    return usage, cost


def fallback_alert_fields(
    *,
    requested_model: str,
    model_used: str,
    request_id: str | None,
    attempts: Sequence[RecordedAttempt[_Mergeable, _Mergeable]],
) -> dict[str, object]:
    """A fallback alert that says why, not only that, the requested model was left.

    Two model names tell an operator something degraded and nothing else; by
    the time anyone reads the alert, the log that held the reason may have
    rotated away. The cause is the last failure on the requested model, and the
    full list follows because a plan with a retry and two models can fail three
    times for three different reasons.
    """
    failures = [a for a in attempts if a.outcome is AttemptOutcome.FAILED]
    cause = next((a for a in reversed(failures) if a.model == requested_model), None)
    return {
        "requested_model": requested_model,
        "model_used": model_used,
        "request_id": request_id,
        "error_type": cause.error_type if cause else None,
        "error_message": cause.error_message if cause else None,
        "failure_phase": (
            cause.failure_phase.value if cause and cause.failure_phase is not None else None
        ),
        "failures": [_failure_fields(attempt) for attempt in failures],
    }


def _failure_fields(attempt: RecordedAttempt[_Mergeable, _Mergeable]) -> dict[str, object]:
    return {
        "attempt": attempt.index,
        "model": attempt.model,
        "provider": attempt.provider,
        "error_type": attempt.error_type,
        "error_message": attempt.error_message,
        "failure_phase": attempt.failure_phase.value if attempt.failure_phase else None,
    }

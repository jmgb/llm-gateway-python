"""Optional extension points, all defaulting to no-op.

The package produces the data; the application decides where it goes. A sink
receives minimal, explicit metadata: never prompts, never responses, never
credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from llm_gateway.audio import AudioExecution
from llm_gateway.contracts import Execution
from llm_gateway.pricing import AudioCost, Cost
from llm_gateway.usage import AudioUsage, TokenUsage


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Everything needed to reconcile a call without storing its content."""

    request_id: str | None
    source: str | None
    provider: str
    requested_model: str
    model_used: str
    usage: TokenUsage
    cost: Cost
    attempts: int
    fallback_used: bool
    latency_ms: int
    finish_reason: str | None
    succeeded: bool


@dataclass(frozen=True, slots=True)
class AudioUsageRecord:
    """Duration accounting, intentionally not part of ``UsageRecord``."""

    request_id: str | None
    source: str | None
    provider: str
    requested_model: str
    model_used: str
    usage: AudioUsage
    cost: AudioCost
    attempts: int
    fallback_used: bool
    latency_ms: int
    succeeded: bool


@runtime_checkable
class UsageSink(Protocol):
    """Where token and cost accounting goes: a ledger, a metric, a log line."""

    def record(self, usage: UsageRecord) -> None: ...


@runtime_checkable
class AudioUsageSink(Protocol):
    """Where transcription duration and cost accounting goes."""

    def record(self, usage: AudioUsageRecord) -> None: ...


@runtime_checkable
class EventSink(Protocol):
    """Structured lifecycle events. Never receives prompt or response text."""

    def emit(self, event: str, fields: dict[str, object]) -> None: ...


@runtime_checkable
class AlertSink(Protocol):
    """Business alerting: a fallback fired, a budget was exceeded."""

    def alert(self, message: str, fields: dict[str, object]) -> None: ...


class NullUsageSink:
    def record(self, usage: UsageRecord) -> None:
        return None


class NullAudioUsageSink:
    def record(self, usage: AudioUsageRecord) -> None:
        return None


class NullEventSink:
    def emit(self, event: str, fields: dict[str, object]) -> None:
        return None


class NullAlertSink:
    def alert(self, message: str, fields: dict[str, object]) -> None:
        return None


def execution_to_record(
    execution: Execution,
    *,
    usage: TokenUsage,
    cost: Cost,
    request_id: str | None,
    source: str | None,
    succeeded: bool,
) -> UsageRecord:
    """Build the sink payload from a finished execution."""
    return UsageRecord(
        request_id=request_id,
        source=source,
        provider=execution.provider,
        requested_model=execution.requested_model,
        model_used=execution.model_used,
        usage=usage,
        cost=cost,
        attempts=execution.attempt_count,
        fallback_used=execution.fallback_used,
        latency_ms=execution.latency_ms,
        finish_reason=execution.finish_reason,
        succeeded=succeeded,
    )


def audio_execution_to_record(
    execution: AudioExecution,
    *,
    usage: AudioUsage,
    cost: AudioCost,
    request_id: str | None,
    source: str | None,
    succeeded: bool,
) -> AudioUsageRecord:
    return AudioUsageRecord(
        request_id=request_id,
        source=source,
        provider=execution.provider,
        requested_model=execution.requested_model,
        model_used=execution.model_used,
        usage=usage,
        cost=cost,
        attempts=execution.attempt_count,
        fallback_used=execution.fallback_used,
        latency_ms=execution.latency_ms,
        succeeded=succeeded,
    )

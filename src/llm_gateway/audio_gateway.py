"""Retries, fallback and duration accounting for transcription calls."""

from __future__ import annotations

import asyncio
import time

from llm_gateway.audio import (
    AudioAttempt,
    AudioExecution,
    ProviderTranscriptionResponse,
    TranscriptionRequest,
    TranscriptionResult,
)
from llm_gateway.contracts import AttemptOutcome, FailurePhase
from llm_gateway.errors import (
    AllTranscriptionsFailed,
    ConfigurationError,
    LLMGatewayError,
    ProviderError,
    ProviderTimeoutError,
)
from llm_gateway.models import builtin_audio_price_catalog, lookup_model
from llm_gateway.ports import (
    AlertSink,
    AudioUsageSink,
    EventSink,
    NullAlertSink,
    NullAudioUsageSink,
    NullEventSink,
    audio_execution_to_record,
)
from llm_gateway.pricing import AudioCost, AudioPriceCatalog
from llm_gateway.providers.base import AudioProviderAdapter
from llm_gateway.registry import ProviderRegistry
from llm_gateway.usage import AudioUsage


class AudioGateway:
    """Provider-agnostic entry point for speech-to-text operations."""

    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        price_catalog: AudioPriceCatalog | None = None,
        usage_sink: AudioUsageSink | None = None,
        event_sink: EventSink | None = None,
        alert_sink: AlertSink | None = None,
    ) -> None:
        self._registry = registry
        self._prices = price_catalog or builtin_audio_price_catalog()
        self._usage_sink = usage_sink or NullAudioUsageSink()
        self._events = event_sink or NullEventSink()
        self._alerts = alert_sink or NullAlertSink()

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        attempts: list[AudioAttempt] = []
        try:
            async with asyncio.timeout(request.timeout_policy.total_seconds):
                return await self._run(request, attempts)
        except TimeoutError:
            self._report_failure(request, attempts)
            raise AllTranscriptionsFailed(
                f"the transcription exceeded its total budget of "
                f"{request.timeout_policy.total_seconds}s after {len(attempts)} attempt(s)",
                attempts=tuple(attempts),
            ) from None

    async def _run(
        self, request: TranscriptionRequest, attempts: list[AudioAttempt]
    ) -> TranscriptionResult:
        started = time.perf_counter()
        plan = [request.model, *request.fallback_policy.models]
        for model in plan:
            self._registry.resolve(model)

        last_failure: LLMGatewayError | None = None
        for model in plan:
            adapter = self._registry.resolve(model)
            outcome = await self._attempt_model(request, model=model, attempts=attempts)
            if isinstance(outcome, ProviderTranscriptionResponse):
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                execution = AudioExecution(
                    requested_model=request.model,
                    model_used=outcome.model_used or model,
                    provider=adapter.name,
                    attempts=tuple(attempts),
                    latency_ms=elapsed_ms,
                )
                usage, cost = _aggregate(attempts)
                if execution.fallback_used:
                    self._alerts.alert(
                        "llm_audio_fallback_used",
                        {
                            "requested_model": request.model,
                            "model_used": execution.model_used,
                            "request_id": request.request_id,
                        },
                    )
                self._record(request, execution, usage=usage, cost=cost, succeeded=True)
                self._events.emit(
                    "llm_audio_transcription_succeeded",
                    _event_fields(request, execution, usage=usage, cost=cost),
                )
                return TranscriptionResult(
                    text=outcome.text,
                    usage=usage,
                    execution=execution,
                    cost=cost,
                    segments=outcome.segments,
                    language=outcome.language,
                )
            last_failure = outcome

        self._report_failure(request, attempts, started=started)
        raise AllTranscriptionsFailed(
            f"all {len(attempts)} transcription attempt(s) failed for model {request.model!r}",
            attempts=tuple(attempts),
        ) from last_failure

    async def _attempt_model(
        self,
        request: TranscriptionRequest,
        *,
        model: str,
        attempts: list[AudioAttempt],
    ) -> ProviderTranscriptionResponse | LLMGatewayError:
        adapter = self._registry.resolve(model)
        policy = request.retry_policy
        last_failure: LLMGatewayError | None = None

        for attempt_number in range(1, policy.max_attempts + 1):
            attempt_started = time.perf_counter()
            try:
                _require_audio_model(model)
                if not isinstance(adapter, AudioProviderAdapter):
                    raise ConfigurationError(
                        f"provider {adapter.name} does not support transcription"
                    )
                async with asyncio.timeout(request.timeout_policy.per_attempt_seconds):
                    response = await adapter.transcribe(request, model=model)
            except TimeoutError as error:
                failure: LLMGatewayError = ProviderTimeoutError(
                    f"transcription attempt exceeded {request.timeout_policy.per_attempt_seconds}s"
                )
                failure.__cause__ = error
            except LLMGatewayError as error:
                failure = error
            else:
                usage = response.usage
                cost = self._prices.estimate(model, usage)
                attempts.append(
                    _record_attempt(
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome="succeeded",
                        usage=usage,
                        cost=cost,
                        started=attempt_started,
                    )
                )
                return response

            last_failure = failure
            attempts.append(
                _record_attempt(
                    index=len(attempts) + 1,
                    model=model,
                    provider=adapter.name,
                    outcome="failed",
                    usage=AudioUsage.unknown(),
                    cost=AudioCost.unavailable(pricing_version=self._prices.version),
                    started=attempt_started,
                    error_type=type(failure).__name__,
                    billable=isinstance(failure, ProviderError),
                    failure_phase=_phase_of(failure),
                )
            )
            if not policy.should_retry(failure, attempt_number=attempt_number):
                return failure
            delay = policy.delay_before(attempt_number=attempt_number)
            if delay:
                await asyncio.sleep(delay)

        assert last_failure is not None
        return last_failure

    def _report_failure(
        self,
        request: TranscriptionRequest,
        attempts: list[AudioAttempt],
        *,
        started: float | None = None,
    ) -> None:
        usage, cost = _aggregate(attempts)
        elapsed_ms = int((time.perf_counter() - started) * 1000) if started is not None else 0
        execution = AudioExecution(
            requested_model=request.model,
            model_used=attempts[-1].model if attempts else request.model,
            provider=attempts[-1].provider if attempts else "unknown",
            attempts=tuple(attempts),
            latency_ms=elapsed_ms,
        )
        self._record(request, execution, usage=usage, cost=cost, succeeded=False)
        self._events.emit(
            "llm_audio_transcription_failed",
            _event_fields(request, execution, usage=usage, cost=cost),
        )

    def _record(
        self,
        request: TranscriptionRequest,
        execution: AudioExecution,
        *,
        usage: AudioUsage,
        cost: AudioCost,
        succeeded: bool,
    ) -> None:
        self._usage_sink.record(
            audio_execution_to_record(
                execution,
                usage=usage,
                cost=cost,
                request_id=request.request_id,
                source=request.source,
                succeeded=succeeded,
            )
        )


def _require_audio_model(model: str) -> None:
    info = lookup_model(model)
    if info is not None and info.pricing_unit != "audio_minutes":
        raise ConfigurationError(f"{model!r} is token-priced; use LLMGateway.generate()")


def _record_attempt(
    *,
    index: int,
    model: str,
    provider: str,
    outcome: str,
    usage: AudioUsage,
    cost: AudioCost,
    started: float,
    error_type: str | None = None,
    billable: bool = True,
    failure_phase: FailurePhase | None = None,
) -> AudioAttempt:
    return AudioAttempt(
        index=index,
        model=model,
        provider=provider,
        outcome=AttemptOutcome.SUCCEEDED if outcome == "succeeded" else AttemptOutcome.FAILED,
        usage=usage,
        cost=cost,
        latency_ms=int((time.perf_counter() - started) * 1000),
        error_type=error_type,
        billable=billable,
        failure_phase=failure_phase,
    )


def _phase_of(failure: LLMGatewayError) -> FailurePhase:
    if isinstance(failure, ConfigurationError):
        return FailurePhase.CONFIGURATION
    if isinstance(failure, ProviderTimeoutError):
        return FailurePhase.TIMEOUT
    return FailurePhase.PROVIDER


def _aggregate(attempts: list[AudioAttempt]) -> tuple[AudioUsage, AudioCost]:
    billable = [attempt for attempt in attempts if attempt.billable]
    if not billable:
        return AudioUsage.unknown(), AudioCost.unavailable()
    usage = billable[0].usage
    cost = billable[0].cost
    for attempt in billable[1:]:
        usage = usage.merge(attempt.usage)
        cost = cost.merge(attempt.cost)
    return usage, cost


def _event_fields(
    request: TranscriptionRequest,
    execution: AudioExecution,
    *,
    usage: AudioUsage,
    cost: AudioCost,
) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "source": request.source,
        "provider": execution.provider,
        "requested_model": execution.requested_model,
        "model_used": execution.model_used,
        "attempts": execution.attempt_count,
        "fallback_used": execution.fallback_used,
        "latency_ms": execution.latency_ms,
        "audio_duration_seconds": usage.duration_seconds,
        "cost_microusd": cost.microusd,
        "cost_measurement": cost.measurement.value,
        "pricing_version": cost.pricing_version,
    }

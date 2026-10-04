"""Retries, fallback and duration accounting for transcription calls."""

from __future__ import annotations

import asyncio
import time

from llm_gateway._attempts import (
    aggregate,
    elapsed_ms,
    fallback_alert_fields,
    new_attempt,
    phase_of,
)
from llm_gateway.audio import (
    AudioAttempt,
    AudioExecution,
    ProviderTranscriptionResponse,
    TranscriptionRequest,
    TranscriptionResult,
)
from llm_gateway.catalogs import builtin_audio_price_catalog
from llm_gateway.contracts import AttemptOutcome, FailurePhase
from llm_gateway.errors import (
    AllTranscriptionsFailed,
    ConfigurationError,
    LLMGatewayError,
    ProviderError,
    ProviderTimeoutError,
    message_of,
)
from llm_gateway.models import lookup_model
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
from llm_gateway.providers.base import AudioProviderAdapter, ProviderAdapter
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
        started = time.perf_counter()
        budget = asyncio.timeout(request.timeout_policy.total_seconds)
        try:
            async with budget:
                return await self._run(request, attempts, started=started)
        except TimeoutError:
            # Only the budget expiring is the call running out of time; an
            # application sink timing out after a success is not.
            if not budget.expired():
                raise
            self._report_failure(request, attempts, started=started)
            raise AllTranscriptionsFailed(
                f"the transcription exceeded its total budget of "
                f"{request.timeout_policy.total_seconds}s after {len(attempts)} attempt(s)",
                attempts=tuple(attempts),
            ) from None

    async def _run(
        self, request: TranscriptionRequest, attempts: list[AudioAttempt], *, started: float
    ) -> TranscriptionResult:
        # Every model is checked before the first is called: a token model
        # discovered only when its fallback turn came would fail after the
        # requested one had already been paid for.
        plan = [request.model, *request.fallback_policy.models]
        adapters = {model: self._registry.resolve(model) for model in plan}
        for model in plan:
            _require_audio_model(model)

        last_failure: LLMGatewayError | None = None
        for model in plan:
            adapter = adapters[model]
            outcome = await self._attempt_model(
                request, model=model, adapter=adapter, attempts=attempts
            )
            if isinstance(outcome, ProviderTranscriptionResponse):
                execution = AudioExecution(
                    requested_model=request.model,
                    model_used=outcome.model_used or model,
                    provider=adapter.name,
                    attempts=tuple(attempts),
                    latency_ms=elapsed_ms(started),
                )
                usage, cost = _aggregate(attempts)
                # Booked first: an alert hook that fails must not leave a paid
                # call unrecorded.
                self._record(request, execution, usage=usage, cost=cost, succeeded=True)
                if execution.fallback_used:
                    self._alerts.alert(
                        "llm_audio_fallback_used",
                        fallback_alert_fields(
                            requested_model=request.model,
                            model_used=execution.model_used,
                            request_id=request.request_id,
                            attempts=attempts,
                        ),
                    )
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
        adapter: ProviderAdapter,
        attempts: list[AudioAttempt],
    ) -> ProviderTranscriptionResponse | LLMGatewayError:
        policy = request.retry_policy

        for attempt_number in range(1, policy.max_attempts + 1):
            attempt_started = time.perf_counter()
            try:
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
            except asyncio.CancelledError:
                attempts.append(
                    new_attempt(
                        AudioAttempt,
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome=AttemptOutcome.FAILED,
                        usage=AudioUsage.unknown(),
                        cost=AudioCost.unavailable(pricing_version=self._prices.version),
                        started=attempt_started,
                        error_type=ProviderTimeoutError.__name__,
                        error_message="transcription attempt cancelled by the total timeout budget",
                        billable=True,
                        failure_phase=FailurePhase.TIMEOUT,
                    )
                )
                raise
            except LLMGatewayError as error:
                failure = error
            else:
                usage = response.usage
                cost = self._prices.estimate(model, usage)
                attempts.append(
                    new_attempt(
                        AudioAttempt,
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome=AttemptOutcome.SUCCEEDED,
                        usage=usage,
                        cost=cost,
                        started=attempt_started,
                    )
                )
                return response

            attempts.append(
                new_attempt(
                    AudioAttempt,
                    index=len(attempts) + 1,
                    model=model,
                    provider=adapter.name,
                    outcome=AttemptOutcome.FAILED,
                    usage=AudioUsage.unknown(),
                    cost=AudioCost.unavailable(pricing_version=self._prices.version),
                    started=attempt_started,
                    error_type=type(failure).__name__,
                    error_message=message_of(failure),
                    billable=isinstance(failure, ProviderError),
                    failure_phase=phase_of(failure),
                )
            )
            if not policy.should_retry(failure, attempt_number=attempt_number):
                return failure
            delay = policy.delay_before(attempt_number=attempt_number)
            if delay:
                await asyncio.sleep(delay)

        return failure

    def _report_failure(
        self,
        request: TranscriptionRequest,
        attempts: list[AudioAttempt],
        *,
        started: float,
    ) -> None:
        usage, cost = _aggregate(attempts)
        execution = AudioExecution(
            requested_model=request.model,
            model_used=attempts[-1].model if attempts else request.model,
            provider=attempts[-1].provider if attempts else "unknown",
            attempts=tuple(attempts),
            latency_ms=elapsed_ms(started),
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


def _aggregate(attempts: list[AudioAttempt]) -> tuple[AudioUsage, AudioCost]:
    return aggregate(
        attempts, unknown_usage=AudioUsage.unknown(), unavailable_cost=AudioCost.unavailable()
    )


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

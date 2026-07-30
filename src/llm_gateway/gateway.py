"""The orchestrator.

One place decides how many times to try, when to switch model, what each
attempt cost and what the caller finally sees. Adapters stay thin because this
does not live in them.

Two invariants worth stating out loud:

* every attempt that reached the provider is recorded and billed, including
  the ones that failed, because a retry that timed out may still be invoiced;
* an exhausted call raises. It never returns a result that looks successful.
"""

from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from llm_gateway.contracts import (
    Attempt,
    AttemptOutcome,
    Execution,
    LLMRequest,
    LLMResult,
    ResponseFormat,
)
from llm_gateway.errors import (
    AllAttemptsFailed,
    LLMGatewayError,
    ProviderError,
    ProviderTimeoutError,
    SchemaValidationError,
)
from llm_gateway.json_payload import parse_json_payload
from llm_gateway.models import builtin_price_catalog
from llm_gateway.ports import (
    AlertSink,
    EventSink,
    NullAlertSink,
    NullEventSink,
    NullUsageSink,
    UsageSink,
    execution_to_record,
)
from llm_gateway.pricing import Cost, CostMeasurement, PriceCatalog
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.registry import ProviderRegistry
from llm_gateway.usage import TokenUsage


class LLMGateway:
    """Provider-agnostic entry point. Holds no credentials of its own."""

    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        price_catalog: PriceCatalog | None = None,
        usage_sink: UsageSink | None = None,
        event_sink: EventSink | None = None,
        alert_sink: AlertSink | None = None,
    ) -> None:
        self._registry = registry
        # Default to the shared catalogue: a consumer that says nothing about
        # prices gets real, versioned ones rather than silent UNAVAILABLE.
        # Passing an explicit catalogue still wins, for negotiated rates.
        self._prices = price_catalog or builtin_price_catalog()
        self._usage_sink = usage_sink or NullUsageSink()
        self._events = event_sink or NullEventSink()
        self._alerts = alert_sink or NullAlertSink()

    async def generate(self, request: LLMRequest) -> LLMResult:
        """Run the request to completion, or raise a typed error."""
        started = time.perf_counter()
        attempts: list[Attempt] = []

        # Resolve every model up front so an unroutable fallback fails before
        # any money is spent, not halfway through a degraded call.
        plan = [request.model, *request.fallback_policy.models]
        for model in plan:
            self._registry.resolve(model)

        for model in plan:
            adapter = self._registry.resolve(model)
            outcome = await self._attempt_model(request, model=model, attempts=attempts)
            if outcome is None:
                continue

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            execution = Execution(
                requested_model=request.model,
                model_used=model,
                provider=adapter.name,
                finish_reason=outcome.finish_reason,
                attempts=tuple(attempts),
                latency_ms=elapsed_ms,
            )
            usage, cost = _aggregate(attempts)
            output = _interpret(outcome, request)

            if execution.fallback_used:
                self._alerts.alert(
                    "llm_fallback_used",
                    {
                        "requested_model": request.model,
                        "model_used": model,
                        "request_id": request.request_id,
                    },
                )
            self._usage_sink.record(
                execution_to_record(
                    execution,
                    usage=usage,
                    cost=cost,
                    request_id=request.request_id,
                    source=request.source,
                    succeeded=True,
                )
            )
            self._events.emit("llm_call_succeeded", _event_fields(request, execution, cost))
            return LLMResult(output=output, usage=usage, execution=execution, cost=cost)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        usage, cost = _aggregate(attempts)
        failed = Execution(
            requested_model=request.model,
            model_used=attempts[-1].model if attempts else request.model,
            provider=attempts[-1].provider if attempts else "unknown",
            finish_reason=None,
            attempts=tuple(attempts),
            latency_ms=elapsed_ms,
        )
        self._usage_sink.record(
            execution_to_record(
                failed,
                usage=usage,
                cost=cost,
                request_id=request.request_id,
                source=request.source,
                succeeded=False,
            )
        )
        self._events.emit("llm_call_failed", _event_fields(request, failed, cost))
        raise AllAttemptsFailed(
            f"all {len(attempts)} attempt(s) failed for model {request.model!r}",
            attempts=tuple(attempts),
        )

    async def _attempt_model(
        self,
        request: LLMRequest,
        *,
        model: str,
        attempts: list[Attempt],
    ) -> ProviderResponse | None:
        """Try one model until its retry policy is exhausted."""
        adapter = self._registry.resolve(model)
        policy = request.retry_policy

        for attempt_number in range(1, policy.max_attempts + 1):
            attempt_started = time.perf_counter()
            try:
                async with asyncio.timeout(request.timeout_policy.per_attempt_seconds):
                    response = await adapter.generate(request, model=model)
            except TimeoutError as error:
                failure: LLMGatewayError = ProviderTimeoutError(
                    f"attempt exceeded {request.timeout_policy.per_attempt_seconds}s"
                )
                failure.__cause__ = error
            except LLMGatewayError as error:
                failure = error
            else:
                attempts.append(
                    _record_attempt(
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome=AttemptOutcome.SUCCEEDED,
                        usage=response.usage,
                        cost=self._prices.estimate(model, response.usage),
                        started=attempt_started,
                    )
                )
                return response

            attempts.append(
                _record_attempt(
                    index=len(attempts) + 1,
                    model=model,
                    provider=adapter.name,
                    outcome=AttemptOutcome.FAILED,
                    usage=TokenUsage.unknown(),
                    # A failed call may still be billed; the amount is unknown,
                    # which is not the same as knowing it was free.
                    cost=Cost.unavailable(),
                    started=attempt_started,
                    error_type=type(failure).__name__,
                    billable=isinstance(failure, ProviderError),
                )
            )
            if not policy.should_retry(failure, attempt_number=attempt_number):
                return None
            delay = policy.delay_before(attempt_number=attempt_number)
            if delay:
                await asyncio.sleep(delay)
        return None


def _record_attempt(
    *,
    index: int,
    model: str,
    provider: str,
    outcome: AttemptOutcome,
    usage: TokenUsage,
    cost: Cost,
    started: float,
    error_type: str | None = None,
    billable: bool = True,
) -> Attempt:
    return Attempt(
        index=index,
        model=model,
        provider=provider,
        outcome=outcome,
        usage=usage,
        cost=cost,
        latency_ms=int((time.perf_counter() - started) * 1000),
        error_type=error_type,
        billable=billable,
    )


def _aggregate(attempts: list[Attempt]) -> tuple[TokenUsage, Cost]:
    """Sum every billable attempt, so a retry is never invisible in the total."""
    billable = [a for a in attempts if a.billable]
    if not billable:
        return TokenUsage.unknown(), Cost.unavailable()

    usage = billable[0].usage
    cost = billable[0].cost
    for attempt in billable[1:]:
        usage = usage.merge(attempt.usage)
        cost = cost.merge(attempt.cost)
    return usage, cost


def _interpret(response: ProviderResponse, request: LLMRequest) -> object:
    """Turn provider text into what the caller asked for."""
    if request.response_format is ResponseFormat.TEXT:
        return response.output_text or ""

    payload = parse_json_payload(response.output_text)
    if request.response_format is ResponseFormat.JSON_OBJECT:
        return payload

    schema = request.response_schema
    assert schema is not None  # guaranteed by LLMRequest validation
    try:
        return schema.model_validate(payload)
    except ValidationError as error:
        raise SchemaValidationError(
            f"the response did not satisfy {schema.__name__}: {error.error_count()} violation(s)"
        ) from error


def _event_fields(request: LLMRequest, execution: Execution, cost: Cost) -> dict[str, object]:
    """Observability payload. Deliberately excludes prompts and responses."""
    return {
        "request_id": request.request_id,
        "source": request.source,
        "provider": execution.provider,
        "requested_model": execution.requested_model,
        "model_used": execution.model_used,
        "attempts": execution.attempt_count,
        "fallback_used": execution.fallback_used,
        "latency_ms": execution.latency_ms,
        "finish_reason": execution.finish_reason,
        "cost_microusd": cost.microusd,
        "cost_measurement": cost.measurement.value
        if isinstance(cost.measurement, CostMeasurement)
        else None,
        "pricing_version": cost.pricing_version,
    }

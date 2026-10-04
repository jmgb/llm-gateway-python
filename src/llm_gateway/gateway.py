"""The orchestrator.

One place decides how many times to try, when to switch model, what each
attempt cost and what the caller finally sees. Adapters stay thin because this
does not live in them.

Three invariants worth stating out loud:

* every attempt that reached the provider is recorded and billed, including
  the ones that failed, because a retry that timed out may still be invoiced;
* an answer that cannot be parsed or does not satisfy the schema is one of
  those failures, decided *inside* the attempt loop rather than after it, so
  the fallback still has a turn and the money is still counted;
* an exhausted call raises. It never returns a result that looks successful.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any, get_args

from pydantic import BaseModel, ValidationError

from llm_gateway._attempts import aggregate, fallback_alert_fields, new_attempt, phase_of
from llm_gateway.audio import TranscriptionRequest, TranscriptionResult
from llm_gateway.audio_gateway import AudioGateway
from llm_gateway.catalogs import builtin_price_catalog
from llm_gateway.contracts import (
    Attempt,
    AttemptOutcome,
    Execution,
    FailurePhase,
    LLMRequest,
    LLMResult,
    ReasoningEffort,
    ResponseFormat,
)
from llm_gateway.errors import (
    AllAttemptsFailed,
    ConfigurationError,
    LLMGatewayError,
    OutputError,
    OutputParsingError,
    ProviderError,
    ProviderTimeoutError,
    SchemaValidationError,
    message_of,
)
from llm_gateway.json_payload import parse_json_payload
from llm_gateway.media import (
    ImageRequest,
    ImageResult,
    VideoJob,
    VideoJobResult,
    VideoRequest,
    VideoResult,
)
from llm_gateway.media_gateway import ImageGateway, VideoGateway
from llm_gateway.models import ModelInfo, lookup_model
from llm_gateway.ports import (
    AlertSink,
    AudioUsageSink,
    EventSink,
    ImageUsageSink,
    NullAlertSink,
    NullEventSink,
    NullUsageSink,
    UsageSink,
    VideoUsageSink,
    execution_to_record,
)
from llm_gateway.pricing import (
    AudioPriceCatalog,
    Cost,
    ImagePriceCatalog,
    PriceCatalog,
    VideoPriceCatalog,
)
from llm_gateway.providers.base import ProviderAdapter, ProviderResponse
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.registry import ProviderRegistry
from llm_gateway.tools import ToolCall
from llm_gateway.usage import TokenUsage


@dataclass(frozen=True, slots=True)
class _Completion:
    """An attempt that produced output the caller can actually be given."""

    response: ProviderResponse
    output: Any
    tool_calls: tuple[ToolCall, ...] = ()


class LLMGateway:
    """Provider-agnostic entry point. Holds no credentials of its own."""

    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        price_catalog: PriceCatalog | None = None,
        audio_price_catalog: AudioPriceCatalog | None = None,
        image_price_catalog: ImagePriceCatalog | None = None,
        video_price_catalog: VideoPriceCatalog | None = None,
        usage_sink: UsageSink | None = None,
        audio_usage_sink: AudioUsageSink | None = None,
        image_usage_sink: ImageUsageSink | None = None,
        video_usage_sink: VideoUsageSink | None = None,
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
        self._audio = AudioGateway(
            registry=registry,
            price_catalog=audio_price_catalog,
            usage_sink=audio_usage_sink,
            event_sink=self._events,
            alert_sink=self._alerts,
        )
        self._images = ImageGateway(
            registry=registry,
            price_catalog=image_price_catalog,
            usage_sink=image_usage_sink,
            event_sink=self._events,
            alert_sink=self._alerts,
        )
        self._videos = VideoGateway(
            registry=registry,
            price_catalog=video_price_catalog,
            usage_sink=video_usage_sink,
            event_sink=self._events,
            alert_sink=self._alerts,
        )

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe audio with duration accounting separate from tokens."""
        return await self._audio.transcribe(request)

    async def generate_image(self, request: ImageRequest) -> ImageResult:
        """Generate or edit an image, with image accounting separate from tokens."""
        return await self._images.generate_image(request)

    async def generate_video(self, request: VideoRequest) -> VideoResult:
        """Generate video, with per-second accounting separate from tokens.

        For providers whose adapter can own the polling loop. One awaited call
        that returns the clip; ``submit_video()`` serves the rest.
        """
        return await self._videos.generate_video(request)

    async def submit_video(self, request: VideoRequest) -> VideoJob:
        """Start a video job and return it, minutes before the clip exists.

        The returned job is plain storable data. Poll it with ``poll_video()``
        from wherever is convenient — a worker, or the handler for the webhook
        registered through ``VideoRequest.webhook_url``.
        """
        return await self._videos.submit_video(request)

    async def poll_video(self, job: VideoJob, *, timeout_seconds: float = 30.0) -> VideoJobResult:
        """Read a submitted job's state, and its clip once there is one.

        ``timeout_seconds`` bounds the status call, not the job — which is
        expected to run for minutes and is why this is a poll at all.
        """
        return await self._videos.poll_video(job, timeout_seconds=timeout_seconds)

    async def generate(self, request: LLMRequest) -> LLMResult:
        """Run the request to completion, or raise a typed error.

        ``timeout_policy.total_seconds`` bounds the **whole** call: every
        attempt, every retry and every backoff pause together. A budget that
        only applied per attempt would let two retries spend twice what the
        caller authorised.
        """
        attempts: list[Attempt] = []
        started = time.perf_counter()
        budget = asyncio.timeout(request.timeout_policy.total_seconds)
        try:
            async with budget:
                return await self._run(request, attempts, started=started, deadline=budget.when())
        except TimeoutError:
            # Only the budget expiring is the call running out of time. A sink
            # of the application's own can time out after a success has been
            # recorded, and reporting that as a failure would book it twice.
            if not budget.expired():
                raise
            self._report_failure(request, attempts, started=started)
            raise AllAttemptsFailed(
                f"the call exceeded its total budget of "
                f"{request.timeout_policy.total_seconds}s after {len(attempts)} attempt(s)",
                attempts=tuple(attempts),
            ) from None

    async def _run(
        self,
        request: LLMRequest,
        attempts: list[Attempt],
        *,
        started: float,
        deadline: float | None,
    ) -> LLMResult:
        # Resolve every model up front so an unroutable fallback fails before
        # any money is spent, not halfway through a degraded call.
        plan = [request.model, *request.fallback_policy.models]
        adapters: dict[str, ProviderAdapter] = {}
        for model in plan:
            adapters[model] = self._registry.resolve(model)
            info = lookup_model(model)
            if info is not None and info.pricing_unit == "audio_minutes":
                raise ConfigurationError(
                    f"{model!r} is audio-priced; use LLMGateway.transcribe() instead"
                )
            # An image model answers with pictures the text path would drop on
            # the floor: Gemini's image reply carries inline parts and an empty
            # `.text`, so without this the caller gets a silent empty success.
            if info is not None and info.modality == "image":
                raise ConfigurationError(
                    f"{model!r} generates images; use LLMGateway.generate_image() instead"
                )
            if info is not None and info.modality == "video":
                raise ConfigurationError(
                    f"{model!r} generates video; use LLMGateway.generate_video() instead"
                )
            if info is not None and info.modality == "realtime":
                raise ConfigurationError(
                    f"{model!r} is served only on OpenAI's Realtime API, which this "
                    "package does not speak"
                )
        requests_by_model = {model: _request_for_model(request, model) for model in plan}

        last_failure: LLMGatewayError | None = None
        for model in plan:
            adapter = adapters[model]
            outcome = await self._attempt_model(
                requests_by_model[model],
                model=model,
                adapter=adapter,
                attempts=attempts,
                deadline=deadline,
            )
            if not isinstance(outcome, _Completion):
                last_failure = outcome
                continue

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            execution = Execution(
                requested_model=request.model,
                model_used=outcome.response.model_used or model,
                provider=adapter.name,
                finish_reason=outcome.response.finish_reason,
                attempts=tuple(attempts),
                latency_ms=elapsed_ms,
            )
            usage, cost = _aggregate(attempts)
            # Booked before anything else runs: an alert hook that fails must
            # not leave a paid call unrecorded.
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
            if execution.fallback_used:
                self._alerts.alert(
                    "llm_fallback_used",
                    fallback_alert_fields(
                        requested_model=request.model,
                        model_used=execution.model_used,
                        request_id=request.request_id,
                        attempts=execution.attempts,
                    ),
                )
            self._events.emit("llm_call_succeeded", _event_fields(request, execution, cost))
            return LLMResult(
                output=outcome.output,
                usage=usage,
                execution=execution,
                cost=cost,
                tool_calls=outcome.tool_calls,
            )

        self._report_failure(request, attempts, started=started)
        raise AllAttemptsFailed(
            f"all {len(attempts)} attempt(s) failed for model {request.model!r}",
            attempts=tuple(attempts),
            # Chained, not swallowed: an exhausted call must still be able to
            # say whether it ran out of provider or out of usable answers.
        ) from last_failure

    def _report_failure(
        self,
        request: LLMRequest,
        attempts: list[Attempt],
        *,
        started: float,
    ) -> None:
        """Emit accounting for a call that never produced a result.

        A failure still spent money, so it is reported exactly like a success.
        """
        usage, cost = _aggregate(attempts)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
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

    async def _attempt_model(
        self,
        request: LLMRequest,
        *,
        model: str,
        adapter: ProviderAdapter,
        attempts: list[Attempt],
        deadline: float | None,
    ) -> _Completion | LLMGatewayError:
        """Try one model until its retry policy is exhausted.

        Returns the completion, or the failure that ended this model's turn —
        which the caller uses to decide whether a fallback still applies.
        """
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
            except asyncio.CancelledError:
                # The call's total budget cut this attempt off mid-flight. The
                # provider already has the request and may invoice it, so it
                # is recorded before the cancellation carries on unwinding.
                attempts.append(
                    new_attempt(
                        Attempt,
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome=AttemptOutcome.FAILED,
                        usage=TokenUsage.unknown(),
                        cost=Cost.unavailable(),
                        started=attempt_started,
                        error_type=ProviderTimeoutError.__name__,
                        error_message="the call's total budget ran out during this attempt",
                        failure_phase=FailurePhase.TIMEOUT,
                    )
                )
                raise
            except Exception as error:
                # An adapter is meant to map what it raises, and one that did
                # not has still sent the request. Classified structurally, it
                # is a failed attempt like any other: recorded, and the
                # fallback still gets its turn instead of a bare crash.
                failure = classify_provider_error(error)
            else:
                usage = response.usage
                cost = self._prices.estimate(model, usage)
                try:
                    output, tool_calls = _interpret(response, request)
                except OutputError as unusable:
                    # The provider answered and will invoice for it, so the
                    # tokens it reported are recorded exactly as on a success.
                    # What is not recorded is a result: this attempt failed.
                    attempts.append(
                        new_attempt(
                            Attempt,
                            index=len(attempts) + 1,
                            model=model,
                            provider=adapter.name,
                            outcome=AttemptOutcome.FAILED,
                            usage=usage,
                            cost=cost,
                            started=attempt_started,
                            error_type=type(unusable).__name__,
                            error_message=message_of(unusable),
                            failure_phase=phase_of(unusable),
                        )
                    )
                    # Deliberately not retried on the same model: the same
                    # prompt and the same model reproduce the same malformed
                    # answer, so a retry buys a second invoice for one failure.
                    # The next model in the plan gets the turn instead.
                    return unusable

                attempts.append(
                    new_attempt(
                        Attempt,
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome=AttemptOutcome.SUCCEEDED,
                        usage=usage,
                        cost=cost,
                        started=attempt_started,
                    )
                )
                return _Completion(response=response, output=output, tool_calls=tool_calls)

            attempts.append(
                new_attempt(
                    Attempt,
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
                    error_message=message_of(failure),
                    billable=isinstance(failure, ProviderError),
                    failure_phase=phase_of(failure),
                )
            )
            if not policy.should_retry(failure, attempt_number=attempt_number):
                return failure
            delay = policy.delay_before(attempt_number=attempt_number)
            if delay:
                # A pause the budget cannot cover would end the call as a
                # timeout and bury the error that actually stopped it.
                if deadline is not None and asyncio.get_running_loop().time() + delay >= deadline:
                    return failure
                await asyncio.sleep(delay)
        return failure


def _aggregate(attempts: list[Attempt]) -> tuple[TokenUsage, Cost]:
    return aggregate(
        attempts, unknown_usage=TokenUsage.unknown(), unavailable_cost=Cost.unavailable()
    )


def _request_for_model(request: LLMRequest, model: str) -> LLMRequest:
    """Strip options the target model does not accept, before it is attempted.

    A fallback inherits the request that was written for a *different* model,
    and a provider rejects the whole call over one option it does not know.
    Nothing raises here: the fallback stays visible in the execution, and only
    the offending option is dropped.
    """
    info = lookup_model(model)
    changes: dict[str, Any] = {}

    effort = _effort_for_model(request, info)
    if effort != request.reasoning_effort:
        changes["reasoning_effort"] = effort

    # Silence in the catalogue is not evidence that an option is rejected, so
    # only a model that declares the refusal loses its temperature.
    if request.temperature is not None and info is not None and not info.supports_temperature:
        changes["temperature"] = None

    return replace(request, **changes) if changes else request


_EFFORT_ORDER: tuple[ReasoningEffort, ...] = get_args(ReasoningEffort)


def _effort_for_model(request: LLMRequest, info: ModelInfo | None) -> ReasoningEffort | None:
    """The supported effort nearest to the one asked for, ties going cheaper.

    Nearest rather than a fixed middle: ``none`` becoming ``medium`` would bill
    the cheapest request mid-range, and ``max`` becoming ``medium`` would
    quietly halve it.
    """
    effort = request.reasoning_effort
    # Unknown and non-thinking models must not receive a provider-specific
    # reasoning field that the API may reject.
    if effort is None or info is None or not info.reasoning_efforts:
        return None
    if effort in info.reasoning_efforts:
        return effort
    wanted = _EFFORT_ORDER.index(effort)
    return min(
        info.reasoning_efforts,
        key=lambda option: (abs(_EFFORT_ORDER.index(option) - wanted), _EFFORT_ORDER.index(option)),
    )


def _interpret(
    response: ProviderResponse, request: LLMRequest
) -> tuple[object, tuple[ToolCall, ...]]:
    """Turn a provider reply into what the caller asked for, or into calls.

    A model that called a tool did not answer, so there is no text to parse and
    no schema to satisfy: asking for JSON here would turn a correct reply into
    a parsing failure and hand a paid-for call to the fallback.
    """
    if response.tool_calls:
        return None, _tool_calls(response, request)

    if request.response_format is ResponseFormat.TEXT:
        return response.output_text or "", ()

    payload = parse_json_payload(response.output_text)
    if request.response_format is ResponseFormat.JSON_OBJECT:
        # A provider without an enforced JSON mode can answer `42` or `true`,
        # which parses and is still not the object the caller was promised.
        if not isinstance(payload, dict):
            raise OutputParsingError(
                f"the model returned a JSON {type(payload).__name__} where an object was required"
            )
        return payload, ()

    schema = request.response_schema
    assert schema is not None  # guaranteed by LLMRequest validation
    try:
        return schema.model_validate(payload), ()
    except ValidationError as error:
        field_names = _schema_field_names(schema)
        details = []
        for item in error.errors():
            location = tuple(
                part
                if isinstance(part, int) or (isinstance(part, str) and part in field_names)
                else "<dynamic>"
                for part in item["loc"]
            )
            details.append(f"loc={location!r} type={item['type']}")
        raise SchemaValidationError(
            f"the response did not satisfy {schema.__name__}: "
            f"{error.error_count()} violation(s): {'; '.join(details)}"
        ) from error


def _tool_calls(response: ProviderResponse, request: LLMRequest) -> tuple[ToolCall, ...]:
    """Parse and check what the application is about to be asked to run.

    A call it cannot dispatch is a failed attempt, not a result: the provider
    answered and is still billed for it, and the fallback still gets its turn.
    Deep schema validation is *not* done here — that needs a JSON Schema
    validator this package does not depend on, and the application that owns
    the function has to defend itself anyway. What is checked is what makes a
    call dispatchable at all: a declared name and a JSON object of arguments.

    No message repeats the arguments. They hold whatever the model was told,
    which is exactly the material that must not reach a log.
    """
    declared = {tool.name for tool in request.tools}
    calls: list[ToolCall] = []
    seen_ids: set[str] = set()
    for raw in response.tool_calls:
        if not isinstance(raw.id, str) or not raw.id.strip():
            raise OutputParsingError("the provider returned a tool call without a correlation id")
        # Two results for one id cannot be told apart, and the continuation
        # replaying them is refused only after this answer has been paid for.
        if raw.id in seen_ids:
            raise OutputParsingError("the provider returned two tool calls with one correlation id")
        seen_ids.add(raw.id)
        if raw.name not in declared:
            raise OutputParsingError(
                f"the provider called {raw.name!r}, which this request did not declare"
            )
        # A function taking no arguments is reported as "" by some providers
        # and as "{}" by others; both mean the same call.
        arguments = parse_json_payload(raw.arguments) if raw.arguments.strip() else {}
        if not isinstance(arguments, dict):
            raise OutputParsingError(
                f"the arguments for {raw.name!r} are {type(arguments).__name__}, not an object"
            )
        calls.append(ToolCall(id=raw.id, name=raw.name, arguments=arguments))
    return tuple(calls)


def _schema_field_names(schema: type[BaseModel]) -> set[str]:
    """Names from the schema are safe to log; dynamic response keys are not."""
    names: set[str] = set()
    pending: list[object] = [schema.model_json_schema()]
    while pending:
        node = pending.pop()
        if isinstance(node, list):
            pending.extend(node)
            continue
        if not isinstance(node, dict):
            continue
        properties = node.get("properties")
        if isinstance(properties, dict):
            names.update(key for key in properties if isinstance(key, str))
        pending.extend(node.values())
    return names


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
        "cost_measurement": cost.measurement.value,
        "pricing_version": cost.pricing_version,
    }

"""Retries, fallback and accounting for image and video generation.

The same orchestration as the token and audio paths, for the same reason: a
provider that rate-limits an image request fails exactly like one that
rate-limits a text request, and a fallback that switched provider silently
would hide which one produced the picture that was charged for.

Image and video are two gateways rather than one generic gateway because
their usage, cost and result types differ all the way down, and a shared
generic would trade three readable classes for one that pleases neither
mypy nor a reader chasing where a video second was priced.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from llm_gateway.catalogs import builtin_image_price_catalog, builtin_video_price_catalog
from llm_gateway.contracts import AttemptOutcome, FailurePhase
from llm_gateway.errors import (
    AllImagesFailed,
    AllVideosFailed,
    ConfigurationError,
    LLMGatewayError,
    ProviderError,
    ProviderTimeoutError,
)
from llm_gateway.media import (
    ImageAttempt,
    ImageExecution,
    ImageRequest,
    ImageResult,
    ProviderImageResponse,
    ProviderVideoResponse,
    VideoAttempt,
    VideoExecution,
    VideoJob,
    VideoJobResult,
    VideoJobStatus,
    VideoRequest,
    VideoResult,
)
from llm_gateway.models import lookup_model
from llm_gateway.ports import (
    AlertSink,
    EventSink,
    ImageUsageSink,
    NullAlertSink,
    NullEventSink,
    NullImageUsageSink,
    NullVideoUsageSink,
    VideoUsageSink,
    image_execution_to_record,
    video_execution_to_record,
)
from llm_gateway.pricing import ImageCost, ImagePriceCatalog, VideoCost, VideoPriceCatalog
from llm_gateway.providers.base import (
    ImageProviderAdapter,
    VideoJobProviderAdapter,
    VideoProviderAdapter,
)
from llm_gateway.registry import ProviderRegistry
from llm_gateway.usage import ImageUsage, VideoUsage


class ImageGateway:
    """Provider-agnostic entry point for image generation and editing."""

    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        price_catalog: ImagePriceCatalog | None = None,
        usage_sink: ImageUsageSink | None = None,
        event_sink: EventSink | None = None,
        alert_sink: AlertSink | None = None,
    ) -> None:
        self._registry = registry
        self._prices = price_catalog or builtin_image_price_catalog()
        self._usage_sink = usage_sink or NullImageUsageSink()
        self._events = event_sink or NullEventSink()
        self._alerts = alert_sink or NullAlertSink()

    async def generate_image(self, request: ImageRequest) -> ImageResult:
        attempts: list[ImageAttempt] = []
        try:
            async with asyncio.timeout(request.timeout_policy.total_seconds):
                return await self._run(request, attempts)
        except TimeoutError:
            self._report_failure(request, attempts)
            raise AllImagesFailed(
                f"the image call exceeded its total budget of "
                f"{request.timeout_policy.total_seconds}s after {len(attempts)} attempt(s)",
                attempts=tuple(attempts),
            ) from None

    async def _run(self, request: ImageRequest, attempts: list[ImageAttempt]) -> ImageResult:
        started = time.perf_counter()
        plan = [request.model, *request.fallback_policy.models]
        for model in plan:
            self._registry.resolve(model)
            _require_image_model(model)

        last_failure: LLMGatewayError | None = None
        for model in plan:
            adapter = self._registry.resolve(model)
            outcome = await self._attempt_model(request, model=model, attempts=attempts)
            if isinstance(outcome, ProviderImageResponse):
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                execution = ImageExecution(
                    requested_model=request.model,
                    model_used=outcome.model_used or model,
                    provider=adapter.name,
                    attempts=tuple(attempts),
                    latency_ms=elapsed_ms,
                )
                usage, cost = _aggregate(attempts)
                if execution.fallback_used:
                    self._alerts.alert(
                        "llm_image_fallback_used",
                        {
                            "requested_model": request.model,
                            "model_used": execution.model_used,
                            "request_id": request.request_id,
                        },
                    )
                self._record(request, execution, usage=usage, cost=cost, succeeded=True)
                self._events.emit(
                    "llm_image_generation_succeeded",
                    _event_fields(request, execution, usage=usage, cost=cost),
                )
                return ImageResult(
                    images=outcome.images,
                    usage=usage,
                    execution=execution,
                    cost=cost,
                )
            last_failure = outcome

        self._report_failure(request, attempts, started=started)
        raise AllImagesFailed(
            f"all {len(attempts)} image attempt(s) failed for model {request.model!r}",
            attempts=tuple(attempts),
        ) from last_failure

    async def _attempt_model(
        self,
        request: ImageRequest,
        *,
        model: str,
        attempts: list[ImageAttempt],
    ) -> ProviderImageResponse | LLMGatewayError:
        adapter = self._registry.resolve(model)
        policy = request.retry_policy
        last_failure: LLMGatewayError | None = None

        for attempt_number in range(1, policy.max_attempts + 1):
            attempt_started = time.perf_counter()
            try:
                if not isinstance(adapter, ImageProviderAdapter):
                    raise ConfigurationError(
                        f"provider {adapter.name} does not support image generation"
                    )
                async with asyncio.timeout(request.timeout_policy.per_attempt_seconds):
                    response = await adapter.generate_image(request, model=model)
                if not response.images:
                    # An empty reply is a failure, not a successful call that
                    # happened to produce nothing: the attempt was still billed.
                    raise ProviderError(f"{adapter.name} returned no image")
            except TimeoutError as error:
                failure: LLMGatewayError = ProviderTimeoutError(
                    f"image attempt exceeded {request.timeout_policy.per_attempt_seconds}s"
                )
                failure.__cause__ = error
            except asyncio.CancelledError:
                attempts.append(
                    _record_attempt(
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome="failed",
                        usage=ImageUsage.unknown(),
                        cost=ImageCost.unavailable(pricing_version=self._prices.version),
                        started=attempt_started,
                        error_type=ProviderTimeoutError.__name__,
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
                    usage=ImageUsage.unknown(),
                    cost=ImageCost.unavailable(pricing_version=self._prices.version),
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
        request: ImageRequest,
        attempts: list[ImageAttempt],
        *,
        started: float | None = None,
    ) -> None:
        usage, cost = _aggregate(attempts)
        elapsed_ms = int((time.perf_counter() - started) * 1000) if started is not None else 0
        execution = ImageExecution(
            requested_model=request.model,
            model_used=attempts[-1].model if attempts else request.model,
            provider=attempts[-1].provider if attempts else "unknown",
            attempts=tuple(attempts),
            latency_ms=elapsed_ms,
        )
        self._record(request, execution, usage=usage, cost=cost, succeeded=False)
        self._events.emit(
            "llm_image_generation_failed",
            _event_fields(request, execution, usage=usage, cost=cost),
        )

    def _record(
        self,
        request: ImageRequest,
        execution: ImageExecution,
        *,
        usage: ImageUsage,
        cost: ImageCost,
        succeeded: bool,
    ) -> None:
        self._usage_sink.record(
            image_execution_to_record(
                execution,
                usage=usage,
                cost=cost,
                request_id=request.request_id,
                source=request.source,
                succeeded=succeeded,
            )
        )


def _require_image_model(model: str) -> None:
    info = lookup_model(model)
    if info is not None and info.modality != "image":
        raise ConfigurationError(f"{model!r} does not generate images; use LLMGateway.generate()")


def _record_attempt(
    *,
    index: int,
    model: str,
    provider: str,
    outcome: str,
    usage: ImageUsage,
    cost: ImageCost,
    started: float,
    error_type: str | None = None,
    billable: bool = True,
    failure_phase: FailurePhase | None = None,
) -> ImageAttempt:
    return ImageAttempt(
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


def _aggregate(attempts: list[ImageAttempt]) -> tuple[ImageUsage, ImageCost]:
    billable = [attempt for attempt in attempts if attempt.billable]
    if not billable:
        return ImageUsage.unknown(), ImageCost.unavailable()
    usage = billable[0].usage
    cost = billable[0].cost
    for attempt in billable[1:]:
        usage = usage.merge(attempt.usage)
        cost = cost.merge(attempt.cost)
    return usage, cost


def _event_fields(
    request: ImageRequest,
    execution: ImageExecution,
    *,
    usage: ImageUsage,
    cost: ImageCost,
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
        "images": usage.images,
        "cost_microusd": cost.microusd,
        "cost_measurement": cost.measurement.value,
        "pricing_version": cost.pricing_version,
    }


class VideoGateway:
    """Provider-agnostic entry point for video generation.

    Video is the slowest thing this package calls: a five-second clip takes
    minutes, so ``VideoRequest`` defaults to a 15-minute total budget and the
    adapter owns the provider's polling loop.
    """

    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        price_catalog: VideoPriceCatalog | None = None,
        usage_sink: VideoUsageSink | None = None,
        event_sink: EventSink | None = None,
        alert_sink: AlertSink | None = None,
    ) -> None:
        self._registry = registry
        self._prices = price_catalog or builtin_video_price_catalog()
        self._usage_sink = usage_sink or NullVideoUsageSink()
        self._events = event_sink or NullEventSink()
        self._alerts = alert_sink or NullAlertSink()

    async def generate_video(self, request: VideoRequest) -> VideoResult:
        attempts: list[VideoAttempt] = []
        try:
            async with asyncio.timeout(request.timeout_policy.total_seconds):
                return await self._run(request, attempts)
        except TimeoutError:
            self._report_failure(request, attempts)
            raise AllVideosFailed(
                f"the video call exceeded its total budget of "
                f"{request.timeout_policy.total_seconds}s after {len(attempts)} attempt(s)",
                attempts=tuple(attempts),
            ) from None

    async def submit_video(self, request: VideoRequest) -> VideoJob:
        """Create a job on a provider that finishes long after this returns.

        Only the submission is bounded by the timeout policy. The clip takes
        minutes, so waiting for it here is what this method exists to avoid.
        """
        attempts: list[VideoAttempt] = []
        started = time.perf_counter()
        # Resolved up front, so a provider that cannot take a job at all says
        # so before the first request rather than after the last retry.
        plan = [request.model, *request.fallback_policy.models]
        for model in plan:
            _require_job_provider(self._registry.resolve(model))
            _require_video_model(model)

        last_failure: LLMGatewayError | None = None
        for model in plan:
            outcome = await self._attempt_submission(request, model=model, attempts=attempts)
            if isinstance(outcome, VideoJob):
                # Stamped here rather than in each adapter, so no provider can
                # forget it and leave its clip's cost unattributable.
                outcome = replace(outcome, request_id=request.request_id, source=request.source)
                if outcome.model != request.model:
                    self._alerts.alert(
                        "llm_video_fallback_used",
                        {
                            "requested_model": request.model,
                            "model_used": outcome.model,
                            "request_id": request.request_id,
                        },
                    )
                self._events.emit(
                    "llm_video_job_submitted",
                    {
                        "request_id": request.request_id,
                        "source": request.source,
                        "provider": outcome.provider,
                        "requested_model": request.model,
                        "model_used": outcome.model,
                        "attempts": len(attempts),
                        "status": outcome.status.value,
                    },
                )
                # Nothing is recorded yet on purpose: the clip does not exist,
                # so any amount here would be invented. The terminal poll bills.
                return outcome
            last_failure = outcome

        self._report_failure(request, attempts, started=started)
        raise AllVideosFailed(
            f"all {len(attempts)} video submission(s) failed for model {request.model!r}",
            attempts=tuple(attempts),
        ) from last_failure

    async def _attempt_submission(
        self,
        request: VideoRequest,
        *,
        model: str,
        attempts: list[VideoAttempt],
    ) -> VideoJob | LLMGatewayError:
        adapter = self._registry.resolve(model)
        policy = request.retry_policy
        last_failure: LLMGatewayError | None = None

        for attempt_number in range(1, policy.max_attempts + 1):
            attempt_started = time.perf_counter()
            try:
                _require_job_provider(adapter)
                assert isinstance(adapter, VideoJobProviderAdapter)  # narrowed by the check
                async with asyncio.timeout(request.timeout_policy.per_attempt_seconds):
                    job = await adapter.submit_video(request, model=model)
            except TimeoutError as error:
                failure: LLMGatewayError = ProviderTimeoutError(
                    f"video submission exceeded {request.timeout_policy.per_attempt_seconds}s"
                )
                failure.__cause__ = error
            except LLMGatewayError as error:
                failure = error
            else:
                return job

            last_failure = failure
            attempts.append(
                _record_video_attempt(
                    index=len(attempts) + 1,
                    model=model,
                    provider=adapter.name,
                    outcome="failed",
                    usage=VideoUsage.unknown(),
                    cost=VideoCost.unavailable(pricing_version=self._prices.version),
                    started=attempt_started,
                    error_type=type(failure).__name__,
                    # A submission that never produced a job produced no clip,
                    # so unlike a generation attempt it was not billed.
                    billable=False,
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

    async def poll_video(self, job: VideoJob, *, timeout_seconds: float = 30.0) -> VideoJobResult:
        """Read a job's state, and its clip once the provider has one.

        Raises only when the *reading* failed. A job the provider gave up on
        is a status, not an exception: the submission worked, and an
        application storing the outcome needs the reason, not a traceback.

        ``timeout_seconds`` bounds this one status call, not the job, which may
        legitimately run for minutes. Its own budget because a poll takes no
        ``VideoRequest``, and an unbounded one blocks the worker that made it
        on a provider that stopped answering.
        """
        adapter = self._registry.by_name(job.provider)
        _require_job_provider(adapter)
        assert isinstance(adapter, VideoJobProviderAdapter)  # narrowed by the check

        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_seconds):
                update = await adapter.poll_video(job)
        except TimeoutError as error:
            raise ProviderTimeoutError(
                f"polling video job {job.id} exceeded {timeout_seconds}s"
            ) from error
        polled = job.with_status(update.status)

        if not update.status.is_terminal:
            return VideoJobResult(
                job=polled,
                cost=VideoCost.unavailable(pricing_version=self._prices.version),
                error=update.error,
            )

        if update.status is VideoJobStatus.SUCCEEDED and not update.videos:
            # A success carrying nothing would be stored as a finished, empty
            # clip and never retried. The provider was still paid for it.
            raise AllVideosFailed(
                f"{job.provider} reported job {job.id} as succeeded with no video",
                attempts=(),
            )

        succeeded = update.status is VideoJobStatus.SUCCEEDED
        usage = update.usage if succeeded else VideoUsage.unknown()
        cost = (
            self._prices.estimate(job.model, usage)
            if succeeded
            else VideoCost.unavailable(pricing_version=self._prices.version)
        )
        execution = VideoExecution(
            requested_model=job.model,
            model_used=job.model,
            provider=job.provider,
            attempts=(
                _record_video_attempt(
                    index=1,
                    model=job.model,
                    provider=job.provider,
                    outcome="succeeded" if succeeded else "failed",
                    usage=usage,
                    cost=cost,
                    started=started,
                    error_type=None if succeeded else "VideoJobFailed",
                    failure_phase=None if succeeded else FailurePhase.PROVIDER,
                ),
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        self._usage_sink.record(
            video_execution_to_record(
                execution,
                usage=usage,
                cost=cost,
                request_id=job.request_id,
                source=job.source,
                succeeded=succeeded,
            )
        )
        self._events.emit(
            "llm_video_job_succeeded" if succeeded else "llm_video_job_failed",
            {
                "provider": job.provider,
                "requested_model": job.model,
                "model_used": job.model,
                "status": update.status.value,
                "video_seconds": usage.seconds,
                "resolution": usage.resolution,
                "cost_microusd": cost.microusd,
                "cost_measurement": cost.measurement.value,
                "pricing_version": cost.pricing_version,
            },
        )
        return VideoJobResult(
            job=polled,
            videos=update.videos,
            usage=usage,
            cost=cost,
            error=update.error,
        )

    async def _run(self, request: VideoRequest, attempts: list[VideoAttempt]) -> VideoResult:
        started = time.perf_counter()
        plan = [request.model, *request.fallback_policy.models]
        for model in plan:
            self._registry.resolve(model)
            _require_video_model(model)

        last_failure: LLMGatewayError | None = None
        for model in plan:
            adapter = self._registry.resolve(model)
            outcome = await self._attempt_model(request, model=model, attempts=attempts)
            if isinstance(outcome, ProviderVideoResponse):
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                execution = VideoExecution(
                    requested_model=request.model,
                    model_used=outcome.model_used or model,
                    provider=adapter.name,
                    attempts=tuple(attempts),
                    latency_ms=elapsed_ms,
                )
                usage, cost = _aggregate_video(attempts)
                if execution.fallback_used:
                    self._alerts.alert(
                        "llm_video_fallback_used",
                        {
                            "requested_model": request.model,
                            "model_used": execution.model_used,
                            "request_id": request.request_id,
                        },
                    )
                self._record(request, execution, usage=usage, cost=cost, succeeded=True)
                self._events.emit(
                    "llm_video_generation_succeeded",
                    _video_event_fields(request, execution, usage=usage, cost=cost),
                )
                return VideoResult(
                    videos=outcome.videos,
                    usage=usage,
                    execution=execution,
                    cost=cost,
                )
            last_failure = outcome

        self._report_failure(request, attempts, started=started)
        raise AllVideosFailed(
            f"all {len(attempts)} video attempt(s) failed for model {request.model!r}",
            attempts=tuple(attempts),
        ) from last_failure

    async def _attempt_model(
        self,
        request: VideoRequest,
        *,
        model: str,
        attempts: list[VideoAttempt],
    ) -> ProviderVideoResponse | LLMGatewayError:
        adapter = self._registry.resolve(model)
        policy = request.retry_policy
        last_failure: LLMGatewayError | None = None

        for attempt_number in range(1, policy.max_attempts + 1):
            attempt_started = time.perf_counter()
            try:
                if not isinstance(adapter, VideoProviderAdapter):
                    raise ConfigurationError(
                        f"provider {adapter.name} does not support video generation"
                    )
                async with asyncio.timeout(request.timeout_policy.per_attempt_seconds):
                    response = await adapter.generate_video(request, model=model)
                if not response.videos:
                    raise ProviderError(f"{adapter.name} returned no video")
            except TimeoutError as error:
                failure: LLMGatewayError = ProviderTimeoutError(
                    f"video attempt exceeded {request.timeout_policy.per_attempt_seconds}s"
                )
                failure.__cause__ = error
            except asyncio.CancelledError:
                attempts.append(
                    _record_video_attempt(
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome="failed",
                        usage=VideoUsage.unknown(),
                        cost=VideoCost.unavailable(pricing_version=self._prices.version),
                        started=attempt_started,
                        error_type=ProviderTimeoutError.__name__,
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
                    _record_video_attempt(
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
                _record_video_attempt(
                    index=len(attempts) + 1,
                    model=model,
                    provider=adapter.name,
                    outcome="failed",
                    usage=VideoUsage.unknown(),
                    cost=VideoCost.unavailable(pricing_version=self._prices.version),
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
        request: VideoRequest,
        attempts: list[VideoAttempt],
        *,
        started: float | None = None,
    ) -> None:
        usage, cost = _aggregate_video(attempts)
        elapsed_ms = int((time.perf_counter() - started) * 1000) if started is not None else 0
        execution = VideoExecution(
            requested_model=request.model,
            model_used=attempts[-1].model if attempts else request.model,
            provider=attempts[-1].provider if attempts else "unknown",
            attempts=tuple(attempts),
            latency_ms=elapsed_ms,
        )
        self._record(request, execution, usage=usage, cost=cost, succeeded=False)
        self._events.emit(
            "llm_video_generation_failed",
            _video_event_fields(request, execution, usage=usage, cost=cost),
        )

    def _record(
        self,
        request: VideoRequest,
        execution: VideoExecution,
        *,
        usage: VideoUsage,
        cost: VideoCost,
        succeeded: bool,
    ) -> None:
        self._usage_sink.record(
            video_execution_to_record(
                execution,
                usage=usage,
                cost=cost,
                request_id=request.request_id,
                source=request.source,
                succeeded=succeeded,
            )
        )


def _require_job_provider(adapter: object) -> None:
    """Refuse a provider whose video is awaited rather than submitted.

    The two shapes are not interchangeable, and the wrong one is worth an
    error: a caller that stored a job id from a provider which never issues
    one would poll something that does not exist.
    """
    if not isinstance(adapter, VideoJobProviderAdapter):
        name = getattr(adapter, "name", "unknown")
        raise ConfigurationError(
            f"provider {name} cannot submit or poll a video job; "
            f"use LLMGateway.generate_video() instead"
        )


def _require_video_model(model: str) -> None:
    info = lookup_model(model)
    if info is not None and info.modality != "video":
        raise ConfigurationError(f"{model!r} does not generate video; use LLMGateway.generate()")


def _record_video_attempt(
    *,
    index: int,
    model: str,
    provider: str,
    outcome: str,
    usage: VideoUsage,
    cost: VideoCost,
    started: float,
    error_type: str | None = None,
    billable: bool = True,
    failure_phase: FailurePhase | None = None,
) -> VideoAttempt:
    return VideoAttempt(
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


def _aggregate_video(attempts: list[VideoAttempt]) -> tuple[VideoUsage, VideoCost]:
    billable = [attempt for attempt in attempts if attempt.billable]
    if not billable:
        return VideoUsage.unknown(), VideoCost.unavailable()
    usage = billable[0].usage
    cost = billable[0].cost
    for attempt in billable[1:]:
        usage = usage.merge(attempt.usage)
        cost = cost.merge(attempt.cost)
    return usage, cost


def _video_event_fields(
    request: VideoRequest,
    execution: VideoExecution,
    *,
    usage: VideoUsage,
    cost: VideoCost,
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
        "video_seconds": usage.seconds,
        "resolution": usage.resolution,
        "cost_microusd": cost.microusd,
        "cost_measurement": cost.measurement.value,
        "pricing_version": cost.pricing_version,
    }

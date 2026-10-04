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
import math
import time
from dataclasses import replace

from llm_gateway._attempts import (
    aggregate,
    elapsed_ms,
    fallback_alert_fields,
    new_attempt,
    phase_of,
)
from llm_gateway.catalogs import builtin_image_price_catalog, builtin_video_price_catalog
from llm_gateway.contracts import AttemptOutcome, FailurePhase
from llm_gateway.errors import (
    AllImagesFailed,
    AllVideosFailed,
    ConfigurationError,
    LLMGatewayError,
    ProviderError,
    ProviderTimeoutError,
    message_of,
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
    ProviderAdapter,
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
            raise AllImagesFailed(
                f"the image call exceeded its total budget of "
                f"{request.timeout_policy.total_seconds}s after {len(attempts)} attempt(s)",
                attempts=tuple(attempts),
            ) from None

    async def _run(
        self, request: ImageRequest, attempts: list[ImageAttempt], *, started: float
    ) -> ImageResult:
        plan = [request.model, *request.fallback_policy.models]
        adapters = {model: self._registry.resolve(model) for model in plan}
        for model in plan:
            _require_image_model(model)

        last_failure: LLMGatewayError | None = None
        for model in plan:
            adapter = adapters[model]
            outcome = await self._attempt_model(
                request, model=model, adapter=adapter, attempts=attempts
            )
            if isinstance(outcome, ProviderImageResponse):
                execution = ImageExecution(
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
                        "llm_image_fallback_used",
                        fallback_alert_fields(
                            requested_model=request.model,
                            model_used=execution.model_used,
                            request_id=request.request_id,
                            attempts=attempts,
                        ),
                    )
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
        adapter: ProviderAdapter,
        attempts: list[ImageAttempt],
    ) -> ProviderImageResponse | LLMGatewayError:
        policy = request.retry_policy

        for attempt_number in range(1, policy.max_attempts + 1):
            attempt_started = time.perf_counter()
            # A failure costs an unknown amount unless the provider said what
            # it used, which only a reply that arrived can do.
            usage = ImageUsage.unknown()
            cost = ImageCost.unavailable(pricing_version=self._prices.version)
            try:
                if not isinstance(adapter, ImageProviderAdapter):
                    raise ConfigurationError(
                        f"provider {adapter.name} does not support image generation"
                    )
                async with asyncio.timeout(request.timeout_policy.per_attempt_seconds):
                    response = await adapter.generate_image(request, model=model)
            except TimeoutError as error:
                failure: LLMGatewayError = ProviderTimeoutError(
                    f"image attempt exceeded {request.timeout_policy.per_attempt_seconds}s"
                )
                failure.__cause__ = error
            except asyncio.CancelledError:
                attempts.append(
                    new_attempt(
                        ImageAttempt,
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome=AttemptOutcome.FAILED,
                        usage=ImageUsage.unknown(),
                        cost=ImageCost.unavailable(pricing_version=self._prices.version),
                        started=attempt_started,
                        error_type=ProviderTimeoutError.__name__,
                        error_message="image attempt cancelled by the total timeout budget",
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
                if response.images:
                    attempts.append(
                        new_attempt(
                            ImageAttempt,
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
                # An empty reply is a failure, not a successful call that
                # happened to produce nothing. It was still billed, so what the
                # provider reported using stays on the attempt.
                failure = ProviderError(f"{adapter.name} returned no image")

            attempts.append(
                new_attempt(
                    ImageAttempt,
                    index=len(attempts) + 1,
                    model=model,
                    provider=adapter.name,
                    outcome=AttemptOutcome.FAILED,
                    usage=usage,
                    cost=cost,
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
        request: ImageRequest,
        attempts: list[ImageAttempt],
        *,
        started: float,
    ) -> None:
        usage, cost = _aggregate(attempts)
        execution = ImageExecution(
            requested_model=request.model,
            model_used=attempts[-1].model if attempts else request.model,
            provider=attempts[-1].provider if attempts else "unknown",
            attempts=tuple(attempts),
            latency_ms=elapsed_ms(started),
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


def _aggregate(attempts: list[ImageAttempt]) -> tuple[ImageUsage, ImageCost]:
    return aggregate(
        attempts, unknown_usage=ImageUsage.unknown(), unavailable_cost=ImageCost.unavailable()
    )


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
        budget = asyncio.timeout(request.timeout_policy.total_seconds)
        try:
            async with budget:
                return await self._submit_video(request, attempts, started=started)
        except TimeoutError:
            # Only the budget expiring is the call running out of time; an
            # application sink timing out after a success is not.
            if not budget.expired():
                raise
            self._report_failure(
                request,
                attempts,
                started=started,
                event="llm_video_job_submission_failed",
            )
            raise AllVideosFailed(
                f"the video submission exceeded its total budget of "
                f"{request.timeout_policy.total_seconds}s after {len(attempts)} attempt(s)",
                attempts=tuple(attempts),
            ) from None

    async def _submit_video(
        self,
        request: VideoRequest,
        attempts: list[VideoAttempt],
        *,
        started: float,
    ) -> VideoJob:
        # Resolved up front, so a provider that cannot take a job at all says
        # so before the first request rather than after the last retry.
        plan = [request.model, *request.fallback_policy.models]
        adapters = {model: _job_provider(self._registry.resolve(model)) for model in plan}
        for model in plan:
            _require_video_model(model)

        last_failure: LLMGatewayError | None = None
        for model in plan:
            outcome = await self._attempt_submission(
                request, model=model, adapter=adapters[model], attempts=attempts
            )
            if isinstance(outcome, VideoJob):
                # Stamped here rather than in each adapter, so no provider can
                # forget it and leave its clip's cost unattributable.
                outcome = replace(outcome, request_id=request.request_id, source=request.source)
                if outcome.status.is_terminal:
                    # Only a poll bills a job, and a poll that is handed a
                    # terminal job takes it as already billed. One that finished
                    # before submission returned is therefore handed back as
                    # queued, so its first poll is the one that records it.
                    outcome = outcome.with_status(VideoJobStatus.QUEUED)
                if any(attempt.billable for attempt in attempts):
                    # The job that came back is billed by the poll that finishes
                    # it; the failures before it never will be. A submission
                    # that failed after reaching the provider may have left a
                    # prediction running whose id nobody holds, so it is
                    # recorded now, as the failure it was, rather than lost.
                    usage, cost = _aggregate_video(attempts)
                    self._record(
                        request,
                        self._failed_execution(request, attempts, started=started),
                        usage=usage,
                        cost=cost,
                        succeeded=False,
                    )
                # After the record: an alert hook that fails must not leave
                # those earlier, possibly paid, attempts unrecorded.
                if outcome.model != request.model:
                    self._alerts.alert(
                        "llm_video_fallback_used",
                        fallback_alert_fields(
                            requested_model=request.model,
                            model_used=outcome.model,
                            request_id=request.request_id,
                            attempts=attempts,
                        ),
                    )
                self._events.emit(
                    "llm_video_job_submitted",
                    {
                        "request_id": request.request_id,
                        "source": request.source,
                        "provider": outcome.provider,
                        "requested_model": request.model,
                        "model_used": outcome.model,
                        "attempts": len(attempts) + 1,
                        "status": outcome.status.value,
                    },
                )
                # The job itself is not recorded yet on purpose: the clip does
                # not exist, so any amount here would be invented. The terminal
                # poll bills it.
                return outcome
            last_failure = outcome

        self._report_failure(
            request,
            attempts,
            started=started,
            event="llm_video_job_submission_failed",
        )
        raise AllVideosFailed(
            f"all {len(attempts)} video submission(s) failed for model {request.model!r}",
            attempts=tuple(attempts),
        ) from last_failure

    async def _attempt_submission(
        self,
        request: VideoRequest,
        *,
        model: str,
        adapter: VideoJobProviderAdapter,
        attempts: list[VideoAttempt],
    ) -> VideoJob | LLMGatewayError:
        policy = request.retry_policy

        for attempt_number in range(1, policy.max_attempts + 1):
            attempt_started = time.perf_counter()
            try:
                async with asyncio.timeout(request.timeout_policy.per_attempt_seconds):
                    job = await adapter.submit_video(request, model=model)
            except asyncio.CancelledError:
                # The outer total budget may expire after the provider has
                # accepted the submission but before its id reaches us. That
                # can leave a billable orphan, so it cannot be recorded free.
                attempts.append(
                    new_attempt(
                        VideoAttempt,
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome=AttemptOutcome.FAILED,
                        usage=VideoUsage.unknown(),
                        cost=VideoCost.unavailable(pricing_version=self._prices.version),
                        started=attempt_started,
                        error_type=ProviderTimeoutError.__name__,
                        error_message="video submission cancelled by the total timeout budget",
                        billable=True,
                        failure_phase=FailurePhase.TIMEOUT,
                    )
                )
                raise
            except TimeoutError as error:
                failure: LLMGatewayError = ProviderTimeoutError(
                    f"video submission exceeded {request.timeout_policy.per_attempt_seconds}s"
                )
                failure.__cause__ = error
            except LLMGatewayError as error:
                failure = error
            else:
                return job

            attempts.append(
                new_attempt(
                    VideoAttempt,
                    index=len(attempts) + 1,
                    model=model,
                    provider=adapter.name,
                    outcome=AttemptOutcome.FAILED,
                    usage=VideoUsage.unknown(),
                    cost=VideoCost.unavailable(pricing_version=self._prices.version),
                    started=attempt_started,
                    error_type=type(failure).__name__,
                    error_message=message_of(failure),
                    # Not only a timeout: an id that never came back, or a
                    # status nobody could read, is raised after the prediction
                    # was created. Only a request refused before dispatch
                    # certainly left nothing running.
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

    async def poll_video(self, job: VideoJob, *, timeout_seconds: float = 30.0) -> VideoJobResult:
        """Read a job's state, and its clip once the provider has one.

        Raises only when the *reading* failed. A job the provider gave up on
        is a status, not an exception: the submission worked, and an
        application storing the outcome needs the reason, not a traceback.

        ``timeout_seconds`` bounds this one status call, not the job, which may
        legitimately run for minutes. Its own budget because a poll takes no
        ``VideoRequest``, and an unbounded one blocks the worker that made it
        on a provider that stopped answering.

        A job passed in already terminal is read but not recorded again: the
        poll that returned it terminal is the one that billed it, and a
        webhook handler re-reading what a worker stored would otherwise put
        the same clip in the ledger twice.
        """
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("polling timeout must be positive and finite")

        adapter = _job_provider(self._registry.by_name(job.provider))

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

        already_billed = job.status.is_terminal
        if update.status is VideoJobStatus.SUCCEEDED and not update.videos:
            # A success carrying nothing would be stored as a finished, empty
            # clip and never retried. The provider was still paid for it, so
            # the refusal is recorded as a billable failure before it is raised.
            message = f"{job.provider} reported job {job.id} as succeeded with no video"
            failed = new_attempt(
                VideoAttempt,
                index=1,
                model=job.model,
                provider=job.provider,
                outcome=AttemptOutcome.FAILED,
                usage=update.usage,
                cost=self._prices.estimate(job.model, update.usage),
                started=started,
                error_type=ProviderError.__name__,
                error_message=message,
                failure_phase=FailurePhase.PROVIDER,
            )
            if not already_billed:
                self._record_job(job, failed, started=started, succeeded=False)
            raise AllVideosFailed(message, attempts=(failed,))

        succeeded = update.status is VideoJobStatus.SUCCEEDED
        usage = update.usage if succeeded else VideoUsage.unknown()
        cost = (
            self._prices.estimate(job.model, usage)
            if succeeded
            else VideoCost.unavailable(pricing_version=self._prices.version)
        )
        if not already_billed:
            attempt = new_attempt(
                VideoAttempt,
                index=1,
                model=job.model,
                provider=job.provider,
                outcome=AttemptOutcome.SUCCEEDED if succeeded else AttemptOutcome.FAILED,
                usage=usage,
                cost=cost,
                started=started,
                error_type=None if succeeded else "VideoJobFailed",
                error_message=None if succeeded else (update.error or None),
                failure_phase=None if succeeded else FailurePhase.PROVIDER,
            )
            self._record_job(job, attempt, started=started, succeeded=succeeded)
            self._events.emit(
                "llm_video_job_succeeded" if succeeded else "llm_video_job_failed",
                {
                    "request_id": job.request_id,
                    "source": job.source,
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

    def _record_job(
        self, job: VideoJob, attempt: VideoAttempt, *, started: float, succeeded: bool
    ) -> None:
        execution = VideoExecution(
            requested_model=job.model,
            model_used=job.model,
            provider=job.provider,
            attempts=(attempt,),
            latency_ms=elapsed_ms(started),
        )
        self._usage_sink.record(
            video_execution_to_record(
                execution,
                usage=attempt.usage,
                cost=attempt.cost,
                request_id=job.request_id,
                source=job.source,
                succeeded=succeeded,
            )
        )

    async def _run(
        self, request: VideoRequest, attempts: list[VideoAttempt], *, started: float
    ) -> VideoResult:
        plan = [request.model, *request.fallback_policy.models]
        adapters = {model: self._registry.resolve(model) for model in plan}
        for model in plan:
            _require_video_model(model)

        last_failure: LLMGatewayError | None = None
        for model in plan:
            adapter = adapters[model]
            outcome = await self._attempt_model(
                request, model=model, adapter=adapter, attempts=attempts
            )
            if isinstance(outcome, ProviderVideoResponse):
                execution = VideoExecution(
                    requested_model=request.model,
                    model_used=outcome.model_used or model,
                    provider=adapter.name,
                    attempts=tuple(attempts),
                    latency_ms=elapsed_ms(started),
                )
                usage, cost = _aggregate_video(attempts)
                # Booked first: an alert hook that fails must not leave a paid
                # call unrecorded.
                self._record(request, execution, usage=usage, cost=cost, succeeded=True)
                if execution.fallback_used:
                    self._alerts.alert(
                        "llm_video_fallback_used",
                        fallback_alert_fields(
                            requested_model=request.model,
                            model_used=execution.model_used,
                            request_id=request.request_id,
                            attempts=attempts,
                        ),
                    )
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
        adapter: ProviderAdapter,
        attempts: list[VideoAttempt],
    ) -> ProviderVideoResponse | LLMGatewayError:
        policy = request.retry_policy

        for attempt_number in range(1, policy.max_attempts + 1):
            attempt_started = time.perf_counter()
            # A failure costs an unknown amount unless the provider said what
            # it used, which only a reply that arrived can do.
            usage = VideoUsage.unknown()
            cost = VideoCost.unavailable(pricing_version=self._prices.version)
            try:
                if not isinstance(adapter, VideoProviderAdapter):
                    raise ConfigurationError(
                        f"provider {adapter.name} does not support video generation"
                    )
                async with asyncio.timeout(request.timeout_policy.per_attempt_seconds):
                    response = await adapter.generate_video(request, model=model)
            except TimeoutError as error:
                failure: LLMGatewayError = ProviderTimeoutError(
                    f"video attempt exceeded {request.timeout_policy.per_attempt_seconds}s"
                )
                failure.__cause__ = error
            except asyncio.CancelledError:
                attempts.append(
                    new_attempt(
                        VideoAttempt,
                        index=len(attempts) + 1,
                        model=model,
                        provider=adapter.name,
                        outcome=AttemptOutcome.FAILED,
                        usage=VideoUsage.unknown(),
                        cost=VideoCost.unavailable(pricing_version=self._prices.version),
                        started=attempt_started,
                        error_type=ProviderTimeoutError.__name__,
                        error_message="video attempt cancelled by the total timeout budget",
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
                if response.videos:
                    attempts.append(
                        new_attempt(
                            VideoAttempt,
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
                # Rendered and invoiced, then delivered nothing: the seconds
                # the provider reported stay on the failed attempt.
                failure = ProviderError(f"{adapter.name} returned no video")

            attempts.append(
                new_attempt(
                    VideoAttempt,
                    index=len(attempts) + 1,
                    model=model,
                    provider=adapter.name,
                    outcome=AttemptOutcome.FAILED,
                    usage=usage,
                    cost=cost,
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

    def _failed_execution(
        self, request: VideoRequest, attempts: list[VideoAttempt], *, started: float
    ) -> VideoExecution:
        return VideoExecution(
            requested_model=request.model,
            model_used=attempts[-1].model if attempts else request.model,
            provider=attempts[-1].provider if attempts else "unknown",
            attempts=tuple(attempts),
            latency_ms=elapsed_ms(started),
        )

    def _report_failure(
        self,
        request: VideoRequest,
        attempts: list[VideoAttempt],
        *,
        started: float,
        event: str = "llm_video_generation_failed",
    ) -> None:
        usage, cost = _aggregate_video(attempts)
        execution = self._failed_execution(request, attempts, started=started)
        self._record(request, execution, usage=usage, cost=cost, succeeded=False)
        self._events.emit(
            event,
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


def _job_provider(adapter: ProviderAdapter) -> VideoJobProviderAdapter:
    """Refuse a provider whose video is awaited rather than submitted.

    The two shapes are not interchangeable, and the wrong one is worth an
    error: a caller that stored a job id from a provider which never issues
    one would poll something that does not exist.
    """
    if not isinstance(adapter, VideoJobProviderAdapter):
        raise ConfigurationError(
            f"provider {adapter.name} cannot submit or poll a video job; "
            f"use LLMGateway.generate_video() instead"
        )
    return adapter


def _require_video_model(model: str) -> None:
    info = lookup_model(model)
    if info is not None and info.modality != "video":
        raise ConfigurationError(f"{model!r} does not generate video; use LLMGateway.generate()")


def _aggregate_video(attempts: list[VideoAttempt]) -> tuple[VideoUsage, VideoCost]:
    return aggregate(
        attempts, unknown_usage=VideoUsage.unknown(), unavailable_cost=VideoCost.unavailable()
    )


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

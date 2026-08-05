"""A video that finishes minutes later cannot be returned by the call that asked for it.

`generate_video()` waits, which works only while the adapter can own the
provider's polling loop. Replicate answers with a prediction id and finishes
long after any sensible request timeout, and the application that submitted it
usually polls from a different process — a worker, or a webhook handler. So the
job is its own contract: `submit_video()` hands back something storable, and
`poll_video()` reads it back later.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from llm_gateway import (
    AllVideosFailed,
    ConfigurationError,
    CostMeasurement,
    GeneratedVideo,
    ImageInput,
    LLMGateway,
    LLMRequest,
    ProviderRegistry,
    ProviderResponse,
    ProviderVideoJobUpdate,
    ProviderVideoResponse,
    RateLimitedError,
    RetryPolicy,
    StaticVideoPriceCatalog,
    VideoJob,
    VideoJobStatus,
    VideoRate,
    VideoRequest,
    VideoUsage,
    VideoUsageRecord,
    lookup_model,
)

MODEL = "wan-video/wan-2.2-5b-fast"
POLLED_MODEL = "wavespeed-ai/minimax-h3/image-to-video"


class RecordingJobAdapter:
    """A provider that answers a submission with an id and nothing else."""

    def __init__(self, name: str, *responses: object) -> None:
        self.name = name
        self._responses = list(responses)
        self.submitted: list[tuple[VideoRequest, str]] = []
        self.polled: list[VideoJob] = []

    async def submit_video(self, request: VideoRequest, *, model: str) -> VideoJob:
        self.submitted.append((request, model))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, VideoJob)
        return response

    async def poll_video(self, job: VideoJob) -> ProviderVideoJobUpdate:
        self.polled.append(job)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, ProviderVideoJobUpdate)
        return response

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise AssertionError("video job adapter is not used for text generation")


class PolledOnlyAdapter:
    """The shape `generate_video()` serves: it waits and returns the clip."""

    name = "wavespeed"

    async def generate_video(self, request: VideoRequest, *, model: str) -> ProviderVideoResponse:
        return ProviderVideoResponse(
            videos=(GeneratedVideo(url="https://cdn.test/clip.mp4"),),
            usage=VideoUsage(seconds=5.0, videos=1, resolution="480p"),
        )

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise AssertionError("video job adapter is not used for text generation")


class VideoSink:
    def __init__(self) -> None:
        self.records: list[VideoUsageRecord] = []

    def record(self, record: VideoUsageRecord) -> None:
        self.records.append(record)


def _request(**kwargs: object) -> VideoRequest:
    return VideoRequest(
        model=str(kwargs.pop("model", MODEL)),
        prompt=str(kwargs.pop("prompt", "a lion sprinting across the savannah")),
        **kwargs,  # type: ignore[arg-type]
    )


def _job(**kwargs: object) -> VideoJob:
    return VideoJob(
        id=str(kwargs.pop("id", "pred-1")),
        model=str(kwargs.pop("model", MODEL)),
        provider=str(kwargs.pop("provider", "replicate")),
        **kwargs,  # type: ignore[arg-type]
    )


def _gateway(*adapters: object, sink: VideoSink | None = None) -> LLMGateway:
    registry = ProviderRegistry()
    for adapter in adapters:
        registry.register(adapter, model_prefixes=())  # type: ignore[arg-type]
    return LLMGateway(registry=registry, video_usage_sink=sink)


class TestTheJobItself:
    def test_a_job_needs_an_id_a_model_and_a_provider(self) -> None:
        """A job with no id cannot be polled, so it is refused where it is made."""
        with pytest.raises(ValueError, match="job id"):
            VideoJob(id="  ", model=MODEL, provider="replicate")
        with pytest.raises(ValueError, match="model"):
            VideoJob(id="pred-1", model=" ", provider="replicate")
        with pytest.raises(ValueError, match="provider"):
            VideoJob(id="pred-1", model=MODEL, provider="")

    def test_a_job_survives_a_round_trip_through_plain_data(self) -> None:
        """The process that polls is rarely the one that submitted."""
        job = _job(status=VideoJobStatus.RUNNING)

        stored = {
            "id": job.id,
            "model": job.model,
            "provider": job.provider,
            "status": job.status.value,
        }
        restored = VideoJob(
            id=stored["id"],
            model=stored["model"],
            provider=stored["provider"],
            status=VideoJobStatus(stored["status"]),
        )

        assert restored == job

    def test_only_a_finished_job_is_terminal(self) -> None:
        assert VideoJobStatus.QUEUED.is_terminal is False
        assert VideoJobStatus.RUNNING.is_terminal is False
        assert VideoJobStatus.SUCCEEDED.is_terminal is True
        assert VideoJobStatus.FAILED.is_terminal is True
        assert VideoJobStatus.CANCELLED.is_terminal is True


class TestSubmitting:
    async def test_a_submission_returns_a_job_rather_than_a_video(self) -> None:
        adapter = RecordingJobAdapter("replicate", _job(status=VideoJobStatus.QUEUED))

        job = await _gateway(adapter).submit_video(
            _request(image=ImageInput(url="https://cdn.test/first-frame.png"))
        )

        assert job.id == "pred-1"
        assert job.provider == "replicate"
        assert job.status is VideoJobStatus.QUEUED
        assert adapter.submitted[0][1] == MODEL

    async def test_a_failed_submission_raises_instead_of_returning_a_dead_job(self) -> None:
        """A job id that was never created would be polled forever."""
        sink = VideoSink()
        adapter = RecordingJobAdapter("replicate", RateLimitedError("429"))

        with pytest.raises(AllVideosFailed) as error:
            await _gateway(adapter, sink=sink).submit_video(_request())

        assert error.value.last_error == "RateLimitedError"
        assert sink.records[0].succeeded is False

    async def test_a_retried_submission_creates_exactly_one_job(self) -> None:
        """Two ids for one request would leave a clip nobody polls, and pays for."""
        adapter = RecordingJobAdapter("replicate", RateLimitedError("429"), _job(id="pred-2"))

        job = await _gateway(adapter).submit_video(
            _request(retry_policy=RetryPolicy.transient(base_delay_seconds=0.0))
        )

        assert job.id == "pred-2"
        assert len(adapter.submitted) == 2

    async def test_a_provider_that_only_polls_cannot_take_a_submission(self) -> None:
        with pytest.raises(ConfigurationError, match="submit"):
            await _gateway(PolledOnlyAdapter()).submit_video(_request(model=POLLED_MODEL))

    async def test_a_provider_that_only_submits_cannot_be_awaited_for_a_clip(self) -> None:
        adapter = RecordingJobAdapter("replicate", _job())

        with pytest.raises(AllVideosFailed):
            await _gateway(adapter).generate_video(_request())

    async def test_an_image_model_cannot_be_submitted_as_a_video_job(self) -> None:
        adapter = RecordingJobAdapter("replicate", _job())

        with pytest.raises(ConfigurationError, match="video"):
            await _gateway(adapter).submit_video(_request(model="bytedance/seedream-4"))


class TestPolling:
    async def test_an_unfinished_job_reports_progress_and_no_video(self) -> None:
        adapter = RecordingJobAdapter(
            "replicate", ProviderVideoJobUpdate(status=VideoJobStatus.RUNNING)
        )

        result = await _gateway(adapter).poll_video(_job())

        assert result.job.status is VideoJobStatus.RUNNING
        assert result.videos == ()
        assert result.cost.measurement is CostMeasurement.UNAVAILABLE
        assert adapter.polled[0].id == "pred-1"

    async def test_a_finished_job_yields_the_clip_and_keeps_its_id(self) -> None:
        adapter = RecordingJobAdapter(
            "replicate",
            ProviderVideoJobUpdate(
                status=VideoJobStatus.SUCCEEDED,
                videos=(GeneratedVideo(url="https://cdn.test/lion.mp4", mime_type="video/mp4"),),
                usage=VideoUsage(seconds=5.0, videos=1),
            ),
        )

        result = await _gateway(adapter).poll_video(_job())

        assert result.job.id == "pred-1"
        assert result.job.status is VideoJobStatus.SUCCEEDED
        assert result.videos[0].url == "https://cdn.test/lion.mp4"
        assert result.usage.seconds == 5.0

    async def test_a_job_the_provider_failed_is_a_status_not_an_exception(self) -> None:
        """The submission was fine; it is the generation that did not finish."""
        adapter = RecordingJobAdapter(
            "replicate",
            ProviderVideoJobUpdate(status=VideoJobStatus.FAILED, error="NSFW content detected"),
        )

        result = await _gateway(adapter).poll_video(_job())

        assert result.job.status is VideoJobStatus.FAILED
        assert result.error == "NSFW content detected"
        assert result.videos == ()

    async def test_a_succeeded_job_with_no_video_is_refused(self) -> None:
        """A success carrying nothing would be stored as a finished empty clip."""
        adapter = RecordingJobAdapter(
            "replicate", ProviderVideoJobUpdate(status=VideoJobStatus.SUCCEEDED)
        )

        with pytest.raises(AllVideosFailed):
            await _gateway(adapter).poll_video(_job())

    async def test_a_transport_failure_while_polling_is_raised_not_swallowed(self) -> None:
        """A 429 on the status call says nothing about the job behind it."""
        adapter = RecordingJobAdapter("replicate", RateLimitedError("429"))

        with pytest.raises(RateLimitedError):
            await _gateway(adapter).poll_video(_job())

    async def test_polling_asks_the_provider_that_holds_the_job(self) -> None:
        replicate = RecordingJobAdapter(
            "replicate", ProviderVideoJobUpdate(status=VideoJobStatus.RUNNING)
        )

        await _gateway(replicate, PolledOnlyAdapter()).poll_video(_job(provider="replicate"))

        assert len(replicate.polled) == 1

    async def test_a_job_from_an_unregistered_provider_is_refused(self) -> None:
        adapter = RecordingJobAdapter("replicate", ProviderVideoJobUpdate())

        with pytest.raises(ConfigurationError, match="luma"):
            await _gateway(adapter).poll_video(_job(provider="luma"))


class TestAccounting:
    async def test_nothing_is_recorded_until_the_job_reaches_a_terminal_state(self) -> None:
        """A clip polled ten times is billed once, not ten times."""
        sink = VideoSink()
        adapter = RecordingJobAdapter(
            "replicate", ProviderVideoJobUpdate(status=VideoJobStatus.RUNNING)
        )

        await _gateway(adapter, sink=sink).poll_video(_job())

        assert sink.records == []

    async def test_a_finished_job_is_recorded_once_with_its_cost(self) -> None:
        sink = VideoSink()
        adapter = RecordingJobAdapter(
            "replicate",
            ProviderVideoJobUpdate(
                status=VideoJobStatus.SUCCEEDED,
                videos=(GeneratedVideo(url="https://cdn.test/lion.mp4"),),
                usage=VideoUsage(seconds=5.0, videos=1),
            ),
        )
        registry = ProviderRegistry()
        registry.register(adapter, model_prefixes=())
        gateway = LLMGateway(
            registry=registry,
            video_usage_sink=sink,
            video_price_catalog=StaticVideoPriceCatalog(
                version="negotiated-2026-08",
                rates={MODEL: VideoRate(usd_per_second=Decimal("0.02"))},
            ),
        )

        result = await gateway.poll_video(_job())

        assert result.cost.amount_usd == Decimal("0.100000")
        assert len(sink.records) == 1
        assert sink.records[0].succeeded is True
        assert sink.records[0].model_used == MODEL

    async def test_a_job_that_failed_is_recorded_as_a_failure(self) -> None:
        sink = VideoSink()
        adapter = RecordingJobAdapter(
            "replicate", ProviderVideoJobUpdate(status=VideoJobStatus.FAILED, error="boom")
        )

        await _gateway(adapter, sink=sink).poll_video(_job())

        assert len(sink.records) == 1
        assert sink.records[0].succeeded is False

    async def test_gpu_billed_video_reports_unavailable_rather_than_free(self) -> None:
        """Replicate bills Wan by GPU time, which no per-second table predicts."""
        adapter = RecordingJobAdapter(
            "replicate",
            ProviderVideoJobUpdate(
                status=VideoJobStatus.SUCCEEDED,
                videos=(GeneratedVideo(url="https://cdn.test/lion.mp4"),),
                usage=VideoUsage(seconds=5.0, videos=1),
            ),
        )

        result = await _gateway(adapter).poll_video(_job())

        assert result.cost.microusd is None
        assert result.cost.measurement is CostMeasurement.UNAVAILABLE

    async def test_the_sink_never_sees_the_prompt_or_the_clip(self) -> None:
        """A prompt is user data and a URL is a capability; neither is accounting."""
        sink = VideoSink()
        adapter = RecordingJobAdapter(
            "replicate",
            ProviderVideoJobUpdate(
                status=VideoJobStatus.SUCCEEDED,
                videos=(GeneratedVideo(url="https://cdn.test/secret.mp4"),),
                usage=VideoUsage(seconds=5.0, videos=1),
            ),
        )

        await _gateway(adapter, sink=sink).poll_video(_job())

        recorded = repr(sink.records[0])
        assert "secret.mp4" not in recorded
        assert "savannah" not in recorded


class TestTheCatalogue:
    def test_the_wan_model_is_catalogued_as_video_without_a_rate(self) -> None:
        info = lookup_model(MODEL)

        assert info is not None
        assert info.modality == "video"
        assert info.provider == "replicate"
        assert info.video_rate is None


class TestWebhooks:
    def test_a_webhook_url_must_be_non_empty_when_given(self) -> None:
        with pytest.raises(ValueError, match="webhook"):
            VideoRequest(model=MODEL, prompt="a lion", webhook_url="  ")

    async def test_the_webhook_url_reaches_the_adapter(self) -> None:
        adapter = RecordingJobAdapter("replicate", _job())

        await _gateway(adapter).submit_video(_request(webhook_url="https://app.test/hooks/video"))

        assert adapter.submitted[0][0].webhook_url == "https://app.test/hooks/video"

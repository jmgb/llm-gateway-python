"""Video is its own operation, priced by the second and by the resolution."""

from __future__ import annotations

import asyncio
import math
from decimal import Decimal

import pytest

from llm_gateway import (
    AllVideosFailed,
    ConfigurationError,
    CostMeasurement,
    FallbackPolicy,
    GeneratedVideo,
    ImageInput,
    LLMGateway,
    LLMRequest,
    Message,
    ProviderRegistry,
    ProviderResponse,
    ProviderVideoResponse,
    RateLimitedError,
    StaticVideoPriceCatalog,
    TimeoutPolicy,
    VideoRate,
    VideoRequest,
    VideoUsage,
    VideoUsageRecord,
    builtin_video_price_catalog,
    lookup_model,
)

MODEL = "wavespeed-ai/minimax-h3/image-to-video"


class RecordingVideoAdapter:
    def __init__(self, name: str, *responses: ProviderVideoResponse | Exception) -> None:
        self.name = name
        self._responses = list(responses)
        self.requests: list[tuple[VideoRequest, str]] = []

    async def generate_video(self, request: VideoRequest, *, model: str) -> ProviderVideoResponse:
        self.requests.append((request, model))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise AssertionError("video test adapter is not used for text generation")


class VideoSink:
    def __init__(self) -> None:
        self.records: list[VideoUsageRecord] = []

    def record(self, record: VideoUsageRecord) -> None:
        self.records.append(record)


def _response(resolution: str = "480p", seconds: float = 5.0) -> ProviderVideoResponse:
    return ProviderVideoResponse(
        videos=(GeneratedVideo(url="https://cdn.test/hunt.mp4", mime_type="video/mp4"),),
        usage=VideoUsage(seconds=seconds, videos=1, resolution=resolution),
    )


def _request(**kwargs: object) -> VideoRequest:
    return VideoRequest(
        model=str(kwargs.pop("model", MODEL)),
        prompt=str(kwargs.pop("prompt", "the lion sprints and leaps at the gazelle")),
        **kwargs,  # type: ignore[arg-type]
    )


class AlertSink:
    def __init__(self) -> None:
        self.alerts: list[tuple[str, dict[str, object]]] = []

    def alert(self, name: str, fields: dict[str, object]) -> None:
        self.alerts.append((name, fields))


def _gateway(
    *adapters: object, sink: VideoSink | None = None, alerts: AlertSink | None = None
) -> LLMGateway:
    registry = ProviderRegistry()
    for adapter in adapters:
        registry.register(adapter, model_prefixes=())  # type: ignore[arg-type]
    return LLMGateway(registry=registry, video_usage_sink=sink, alert_sink=alerts)


def test_a_video_model_is_declared_in_the_catalogue() -> None:
    info = lookup_model(MODEL)

    assert info is not None
    assert info.modality == "video"
    assert info.provider == "wavespeed"
    assert info.pricing_unit == "video_seconds"


async def test_generate_refuses_a_video_model() -> None:
    registry = ProviderRegistry()
    registry.register(RecordingVideoAdapter("wavespeed"), model_prefixes=())
    gateway = LLMGateway(registry=registry)

    with pytest.raises(ConfigurationError, match="generate_video"):
        await gateway.generate(
            LLMRequest(model=MODEL, messages=(Message(role="user", content="a lion"),))
        )


def test_a_video_request_requires_a_prompt_and_a_positive_duration() -> None:
    with pytest.raises(ValueError, match="prompt"):
        VideoRequest(model=MODEL, prompt="  ")
    with pytest.raises(ValueError, match="duration"):
        VideoRequest(model=MODEL, prompt="a lion", duration_seconds=0)


def test_a_generated_video_must_carry_a_url_or_bytes() -> None:
    with pytest.raises(ValueError, match="data or url"):
        GeneratedVideo()

    with pytest.raises(ValueError, match="non-empty"):
        GeneratedVideo(data=b"")
    with pytest.raises(ValueError, match="non-empty"):
        GeneratedVideo(url="   ")


@pytest.mark.parametrize("seconds", [-1.0, math.inf, math.nan])
def test_video_usage_requires_finite_non_negative_seconds(seconds: float) -> None:
    with pytest.raises(ValueError, match="non-negative and finite"):
        VideoUsage(seconds=seconds)


def test_resolution_changes_the_rate_rather_than_being_ignored() -> None:
    catalog = builtin_video_price_catalog()

    cheap = catalog.estimate(MODEL, VideoUsage(seconds=5.0, videos=1, resolution="480p"))
    dear = catalog.estimate(MODEL, VideoUsage(seconds=5.0, videos=1, resolution="768p"))

    assert cheap.amount_usd == Decimal("0.200000")
    assert dear.amount_usd == Decimal("0.400000")


def test_an_unknown_resolution_costs_unavailable_rather_than_the_cheaper_rate() -> None:
    """Guessing 480p on a 768p clip would halve a real invoice."""
    catalog = builtin_video_price_catalog()

    cost = catalog.estimate(MODEL, VideoUsage(seconds=5.0, videos=1, resolution="1080p"))

    assert cost.microusd is None
    assert cost.measurement is CostMeasurement.UNAVAILABLE


def test_a_rate_that_prices_nothing_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="per-second price"):
        VideoRate()


def test_resolution_rates_are_non_negative_and_immutable() -> None:
    source = {"480p": Decimal("0.04")}
    rate = VideoRate(usd_per_second_by_resolution=source)

    source["480p"] = Decimal("99")
    assert rate.for_resolution("480p") == Decimal("0.04")
    with pytest.raises(TypeError):
        rate.usd_per_second_by_resolution["480p"] = Decimal("0.08")  # type: ignore[index]

    with pytest.raises(ValueError, match="negative"):
        VideoRate(usd_per_second_by_resolution={"480p": Decimal("-0.01")})


def test_merging_different_resolutions_does_not_claim_either_one() -> None:
    merged = VideoUsage(seconds=5, resolution="480p").merge(
        VideoUsage(seconds=5, resolution="768p")
    )

    assert merged.seconds == 10
    assert merged.resolution is None
    assert merged.complete is False


def test_merging_an_unknown_usage_keeps_the_resolution_the_other_side_knows() -> None:
    """A failed attempt that reported nothing says nothing about the clip's tier."""
    merged = VideoUsage.unknown().merge(VideoUsage(seconds=5, videos=1, resolution="480p"))

    assert merged.resolution == "480p"
    assert merged.seconds == 5
    assert merged.complete is False


async def test_a_generated_video_is_returned_with_its_own_cost() -> None:
    sink = VideoSink()
    adapter = RecordingVideoAdapter("wavespeed", _response())

    result = await _gateway(adapter, sink=sink).generate_video(
        _request(image=ImageInput(url="https://cdn.test/lion.png"), resolution="480p")
    )

    assert result.videos[0].url == "https://cdn.test/hunt.mp4"
    assert result.usage.seconds == 5.0
    assert result.cost.amount_usd == Decimal("0.200000")
    assert sink.records[0].usage.resolution == "480p"
    assert sink.records[0].succeeded is True


async def test_an_estimated_duration_never_reports_an_actual_amount() -> None:
    adapter = RecordingVideoAdapter(
        "wavespeed",
        ProviderVideoResponse(
            videos=(GeneratedVideo(url="https://cdn.test/hunt.mp4"),),
            usage=VideoUsage(seconds=5.0, videos=1, resolution="480p", partial_aggregate=True),
        ),
    )

    result = await _gateway(adapter).generate_video(_request())

    assert result.cost.measurement is CostMeasurement.ESTIMATED


async def test_every_attempt_failing_raises_instead_of_returning_no_video() -> None:
    sink = VideoSink()
    adapter = RecordingVideoAdapter("wavespeed", RateLimitedError("429"))

    with pytest.raises(AllVideosFailed) as error:
        await _gateway(adapter, sink=sink).generate_video(_request())

    assert error.value.last_error == "RateLimitedError"
    assert sink.records[0].succeeded is False


async def test_a_provider_returning_no_video_is_a_failure_not_an_empty_result() -> None:
    adapter = RecordingVideoAdapter(
        "wavespeed", ProviderVideoResponse(videos=(), usage=VideoUsage(videos=0))
    )

    with pytest.raises(AllVideosFailed):
        await _gateway(adapter).generate_video(_request())


async def test_an_image_model_cannot_be_sent_through_the_video_path() -> None:
    adapter = RecordingVideoAdapter("wavespeed", _response())

    with pytest.raises(ConfigurationError, match="generate"):
        await _gateway(adapter).generate_video(_request(model="wavespeed-ai/hidream-i1-dev"))


async def test_an_injected_catalogue_overrides_the_built_in_rates() -> None:
    adapter = RecordingVideoAdapter("wavespeed", _response())
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=())
    gateway = LLMGateway(
        registry=registry,
        video_price_catalog=StaticVideoPriceCatalog(
            version="negotiated-2026-08",
            rates={MODEL: VideoRate(usd_per_second=Decimal("0.01"))},
        ),
    )

    result = await gateway.generate_video(_request())

    assert result.cost.amount_usd == Decimal("0.050000")
    assert result.cost.pricing_version == "negotiated-2026-08"


async def test_an_empty_reply_keeps_the_usage_the_provider_reported() -> None:
    """The provider rendered five seconds and invoiced them; an empty list refunds nothing."""
    adapter = RecordingVideoAdapter(
        "wavespeed",
        ProviderVideoResponse(
            videos=(), usage=VideoUsage(seconds=5.0, videos=0, resolution="480p")
        ),
    )

    with pytest.raises(AllVideosFailed) as raised:
        await _gateway(adapter).generate_video(_request())

    attempt = raised.value.attempts[0]
    assert attempt.usage.seconds == 5.0
    assert attempt.cost.amount_usd == Decimal("0.200000")
    assert attempt.billable is True


async def test_a_video_fallback_alert_says_why_the_requested_model_was_left() -> None:
    alerts = AlertSink()
    wavespeed = RecordingVideoAdapter("wavespeed", RateLimitedError("429"))
    replicate = RecordingVideoAdapter("replicate", _response())

    await _gateway(wavespeed, replicate, alerts=alerts).generate_video(
        _request(fallback_policy=FallbackPolicy.models_in_order("bytedance/seedance-2.5"))
    )

    name, fields = alerts.alerts[0]
    assert name == "llm_video_fallback_used"
    assert fields["error_type"] == "RateLimitedError"
    assert fields["failure_phase"] == "provider"
    assert len(fields["failures"]) == 1  # type: ignore[arg-type]


async def test_a_call_cut_off_by_its_total_budget_reports_how_long_it_ran() -> None:
    """Zero latency on a call that spent its whole budget reads as an instant failure."""

    class SlowAdapter(RecordingVideoAdapter):
        async def generate_video(
            self, request: VideoRequest, *, model: str
        ) -> ProviderVideoResponse:
            await asyncio.sleep(1)
            raise AssertionError("the total budget should interrupt this call")

    sink = VideoSink()

    with pytest.raises(AllVideosFailed):
        await _gateway(SlowAdapter("wavespeed"), sink=sink).generate_video(
            _request(timeout_policy=TimeoutPolicy(total_seconds=0.02))
        )

    assert sink.records[0].latency_ms >= 15


class TimingOutVideoSink(VideoSink):
    """The application's own ledger timing out, not the call's budget."""

    def record(self, record: VideoUsageRecord) -> None:
        super().record(record)
        raise TimeoutError("the usage database did not answer")


async def test_a_sink_timeout_is_not_mistaken_for_the_video_call_exceeding_its_budget() -> None:
    """Reporting the paid clip again as a failure would book it twice."""
    sink = TimingOutVideoSink()
    adapter = RecordingVideoAdapter("wavespeed", _response())

    with pytest.raises(TimeoutError, match="usage database"):
        await _gateway(adapter, sink=sink).generate_video(_request())

    assert [record.succeeded for record in sink.records] == [True]

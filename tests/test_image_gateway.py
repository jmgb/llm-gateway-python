"""Image orchestration: the same retry, fallback and accounting as every other call."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from llm_gateway import (
    AllImagesFailed,
    ConfigurationError,
    CostMeasurement,
    FallbackPolicy,
    GeneratedImage,
    ImageRate,
    ImageRequest,
    ImageUsage,
    ImageUsageRecord,
    LLMGateway,
    LLMRequest,
    ProviderImageResponse,
    ProviderRegistry,
    ProviderResponse,
    RateLimitedError,
    RetryPolicy,
    StaticImagePriceCatalog,
    TimeoutPolicy,
    TokenUsage,
)


class RecordingImageAdapter:
    def __init__(self, name: str, *responses: ProviderImageResponse | Exception) -> None:
        self.name = name
        self._responses = list(responses)
        self.requests: list[tuple[ImageRequest, str]] = []

    async def generate_image(self, request: ImageRequest, *, model: str) -> ProviderImageResponse:
        self.requests.append((request, model))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise AssertionError("image test adapter is not used for text generation")


class TextOnlyAdapter:
    name = "wavespeed"

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise AssertionError("not reached")


class ImageSink:
    def __init__(self) -> None:
        self.records: list[ImageUsageRecord] = []

    def record(self, record: ImageUsageRecord) -> None:
        self.records.append(record)


def _response(url: str = "https://cdn.test/one.png") -> ProviderImageResponse:
    return ProviderImageResponse(
        images=(GeneratedImage(url=url, mime_type="image/png"),),
        usage=ImageUsage(images=1),
    )


def _request(**kwargs: object) -> ImageRequest:
    return ImageRequest(
        model=str(kwargs.pop("model", "black-forest-labs/flux-kontext-pro")),
        prompt=str(kwargs.pop("prompt", "a cat wearing a hat")),
        **kwargs,  # type: ignore[arg-type]
    )


class AlertSink:
    def __init__(self) -> None:
        self.alerts: list[tuple[str, dict[str, object]]] = []

    def alert(self, name: str, fields: dict[str, object]) -> None:
        self.alerts.append((name, fields))


def _gateway(
    *adapters: object, sink: ImageSink | None = None, alerts: AlertSink | None = None
) -> LLMGateway:
    registry = ProviderRegistry()
    for adapter in adapters:
        registry.register(adapter, model_prefixes=())  # type: ignore[arg-type]
    return LLMGateway(
        registry=registry,
        image_price_catalog=StaticImagePriceCatalog(
            version="image-test",
            rates={
                "black-forest-labs/flux-kontext-pro": ImageRate(usd_per_image=Decimal("0.04")),
                "wavespeed-ai/hidream-i1-dev": ImageRate(usd_per_image=Decimal("0.012")),
            },
        ),
        image_usage_sink=sink,
        alert_sink=alerts,
    )


async def test_a_generated_image_is_returned_with_its_own_cost() -> None:
    sink = ImageSink()
    adapter = RecordingImageAdapter("replicate", _response())

    result = await _gateway(adapter, sink=sink).generate_image(_request())

    assert result.images[0].url == "https://cdn.test/one.png"
    assert result.usage.images == 1
    assert result.cost.amount_usd == Decimal("0.040000")
    assert result.cost.measurement is CostMeasurement.ACTUAL
    assert sink.records[0].usage.images == 1
    assert sink.records[0].provider == "replicate"


async def test_image_generation_falls_back_across_providers_and_says_so() -> None:
    replicate = RecordingImageAdapter("replicate", RateLimitedError("429"))
    wavespeed = RecordingImageAdapter("wavespeed", _response("https://cdn.test/backup.png"))

    result = await _gateway(replicate, wavespeed).generate_image(
        _request(fallback_policy=FallbackPolicy.models_in_order("wavespeed-ai/hidream-i1-dev"))
    )

    assert result.execution.fallback_used is True
    assert result.execution.model_used == "wavespeed-ai/hidream-i1-dev"
    assert result.execution.attempt_count == 2
    assert result.cost.amount_usd == Decimal("0.012000")


async def test_a_failed_attempt_stays_visible_in_the_execution() -> None:
    adapter = RecordingImageAdapter("replicate", RateLimitedError("429"), _response())

    result = await _gateway(adapter).generate_image(
        _request(retry_policy=RetryPolicy.transient(max_attempts=2, base_delay_seconds=0.0))
    )

    assert result.execution.attempt_count == 2
    assert result.execution.attempts[0].outcome.value == "failed"
    assert result.execution.attempts[0].error_type == "RateLimitedError"


async def test_every_attempt_failing_raises_instead_of_returning_no_image() -> None:
    adapter = RecordingImageAdapter("replicate", RateLimitedError("429"))
    sink = ImageSink()

    with pytest.raises(AllImagesFailed) as error:
        await _gateway(adapter, sink=sink).generate_image(_request())

    assert error.value.last_error == "RateLimitedError"
    assert sink.records[0].succeeded is False


async def test_a_provider_returning_no_image_is_a_failure_not_an_empty_result() -> None:
    adapter = RecordingImageAdapter(
        "replicate", ProviderImageResponse(images=(), usage=ImageUsage(images=0))
    )

    with pytest.raises(AllImagesFailed):
        await _gateway(adapter).generate_image(_request())


async def test_a_text_model_cannot_be_sent_through_the_image_path() -> None:
    adapter = RecordingImageAdapter("openai", _response())

    with pytest.raises(ConfigurationError, match="generate"):
        await _gateway(adapter).generate_image(_request(model="gpt-6-luna"))


async def test_a_provider_without_image_support_fails_configuration_not_silently() -> None:
    with pytest.raises(AllImagesFailed):
        await _gateway(TextOnlyAdapter()).generate_image(
            _request(model="wavespeed-ai/hidream-i1-dev")
        )


async def test_sinks_receive_metadata_only_never_the_prompt_or_the_image() -> None:
    sink = ImageSink()
    adapter = RecordingImageAdapter(
        "gemini",
        ProviderImageResponse(
            images=(GeneratedImage(data=b"\x89PNG"),),
            usage=ImageUsage(images=1, tokens=TokenUsage(input_tokens=8, output_tokens=1200)),
        ),
    )

    await _gateway(adapter, sink=sink).generate_image(
        _request(model="gemini-3.1-flash-image", request_id="req-1", source="whatsapp")
    )

    record = sink.records[0]
    fields = (
        vars(record)
        if hasattr(record, "__dict__")
        else {name: getattr(record, name) for name in record.__slots__}
    )
    assert "a cat wearing a hat" not in repr(fields)
    assert record.request_id == "req-1"
    assert record.source == "whatsapp"


async def test_an_image_fallback_alert_says_why_the_requested_model_was_left() -> None:
    alerts = AlertSink()
    replicate = RecordingImageAdapter("replicate", RateLimitedError("429"))
    wavespeed = RecordingImageAdapter("wavespeed", _response("https://cdn.test/backup.png"))

    await _gateway(replicate, wavespeed, alerts=alerts).generate_image(
        _request(fallback_policy=FallbackPolicy.models_in_order("wavespeed-ai/hidream-i1-dev"))
    )

    name, fields = alerts.alerts[0]
    assert name == "llm_image_fallback_used"
    assert fields["error_type"] == "RateLimitedError"
    assert fields["error_message"] == "429"
    assert fields["failure_phase"] == "provider"
    assert len(fields["failures"]) == 1  # type: ignore[arg-type]


async def test_an_empty_reply_keeps_the_usage_the_provider_reported() -> None:
    """Gemini bills the tokens of a reply that carried no picture all the same."""
    adapter = RecordingImageAdapter(
        "gemini",
        ProviderImageResponse(
            images=(),
            usage=ImageUsage(images=0, tokens=TokenUsage(input_tokens=8, output_tokens=12)),
        ),
    )

    with pytest.raises(AllImagesFailed) as raised:
        await _gateway(adapter).generate_image(_request(model="gemini-3.1-flash-image"))

    attempt = raised.value.attempts[0]
    assert attempt.usage.tokens == TokenUsage(input_tokens=8, output_tokens=12)
    assert attempt.billable is True


async def test_a_call_cut_off_by_its_total_budget_reports_how_long_it_ran() -> None:
    """Zero latency on a call that spent its whole budget reads as an instant failure."""

    class SlowAdapter(RecordingImageAdapter):
        async def generate_image(
            self, request: ImageRequest, *, model: str
        ) -> ProviderImageResponse:
            await asyncio.sleep(1)
            raise AssertionError("the total budget should interrupt this call")

    sink = ImageSink()

    with pytest.raises(AllImagesFailed):
        await _gateway(SlowAdapter("replicate"), sink=sink).generate_image(
            _request(timeout_policy=TimeoutPolicy(total_seconds=0.02))
        )

    assert sink.records[0].latency_ms >= 15


class TimingOutImageSink(ImageSink):
    """The application's own ledger timing out, not the call's budget."""

    def record(self, record: ImageUsageRecord) -> None:
        super().record(record)
        raise TimeoutError("the usage database did not answer")


async def test_a_sink_timeout_is_not_mistaken_for_the_image_call_exceeding_its_budget() -> None:
    """Reporting the paid image again as a failure would book it twice."""
    sink = TimingOutImageSink()
    adapter = RecordingImageAdapter("replicate", _response())

    with pytest.raises(TimeoutError, match="usage database"):
        await _gateway(adapter, sink=sink).generate_image(_request())

    assert [record.succeeded for record in sink.records] == [True]

"""Audio orchestration, fallback and accounting stay outside token calls."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from llm_gateway import (
    AllTranscriptionsFailed,
    AudioInput,
    AudioRate,
    AudioUsage,
    AudioUsageRecord,
    ConfigurationError,
    FailurePhase,
    FallbackPolicy,
    LLMGateway,
    LLMRequest,
    ProviderRegistry,
    ProviderResponse,
    ProviderTranscriptionResponse,
    RateLimitedError,
    StaticAudioPriceCatalog,
    TimeoutPolicy,
    TokenUsage,
    TranscriptionRequest,
)


class RecordingAudioAdapter:
    def __init__(self, name: str, *responses: ProviderTranscriptionResponse | Exception) -> None:
        self.name = name
        self._responses = list(responses)
        self.requests: list[TranscriptionRequest] = []

    async def transcribe(
        self, request: TranscriptionRequest, *, model: str
    ) -> ProviderTranscriptionResponse:
        self.requests.append(request)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise AssertionError("audio test adapter is not used for text generation")


class AudioSink:
    def __init__(self) -> None:
        self.records: list[AudioUsageRecord] = []

    def record(self, record: AudioUsageRecord) -> None:
        self.records.append(record)


def _response(text: str, duration: float | None) -> ProviderTranscriptionResponse:
    return ProviderTranscriptionResponse(
        text=text,
        usage=AudioUsage(duration_seconds=duration),
        model_used=None,
    )


def _request(**kwargs: object) -> TranscriptionRequest:
    return TranscriptionRequest(
        model=str(kwargs.pop("model", "gpt-transcribe")),
        audio=AudioInput(data=b"audio", duration_seconds=60.0),
        fallback_policy=kwargs.pop("fallback_policy", FallbackPolicy.disabled()),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


class AlertSink:
    def __init__(self) -> None:
        self.alerts: list[tuple[str, dict[str, object]]] = []

    def alert(self, name: str, fields: dict[str, object]) -> None:
        self.alerts.append((name, fields))


def _gateway(
    adapter: RecordingAudioAdapter,
    sink: AudioSink | None = None,
    *,
    also: tuple[RecordingAudioAdapter, ...] = (),
    alerts: AlertSink | None = None,
) -> LLMGateway:
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=("gpt-", "whisper-", "assemblyai-"))
    # Catalogued models route by their declared provider, so no prefix needed.
    for other in also:
        registry.register(other, model_prefixes=())
    return LLMGateway(
        registry=registry,
        audio_price_catalog=StaticAudioPriceCatalog(
            version="audio-test",
            rates={
                "gpt-transcribe": AudioRate(Decimal("0.0045")),
                "whisper-large-v3-turbo": AudioRate(
                    Decimal("0.0006666666666666666666666667"), minimum_billable_seconds=10
                ),
            },
        ),
        audio_usage_sink=sink,
        alert_sink=alerts,
    )


async def test_transcription_cost_is_audio_cost_not_token_cost() -> None:
    sink = AudioSink()
    adapter = RecordingAudioAdapter("openai", _response("hola", 60.0))

    result = await _gateway(adapter, sink).transcribe(_request())

    assert result.text == "hola"
    assert result.usage.duration_seconds == 60.0
    assert result.cost.amount_usd == Decimal("0.004500")
    assert result.cost.pricing_version == "audio-test"
    assert not isinstance(result.execution.attempts[0].usage, TokenUsage)
    assert sink.records[0].usage.duration_seconds == 60.0


async def test_transcription_can_fallback_from_openai_to_groq() -> None:
    openai = RecordingAudioAdapter("openai", RateLimitedError("429"))
    groq = RecordingAudioAdapter("groq", _response("from groq", 60.0))
    result = await _gateway(openai, also=(groq,)).transcribe(
        _request(fallback_policy=FallbackPolicy.models_in_order("whisper-large-v3-turbo"))
    )

    assert result.text == "from groq"
    assert result.execution.fallback_used is True
    assert result.execution.model_used == "whisper-large-v3-turbo"
    assert [a.provider for a in result.execution.attempts] == ["openai", "groq"]


async def test_unknown_duration_is_unavailable_not_zero() -> None:
    adapter = RecordingAudioAdapter("openai", _response("hola", None))

    result = await _gateway(adapter).transcribe(
        TranscriptionRequest(model="gpt-transcribe", audio=AudioInput(data=b"audio"))
    )

    assert result.cost.amount_usd is None
    assert result.cost.measurement.value == "UNAVAILABLE"


async def test_a_token_model_cannot_enter_the_transcription_path() -> None:
    adapter = RecordingAudioAdapter("openai", _response("no", 60.0))

    with pytest.raises(ConfigurationError, match="token-priced"):
        await _gateway(adapter).transcribe(
            TranscriptionRequest(model="gpt-6-luna", audio=AudioInput(data=b"audio"))
        )

    assert adapter.requests == []


async def test_total_timeout_records_the_provider_attempt_it_interrupts() -> None:
    class SlowAdapter(RecordingAudioAdapter):
        async def transcribe(
            self, request: TranscriptionRequest, *, model: str
        ) -> ProviderTranscriptionResponse:
            self.requests.append(request)
            await asyncio.sleep(1)
            raise AssertionError("the total timeout should interrupt the provider call")

    adapter = SlowAdapter("openai")

    with pytest.raises(AllTranscriptionsFailed) as raised:
        await _gateway(adapter).transcribe(
            _request(
                timeout_policy=TimeoutPolicy(
                    total_seconds=0.02,
                    per_attempt_seconds_override=1,
                )
            )
        )

    assert len(raised.value.attempts) == 1
    attempt = raised.value.attempts[0]
    assert attempt.failure_phase is FailurePhase.TIMEOUT
    assert attempt.billable is True


async def test_a_token_model_in_the_fallback_plan_is_refused_before_anything_is_spent() -> None:
    """Found only when its turn came, it would fail after the first model was paid."""
    adapter = RecordingAudioAdapter("openai", _response("hola", 60.0))

    with pytest.raises(ConfigurationError, match="token-priced"):
        await _gateway(adapter).transcribe(
            _request(fallback_policy=FallbackPolicy.models_in_order("gpt-6-luna"))
        )

    assert adapter.requests == []


async def test_an_audio_fallback_alert_says_why_the_requested_model_was_left() -> None:
    alerts = AlertSink()
    openai = RecordingAudioAdapter("openai", RateLimitedError("429"))
    groq = RecordingAudioAdapter("groq", _response("from groq", 60.0))

    await _gateway(openai, also=(groq,), alerts=alerts).transcribe(
        _request(fallback_policy=FallbackPolicy.models_in_order("whisper-large-v3-turbo"))
    )

    name, fields = alerts.alerts[0]
    assert name == "llm_audio_fallback_used"
    assert fields["error_type"] == "RateLimitedError"
    assert fields["failure_phase"] == "provider"
    assert len(fields["failures"]) == 1  # type: ignore[arg-type]


async def test_a_call_cut_off_by_its_total_budget_reports_how_long_it_ran() -> None:
    """Zero latency on a call that spent its whole budget reads as an instant failure."""

    class SlowAdapter(RecordingAudioAdapter):
        async def transcribe(
            self, request: TranscriptionRequest, *, model: str
        ) -> ProviderTranscriptionResponse:
            await asyncio.sleep(1)
            raise AssertionError("the total budget should interrupt this call")

    sink = AudioSink()

    with pytest.raises(AllTranscriptionsFailed):
        await _gateway(SlowAdapter("openai"), sink).transcribe(
            _request(timeout_policy=TimeoutPolicy(total_seconds=0.02))
        )

    assert sink.records[0].latency_ms >= 15


class TimingOutAudioSink(AudioSink):
    """The application's own ledger timing out, not the call's budget."""

    def record(self, record: AudioUsageRecord) -> None:
        super().record(record)
        raise TimeoutError("the usage database did not answer")


async def test_a_sink_timeout_is_not_mistaken_for_the_transcription_exceeding_its_budget() -> None:
    """Reporting the paid transcription again as a failure would book it twice."""
    sink = TimingOutAudioSink()
    adapter = RecordingAudioAdapter("openai", _response("hello", 60.0))

    with pytest.raises(TimeoutError, match="usage database"):
        await _gateway(adapter, sink).transcribe(_request())

    assert [record.succeeded for record in sink.records] == [True]

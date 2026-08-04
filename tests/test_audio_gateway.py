"""Audio orchestration, fallback and accounting stay outside token calls."""

from __future__ import annotations

from decimal import Decimal

import pytest

from llm_gateway import (
    AllTranscriptionsFailed,
    AudioInput,
    AudioRate,
    AudioUsage,
    AudioUsageRecord,
    FallbackPolicy,
    LLMGateway,
    LLMRequest,
    ProviderRegistry,
    ProviderResponse,
    ProviderTranscriptionResponse,
    RateLimitedError,
    StaticAudioPriceCatalog,
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


def _gateway(adapter: RecordingAudioAdapter, sink: AudioSink | None = None) -> LLMGateway:
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=("gpt-", "whisper-", "assemblyai-"))
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
    adapter = RecordingAudioAdapter(
        "openai", RateLimitedError("429"), _response("desde groq", 60.0)
    )

    result = await _gateway(adapter).transcribe(
        _request(fallback_policy=FallbackPolicy.models_in_order("whisper-large-v3-turbo"))
    )

    assert result.text == "desde groq"
    assert result.execution.fallback_used is True
    assert result.execution.model_used == "whisper-large-v3-turbo"
    assert len(result.execution.attempts) == 2


async def test_unknown_duration_is_unavailable_not_zero() -> None:
    adapter = RecordingAudioAdapter("openai", _response("hola", None))

    result = await _gateway(adapter).transcribe(
        TranscriptionRequest(model="gpt-transcribe", audio=AudioInput(data=b"audio"))
    )

    assert result.cost.amount_usd is None
    assert result.cost.measurement.value == "UNAVAILABLE"


async def test_a_token_model_cannot_enter_the_transcription_path() -> None:
    adapter = RecordingAudioAdapter("openai", _response("no", 60.0))

    with pytest.raises(AllTranscriptionsFailed):
        await _gateway(adapter).transcribe(
            TranscriptionRequest(model="gpt-5.6-luna", audio=AudioInput(data=b"audio"))
        )

    assert adapter.requests == []

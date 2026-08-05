"""The audio contract keeps source, duration and pricing distinct from text."""

from __future__ import annotations

import pytest

from llm_gateway import AudioInput, AudioUsage, TranscriptionRequest


def test_audio_input_can_carry_bytes_and_duration_without_token_fields() -> None:
    audio = AudioInput(
        data=b"audio",
        filename="voice.webm",
        mime_type="audio/webm",
        duration_seconds=12.5,
    )

    request = TranscriptionRequest(model="gpt-transcribe", audio=audio, language="es")

    assert request.audio.data == b"audio"
    assert request.audio.duration_seconds == 12.5
    assert not hasattr(request, "messages")


def test_audio_input_can_carry_a_remote_url() -> None:
    audio = AudioInput(url="https://storage.example/voice.m4a")

    assert audio.url == "https://storage.example/voice.m4a"
    assert audio.data is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "data or url"),
        ({"data": b"", "url": None}, "non-empty"),
        ({"url": "  "}, "non-empty"),
        ({"data": b"audio", "duration_seconds": -1}, "non-negative"),
    ],
)
def test_audio_input_rejects_invalid_sources(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AudioInput(**kwargs)  # type: ignore[arg-type]


def test_audio_input_can_keep_bytes_and_url_for_cross_provider_fallbacks() -> None:
    audio = AudioInput(data=b"audio", url="https://storage.example/voice.m4a")

    assert audio.data == b"audio"
    assert audio.url == "https://storage.example/voice.m4a"


def test_transcription_does_not_assume_a_product_language() -> None:
    request = TranscriptionRequest(model="gpt-transcribe", audio=AudioInput(data=b"audio"))

    assert request.language is None


@pytest.mark.parametrize("duration", [-1.0, float("inf"), float("nan")])
def test_reported_audio_usage_must_be_non_negative_and_finite(duration: float) -> None:
    with pytest.raises(ValueError, match="non-negative and finite"):
        AudioUsage(duration_seconds=duration)

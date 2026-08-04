"""Provider-specific audio translations, with no network or SDK dependency."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm_gateway import AudioInput, TranscriptionRequest
from llm_gateway.errors import ConfigurationError, ProviderError, RateLimitedError
from llm_gateway.providers.assemblyai import AssemblyAIAdapter
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openai import OpenAIAdapter


class Recorder:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _request(**kwargs: Any) -> TranscriptionRequest:
    return TranscriptionRequest(
        model=kwargs.pop("model", "gpt-transcribe"),
        language=kwargs.pop("language", "es"),
        audio=kwargs.pop(
            "audio",
            AudioInput(
                data=b"audio-bytes",
                filename="voice.webm",
                mime_type="audio/webm",
                duration_seconds=12.5,
            ),
        ),
        **kwargs,
    )


class TestOpenAITranscription:
    async def test_it_sends_a_multipart_file_to_the_audio_endpoint(self) -> None:
        recorder = Recorder(SimpleNamespace(text="hola", duration=12.5, language="es"))
        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=recorder))
        )

        response = await OpenAIAdapter(client).transcribe(
            _request(prompt="context"), model="gpt-transcribe"
        )

        assert response.text == "hola"
        assert response.usage.duration_seconds == 12.5
        assert recorder.kwargs == {
            "model": "gpt-transcribe",
            "file": ("voice.webm", b"audio-bytes", "audio/webm"),
            "response_format": "json",
            "languages": ["es"],
            "prompt": "context",
        }

    async def test_it_reads_duration_usage_from_the_provider_response(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                text="hola",
                usage=SimpleNamespace(type="duration", seconds=9.5),
                languages=[SimpleNamespace(code="es")],
            )
        )
        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=recorder))
        )

        response = await OpenAIAdapter(client).transcribe(
            _request(audio=AudioInput(data=b"audio"), language=None), model="gpt-transcribe"
        )

        assert response.usage.duration_seconds == 9.5
        assert response.usage.complete is True
        assert response.language == "es"

    async def test_caller_duration_is_only_an_estimate_when_the_provider_omits_usage(self) -> None:
        recorder = Recorder(SimpleNamespace(text="hola"))
        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=recorder))
        )

        response = await OpenAIAdapter(client).transcribe(_request(), model="gpt-transcribe")

        assert response.usage.duration_seconds == 12.5
        assert response.usage.complete is False

    async def test_it_rejects_speaker_labels_instead_of_silently_dropping_them(self) -> None:
        recorder = Recorder(SimpleNamespace(text="hola"))
        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=recorder))
        )

        with pytest.raises(ConfigurationError, match="speaker labels"):
            await OpenAIAdapter(client).transcribe(
                _request(speaker_labels=True), model="gpt-transcribe"
            )


class TestGroqTranscription:
    async def test_it_preserves_the_mime_type_for_uploaded_bytes(self) -> None:
        recorder = Recorder(SimpleNamespace(text="texto", duration=8.0))
        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=recorder))
        )

        await GroqAdapter(client).transcribe(
            _request(model="whisper-large-v3-turbo"),
            model="whisper-large-v3-turbo",
        )

        assert recorder.kwargs["file"] == (
            "voice.webm",
            b"audio-bytes",
            "audio/webm",
        )

    async def test_it_uses_the_whisper_url_transport(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                text="texto",
                duration=8.0,
                language="es",
                segments=[SimpleNamespace(start=0.0, end=1.0, text="texto")],
            )
        )
        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=recorder))
        )

        response = await GroqAdapter(client).transcribe(
            _request(
                model="whisper-large-v3-turbo",
                audio=AudioInput(url="https://storage.example/voice.m4a"),
            ),
            model="whisper-large-v3-turbo",
        )

        assert response.text == "texto"
        assert response.segments[0].start_seconds == 0.0
        assert recorder.kwargs == {
            "url": "https://storage.example/voice.m4a",
            "model": "whisper-large-v3-turbo",
            "response_format": "verbose_json",
            "language": "es",
            "temperature": 0.0,
        }

    async def test_it_rejects_speaker_labels_instead_of_silently_dropping_them(self) -> None:
        recorder = Recorder(SimpleNamespace(text="texto", duration=8.0))
        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=recorder))
        )

        with pytest.raises(ConfigurationError, match="speaker labels"):
            await GroqAdapter(client).transcribe(
                _request(model="whisper-large-v3-turbo", speaker_labels=True),
                model="whisper-large-v3-turbo",
            )


class FakeAssemblyAIClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []

    async def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        self.posts.append((path, json))
        return {"id": "transcript-1", "status": "queued"}

    async def get(self, path: str) -> dict[str, Any]:
        self.gets.append(path)
        return {
            "id": "transcript-1",
            "status": "completed",
            "text": "acta transcrita",
            "audio_duration": 42.0,
            "language_code": "es",
            "utterances": [{"start": 0, "end": 1000, "text": "acta transcrita", "speaker": "A"}],
        }


class TestAssemblyAITranscription:
    async def test_it_submits_and_polls_a_public_audio_url(self) -> None:
        client = FakeAssemblyAIClient()
        response = await AssemblyAIAdapter(client, poll_interval_seconds=0).transcribe(
            _request(
                model="assemblyai-universal-3-pro",
                audio=AudioInput(url="https://storage.example/acta.m4a"),
                speaker_labels=True,
                prompt="Construction site meeting",
            ),
            model="assemblyai-universal-3-pro",
        )

        assert response.text == "acta transcrita"
        assert response.usage.duration_seconds == 42.0
        assert response.segments[0].speaker == "A"
        assert client.posts == [
            (
                "/transcript",
                {
                    "audio_url": "https://storage.example/acta.m4a",
                    "language_code": "es",
                    "prompt": "Construction site meeting",
                    "speaker_labels": True,
                    "speech_models": ["universal-3-pro"],
                },
            )
        ]
        assert client.gets == ["/transcript/transcript-1"]

    async def test_universal_2_rejects_a_prompt_it_cannot_honor(self) -> None:
        with pytest.raises(ConfigurationError, match="prompt"):
            await AssemblyAIAdapter(FakeAssemblyAIClient()).transcribe(
                _request(
                    model="assemblyai-universal-2",
                    audio=AudioInput(url="https://storage.example/acta.m4a"),
                    prompt="Context",
                ),
                model="assemblyai-universal-2",
            )

    async def test_http_failures_are_mapped_to_typed_provider_errors(self) -> None:
        class TooManyRequests(Exception):
            status_code = 429

        class FailingClient:
            async def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
                raise TooManyRequests

        with pytest.raises(RateLimitedError):
            await AssemblyAIAdapter(FailingClient()).transcribe(
                _request(
                    model="assemblyai-universal-2",
                    audio=AudioInput(url="https://storage.example/acta.m4a"),
                ),
                model="assemblyai-universal-2",
            )

    async def test_malformed_provider_responses_raise_a_typed_error(self) -> None:
        class MissingIdClient:
            async def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
                return {"status": "queued"}

        with pytest.raises(ProviderError):
            await AssemblyAIAdapter(MissingIdClient()).transcribe(
                _request(
                    model="assemblyai-universal-2",
                    audio=AudioInput(url="https://storage.example/acta.m4a"),
                ),
                model="assemblyai-universal-2",
            )

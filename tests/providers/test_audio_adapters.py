"""Provider-specific audio translations, with no network or SDK dependency."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from llm_gateway import AudioInput, TranscriptionRequest
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
            "file": ("voice.webm", b"audio-bytes"),
            "response_format": "json",
            "language": "es",
            "prompt": "context",
        }


class TestGroqTranscription:
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
                model="assemblyai-universal-3-5-pro",
                audio=AudioInput(url="https://storage.example/acta.m4a"),
                speaker_labels=True,
            ),
            model="assemblyai-universal-3-5-pro",
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
                    "speaker_labels": True,
                    "speech_models": ["universal-3-5-pro"],
                },
            )
        ]
        assert client.gets == ["/transcript/transcript-1"]

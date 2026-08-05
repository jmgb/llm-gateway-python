"""Provider-specific image and video translations, with no network and no SDK."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm_gateway import ImageInput, ImageRequest, VideoRequest
from llm_gateway.errors import ConfigurationError, ProviderError, RateLimitedError
from llm_gateway.providers.gemini import GeminiAdapter
from llm_gateway.providers.replicate import ReplicateAdapter
from llm_gateway.providers.wavespeed import WaveSpeedAdapter


class Recorder:
    def __init__(self, *responses: Any) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append({"args": args, **kwargs})
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(response, Exception):
            raise response
        return response


def _request(**kwargs: Any) -> ImageRequest:
    return ImageRequest(
        model=kwargs.pop("model", "gemini-3.1-flash-image"),
        prompt=kwargs.pop("prompt", "a cat wearing a hat"),
        **kwargs,
    )


def _inline_part(data: bytes, mime_type: str = "image/png") -> SimpleNamespace:
    return SimpleNamespace(inline_data=SimpleNamespace(data=data, mime_type=mime_type), text=None)


class TestGeminiImages:
    async def test_it_asks_for_an_image_modality_and_returns_the_bytes(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[_inline_part(b"\x89PNG")]),
                        finish_reason="STOP",
                    )
                ],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=8,
                    candidates_token_count=1290,
                    thoughts_token_count=None,
                    cached_content_token_count=None,
                ),
            )
        )
        client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=recorder))
        )

        response = await GeminiAdapter(client).generate_image(
            _request(), model="gemini-3.1-flash-image"
        )

        assert response.images[0].data == b"\x89PNG"
        assert response.images[0].mime_type == "image/png"
        assert response.usage.images == 1
        assert response.usage.tokens is not None
        assert response.usage.tokens.output_tokens == 1290
        assert recorder.calls[0]["config"]["response_modalities"] == ["IMAGE"]

    async def test_an_edit_sends_the_source_image_inline(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[_inline_part(b"edited")]),
                        finish_reason="STOP",
                    )
                ],
                usage_metadata=None,
            )
        )
        client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=recorder))
        )

        await GeminiAdapter(client).generate_image(
            _request(image=ImageInput(data=b"original", mime_type="image/jpeg")),
            model="gemini-3.1-flash-image",
        )

        parts = recorder.calls[0]["contents"][0]["parts"]
        assert parts[0]["text"] == "a cat wearing a hat"
        assert parts[1]["inline_data"] == {"mime_type": "image/jpeg", "data": b"original"}

    async def test_it_refuses_a_url_it_would_have_to_download_itself(self) -> None:
        client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=Recorder()))
        )

        with pytest.raises(ConfigurationError, match="bytes"):
            await GeminiAdapter(client).generate_image(
                _request(image=ImageInput(url="https://cdn.test/cat.png")),
                model="gemini-3.1-flash-image",
            )

    async def test_a_reply_with_only_text_is_an_error_not_an_empty_result(self) -> None:
        """Gemini answers a description instead of an image often enough to matter."""
        recorder = Recorder(
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(inline_data=None, text="A cat, described.")]
                        ),
                        finish_reason="STOP",
                    )
                ],
                usage_metadata=None,
            )
        )
        client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=recorder))
        )

        with pytest.raises(ProviderError, match="no image"):
            await GeminiAdapter(client).generate_image(_request(), model="gemini-3.1-flash-image")

    async def test_a_blocked_generation_names_the_finish_reason(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                candidates=[SimpleNamespace(content=None, finish_reason="IMAGE_SAFETY")],
                usage_metadata=None,
            )
        )
        client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=recorder))
        )

        with pytest.raises(ProviderError, match="IMAGE_SAFETY"):
            await GeminiAdapter(client).generate_image(_request(), model="gemini-3.1-flash-image")


class TestReplicateImages:
    async def test_it_runs_the_model_and_reads_the_output_url(self) -> None:
        recorder = Recorder(SimpleNamespace(url="https://cdn.test/out.png"))
        client = SimpleNamespace(async_run=recorder)

        response = await ReplicateAdapter(client).generate_image(
            _request(model="prunaai/p-image"), model="prunaai/p-image"
        )

        assert response.images[0].url == "https://cdn.test/out.png"
        assert response.usage.images == 1
        assert recorder.calls[0]["args"] == ("prunaai/p-image",)
        assert recorder.calls[0]["input"] == {"prompt": "a cat wearing a hat"}

    async def test_an_edit_sends_the_source_url_and_the_aspect_ratio(self) -> None:
        recorder = Recorder(["https://cdn.test/edited.png"])
        client = SimpleNamespace(async_run=recorder)

        response = await ReplicateAdapter(client).generate_image(
            _request(
                model="black-forest-labs/flux-kontext-pro",
                image=ImageInput(url="https://cdn.test/in.png"),
                aspect_ratio="16:9",
            ),
            model="black-forest-labs/flux-kontext-pro",
        )

        assert response.images[0].url == "https://cdn.test/edited.png"
        assert recorder.calls[0]["input"] == {
            "prompt": "a cat wearing a hat",
            "input_image": "https://cdn.test/in.png",
            "aspect_ratio": "16:9",
        }

    async def test_it_refuses_raw_bytes_it_would_have_to_host(self) -> None:
        client = SimpleNamespace(async_run=Recorder())

        with pytest.raises(ConfigurationError, match="URL"):
            await ReplicateAdapter(client).generate_image(
                _request(model="prunaai/p-image", image=ImageInput(data=b"raw")),
                model="prunaai/p-image",
            )

    async def test_an_empty_output_is_a_provider_error(self) -> None:
        client = SimpleNamespace(async_run=Recorder([]))

        with pytest.raises(ProviderError, match="no image"):
            await ReplicateAdapter(client).generate_image(
                _request(model="prunaai/p-image"), model="prunaai/p-image"
            )

    async def test_provider_failures_are_classified_structurally(self) -> None:
        failure = Exception("rate limited")
        failure.status_code = 429  # type: ignore[attr-defined]
        client = SimpleNamespace(async_run=Recorder(failure))

        with pytest.raises(RateLimitedError):
            await ReplicateAdapter(client).generate_image(
                _request(model="prunaai/p-image"), model="prunaai/p-image"
            )


class FakeWaveSpeedClient:
    def __init__(self, *results: dict[str, Any]) -> None:
        self._results = list(results)
        self.posted: list[tuple[str, dict[str, Any]]] = []
        self.got: list[str] = []

    async def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        self.posted.append((path, json))
        return {"data": {"id": "task-1"}}

    async def get(self, path: str) -> dict[str, Any]:
        self.got.append(path)
        return {"data": self._results.pop(0)}


class TestWaveSpeedImages:
    async def test_it_submits_a_task_and_polls_until_the_url_is_ready(self) -> None:
        client = FakeWaveSpeedClient(
            {"status": "processing"},
            {"status": "completed", "outputs": ["https://cdn.test/hidream.png"]},
        )

        response = await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_image(
            _request(model="wavespeed-ai/hidream-i1-dev"),
            model="wavespeed-ai/hidream-i1-dev",
        )

        assert response.images[0].url == "https://cdn.test/hidream.png"
        assert response.usage.images == 1
        assert client.posted[0][0] == "/api/v3/wavespeed-ai/hidream-i1-dev"
        assert client.posted[0][1] == {"prompt": "a cat wearing a hat"}
        assert client.got == [
            "/api/v3/predictions/task-1/result",
            "/api/v3/predictions/task-1/result",
        ]

    async def test_a_non_success_api_code_never_becomes_a_prediction(self) -> None:
        class RejectedClient(FakeWaveSpeedClient):
            async def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
                self.posted.append((path, json))
                return {"code": 400, "message": "invalid", "data": {"id": "charged-task"}}

        client = RejectedClient({"status": "completed", "outputs": ["unexpected"]})

        with pytest.raises(ProviderError, match="code 400"):
            await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_image(
                _request(model="wavespeed-ai/hidream-i1-dev"),
                model="wavespeed-ai/hidream-i1-dev",
            )

        assert client.got == []

    async def test_a_failed_task_raises_instead_of_returning_nothing(self) -> None:
        client = FakeWaveSpeedClient({"status": "failed", "error": "nope"})

        with pytest.raises(ProviderError):
            await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_image(
                _request(model="wavespeed-ai/hidream-i1-dev"),
                model="wavespeed-ai/hidream-i1-dev",
            )

    async def test_unsupported_options_raise_rather_than_being_dropped(self) -> None:
        client = FakeWaveSpeedClient({"status": "completed", "outputs": ["x"]})
        adapter = WaveSpeedAdapter(client, poll_interval_seconds=0.0)

        with pytest.raises(ConfigurationError, match="aspect ratio"):
            await adapter.generate_image(
                _request(model="wavespeed-ai/hidream-i1-dev", aspect_ratio="16:9"),
                model="wavespeed-ai/hidream-i1-dev",
            )
        with pytest.raises(ConfigurationError, match="editing"):
            await adapter.generate_image(
                _request(
                    model="wavespeed-ai/hidream-i1-dev",
                    image=ImageInput(url="https://cdn.test/in.png"),
                ),
                model="wavespeed-ai/hidream-i1-dev",
            )


class FakeWaveSpeedVideoClient:
    def __init__(self, *results: dict[str, Any]) -> None:
        self._results = list(results)
        self.posted: list[tuple[str, dict[str, Any]]] = []
        self.got: list[str] = []

    async def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        self.posted.append((path, json))
        return {"data": {"id": "video-1"}}

    async def get(self, path: str) -> dict[str, Any]:
        self.got.append(path)
        return {"data": self._results.pop(0)}


def _video_request(**kwargs: Any) -> VideoRequest:
    return VideoRequest(
        model=kwargs.pop("model", "wavespeed-ai/minimax-h3/image-to-video"),
        prompt=kwargs.pop("prompt", "the lion charges and leaps at the gazelle"),
        **kwargs,
    )


class TestWaveSpeedVideo:
    async def test_an_image_to_video_model_requires_a_first_frame(self) -> None:
        client = FakeWaveSpeedVideoClient(
            {"status": "completed", "outputs": ["https://cdn.test/hunt.mp4"]}
        )

        with pytest.raises(ConfigurationError, match="first frame"):
            await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_video(
                _video_request(), model="wavespeed-ai/minimax-h3/image-to-video"
            )

        assert client.posted == []

    async def test_it_sends_the_frame_the_prompt_and_the_clip_settings(self) -> None:
        client = FakeWaveSpeedVideoClient(
            {"status": "processing"},
            {"status": "completed", "outputs": ["https://cdn.test/hunt.mp4"]},
        )

        response = await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video_request(
                image=ImageInput(url="https://cdn.test/lion.png"),
                resolution="480p",
                duration_seconds=5,
            ),
            model="wavespeed-ai/minimax-h3/image-to-video",
        )

        assert response.videos[0].url == "https://cdn.test/hunt.mp4"
        assert response.videos[0].mime_type == "video/mp4"
        assert response.usage.seconds == 5.0
        assert response.usage.resolution == "480p"
        assert client.posted[0][0] == "/api/v3/wavespeed-ai/minimax-h3/image-to-video"
        assert client.posted[0][1] == {
            "prompt": "the lion charges and leaps at the gazelle",
            "image": "https://cdn.test/lion.png",
            "resolution": "480p",
            "duration": 5,
        }

    async def test_a_first_frame_in_bytes_travels_as_a_data_uri(self) -> None:
        """This is what lets one provider's image be animated by another."""
        client = FakeWaveSpeedVideoClient({"status": "completed", "outputs": ["https://x/y.mp4"]})

        await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video_request(image=ImageInput(data=b"\x89PNG", mime_type="image/png")),
            model="wavespeed-ai/minimax-h3/image-to-video",
        )

        assert client.posted[0][1]["image"] == "data:image/png;base64,iVBORw=="

    async def test_a_requested_duration_is_an_estimate_not_a_measurement(self) -> None:
        """WaveSpeed reports no clip length and snaps output to its frame grid."""
        client = FakeWaveSpeedVideoClient({"status": "completed", "outputs": ["https://x/y.mp4"]})

        response = await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video_request(
                duration_seconds=5,
                image=ImageInput(url="https://cdn.test/lion.png"),
            ),
            model="wavespeed-ai/minimax-h3/image-to-video",
        )

        assert response.usage.complete is False

    async def test_a_failed_task_raises_instead_of_returning_nothing(self) -> None:
        client = FakeWaveSpeedVideoClient({"status": "failed", "error": "nope"})

        with pytest.raises(ProviderError):
            await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_video(
                _video_request(image=ImageInput(url="https://cdn.test/lion.png")),
                model="wavespeed-ai/minimax-h3/image-to-video",
            )

    @pytest.mark.parametrize("status", ["cancelled", "timeout"])
    async def test_every_terminal_task_status_stops_polling(self, status: str) -> None:
        client = FakeWaveSpeedVideoClient({"status": status, "error": "stopped"})

        with pytest.raises(ProviderError, match=status):
            await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_video(
                _video_request(image=ImageInput(url="https://cdn.test/lion.png")),
                model="wavespeed-ai/minimax-h3/image-to-video",
            )

        assert client.got == ["/api/v3/predictions/video-1/result"]

    async def test_a_non_success_poll_code_stops_instead_of_timing_out(self) -> None:
        class RejectedPollClient(FakeWaveSpeedVideoClient):
            async def get(self, path: str) -> dict[str, Any]:
                self.got.append(path)
                return {"code": 500, "message": "failed", "data": {"status": "processing"}}

        client = RejectedPollClient()

        with pytest.raises(ProviderError, match="code 500"):
            await WaveSpeedAdapter(client, max_poll_attempts=1).generate_video(
                _video_request(image=ImageInput(url="https://cdn.test/lion.png")),
                model="wavespeed-ai/minimax-h3/image-to-video",
            )

"""Provider-specific image and video translations, with no network and no SDK."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm_gateway import ImageInput, ImageRequest, VideoJob, VideoJobStatus, VideoRequest
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


WAN = "wan-video/wan-2.2-5b-fast"


class FakePredictions:
    """Replicate's prediction endpoints, without the SDK or the network.

    The synchronous ``create``/``get`` are here on purpose, raising rather than
    answering. The real SDK ships both halves, and awaiting the blocking one
    fails only at runtime — a fake that offered a single async ``create`` would
    let exactly that bug pass, and did once.
    """

    def __init__(self, *predictions: Any) -> None:
        self._predictions = list(predictions)
        self.created: list[dict[str, Any]] = []
        self.fetched: list[str] = []

    def create(self, **kwargs: Any) -> Any:
        raise AssertionError("the blocking create() must not be awaited; use async_create")

    def get(self, prediction_id: str) -> Any:
        raise AssertionError("the blocking get() must not be awaited; use async_get")

    async def async_create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return self._next()

    async def async_get(self, prediction_id: str) -> Any:
        self.fetched.append(prediction_id)
        return self._next()

    def _next(self) -> Any:
        prediction = (
            self._predictions.pop(0) if len(self._predictions) > 1 else self._predictions[0]
        )
        if isinstance(prediction, Exception):
            raise prediction
        return prediction


def _prediction(status: str = "starting", **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(
        id=kwargs.pop("id", "pred-1"),
        status=status,
        output=kwargs.pop("output", None),
        error=kwargs.pop("error", None),
        **kwargs,
    )


def _client(*predictions: Any) -> SimpleNamespace:
    return SimpleNamespace(predictions=FakePredictions(*predictions))


class TestReplicateVideoSubmission:
    async def test_a_text_to_video_submission_returns_a_job_not_a_clip(self) -> None:
        client = _client(_prediction("starting"))

        job = await ReplicateAdapter(client).submit_video(_video_request(model=WAN), model=WAN)

        assert job.id == "pred-1"
        assert job.provider == "replicate"
        assert job.model == WAN
        assert job.status is VideoJobStatus.QUEUED
        assert client.predictions.created[0]["model"] == WAN
        # Prompt plus the cheapest tier, which a silent request gets rather
        # than Wan's own 720p default.
        assert client.predictions.created[0]["input"] == {
            "prompt": "the lion charges and leaps at the gazelle",
            "resolution": "480p",
        }

    async def test_a_first_frame_travels_as_the_url_replicate_will_fetch(self) -> None:
        client = _client(_prediction("processing"))

        job = await ReplicateAdapter(client).submit_video(
            _video_request(
                model=WAN,
                image=ImageInput(url="https://cdn.test/lion.png"),
                resolution="480p",
            ),
            model=WAN,
        )

        assert job.status is VideoJobStatus.RUNNING
        assert client.predictions.created[0]["input"] == {
            "prompt": "the lion charges and leaps at the gazelle",
            "image": "https://cdn.test/lion.png",
            "resolution": "480p",
        }

    async def test_a_first_frame_in_bytes_travels_as_a_data_uri(self) -> None:
        """This is what lets a frame from one provider be animated by another.

        Replicate fetches a URL and equally accepts an inline data URI, so a
        caller holding freshly generated bytes does not have to host them
        somewhere public first just to animate them.
        """
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(model=WAN, image=ImageInput(data=b"\x89PNG", mime_type="image/png")),
            model=WAN,
        )

        assert client.predictions.created[0]["input"]["image"] == ("data:image/png;base64,iVBORw==")

    async def test_a_duration_in_seconds_is_refused_rather_than_guessed(self) -> None:
        """Wan takes a frame count and a frame rate; seconds are not either one."""
        client = _client(_prediction())

        with pytest.raises(ConfigurationError, match="duration"):
            await ReplicateAdapter(client).submit_video(
                _video_request(model=WAN, duration_seconds=5), model=WAN
            )

        assert client.predictions.created == []

    async def test_a_webhook_is_registered_with_the_submission(self) -> None:
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(model=WAN, webhook_url="https://app.test/hooks/video"), model=WAN
        )

        assert client.predictions.created[0]["webhook"] == "https://app.test/hooks/video"
        assert client.predictions.created[0]["webhook_events_filter"] == ["completed"]

    async def test_no_webhook_keys_are_sent_when_none_was_asked_for(self) -> None:
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(_video_request(model=WAN), model=WAN)

        assert "webhook" not in client.predictions.created[0]

    async def test_a_prediction_without_an_id_is_a_provider_error(self) -> None:
        """An id-less prediction cannot be polled, and may already be running."""
        client = _client(_prediction(id=""))

        with pytest.raises(ProviderError, match="prediction id"):
            await ReplicateAdapter(client).submit_video(_video_request(model=WAN), model=WAN)

    async def test_submission_failures_are_classified_structurally(self) -> None:
        failure = Exception("rate limited")
        failure.status_code = 429  # type: ignore[attr-defined]
        client = _client(failure)

        with pytest.raises(RateLimitedError):
            await ReplicateAdapter(client).submit_video(_video_request(model=WAN), model=WAN)


class TestReplicateVideoPolling:
    async def test_a_finished_prediction_yields_the_clip(self) -> None:
        client = _client(_prediction("succeeded", output="https://cdn.test/lion.mp4"))

        update = await ReplicateAdapter(client).poll_video(
            VideoJob(id="pred-1", model=WAN, provider="replicate")
        )

        assert update.status is VideoJobStatus.SUCCEEDED
        assert update.videos[0].url == "https://cdn.test/lion.mp4"
        assert update.videos[0].mime_type == "video/mp4"
        assert update.usage.videos == 1
        assert client.predictions.fetched == ["pred-1"]

    async def test_an_sdk_file_output_is_read_like_a_url(self) -> None:
        client = _client(_prediction("succeeded", output=SimpleNamespace(url="https://x/y.mp4")))

        update = await ReplicateAdapter(client).poll_video(
            VideoJob(id="pred-1", model=WAN, provider="replicate")
        )

        assert update.videos[0].url == "https://x/y.mp4"

    async def test_a_list_output_keeps_every_clip(self) -> None:
        client = _client(_prediction("succeeded", output=["https://x/a.mp4", "https://x/b.mp4"]))

        update = await ReplicateAdapter(client).poll_video(
            VideoJob(id="pred-1", model=WAN, provider="replicate")
        )

        assert len(update.videos) == 2
        assert update.usage.videos == 2

    @pytest.mark.parametrize(
        ("provider_status", "expected"),
        [
            ("starting", VideoJobStatus.QUEUED),
            ("processing", VideoJobStatus.RUNNING),
            ("succeeded", VideoJobStatus.SUCCEEDED),
            ("failed", VideoJobStatus.FAILED),
            ("canceled", VideoJobStatus.CANCELLED),
        ],
    )
    async def test_every_replicate_status_maps_to_a_neutral_one(
        self, provider_status: str, expected: VideoJobStatus
    ) -> None:
        client = _client(_prediction(provider_status, output="https://x/y.mp4"))

        update = await ReplicateAdapter(client).poll_video(
            VideoJob(id="pred-1", model=WAN, provider="replicate")
        )

        assert update.status is expected

    async def test_an_unknown_status_is_refused_rather_than_read_as_running(self) -> None:
        """Reading an unknown state as "still working" polls a dead job forever."""
        client = _client(_prediction("exploded"))

        with pytest.raises(ProviderError, match="exploded"):
            await ReplicateAdapter(client).poll_video(
                VideoJob(id="pred-1", model=WAN, provider="replicate")
            )

    async def test_a_failed_prediction_carries_the_provider_reason(self) -> None:
        client = _client(_prediction("failed", error="NSFW content detected"))

        update = await ReplicateAdapter(client).poll_video(
            VideoJob(id="pred-1", model=WAN, provider="replicate")
        )

        assert update.status is VideoJobStatus.FAILED
        assert update.error == "NSFW content detected"
        assert update.videos == ()

    async def test_polling_failures_are_classified_structurally(self) -> None:
        failure = Exception("rate limited")
        failure.status_code = 429  # type: ignore[attr-defined]
        client = _client(failure)

        with pytest.raises(RateLimitedError):
            await ReplicateAdapter(client).poll_video(
                VideoJob(id="pred-1", model=WAN, provider="replicate")
            )

    async def test_replicate_never_claims_a_clip_length_it_does_not_report(self) -> None:
        """Wan is billed by GPU time and reports no duration; zero would be a lie."""
        client = _client(_prediction("succeeded", output="https://x/y.mp4"))

        update = await ReplicateAdapter(client).poll_video(
            VideoJob(id="pred-1", model=WAN, provider="replicate")
        )

        assert update.usage.seconds is None


KLING = "kwaivgi/kling-v3-video"
SEEDANCE = "bytedance/seedance-2.0"


class TestReplicateVideoModelShapes:
    """Each model names its inputs differently, and the names are not guessable.

    Wan takes a first frame as ``image``, Kling as ``start_image``. Wan sizes a
    clip in frames while both of the others take seconds. Kling has no
    ``resolution`` field at all — it has ``mode``. Sending the wrong key is not
    an error at the provider: the option is dropped, the default is generated,
    and the caller is billed for a clip nobody asked for.
    """

    async def test_kling_takes_the_first_frame_as_start_image(self) -> None:
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(model=KLING, image=ImageInput(url="https://cdn.test/lion.png")),
            model=KLING,
        )

        payload = client.predictions.created[0]["input"]
        assert payload["start_image"] == "https://cdn.test/lion.png"
        assert "image" not in payload

    async def test_seedance_takes_the_first_frame_as_image(self) -> None:
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(model=SEEDANCE, image=ImageInput(url="https://cdn.test/lion.png")),
            model=SEEDANCE,
        )

        assert client.predictions.created[0]["input"]["image"] == "https://cdn.test/lion.png"

    @pytest.mark.parametrize("model", [KLING, SEEDANCE])
    async def test_a_duration_in_seconds_is_sent_where_the_model_takes_seconds(
        self, model: str
    ) -> None:
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(model=model, duration_seconds=10), model=model
        )

        assert client.predictions.created[0]["input"]["duration"] == 10

    async def test_kling_maps_a_resolution_onto_the_mode_it_actually_has(self) -> None:
        """Kling exposes no resolution field; 1080p is spelled ``mode="pro"``."""
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(model=KLING, resolution="1080p"), model=KLING
        )

        payload = client.predictions.created[0]["input"]
        assert payload["mode"] == "pro"
        assert "resolution" not in payload

    @pytest.mark.parametrize(
        ("resolution", "mode"), [("720p", "standard"), ("1080p", "pro"), ("4k", "4k")]
    )
    async def test_every_kling_mode_is_reachable_by_its_resolution(
        self, resolution: str, mode: str
    ) -> None:
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(model=KLING, resolution=resolution), model=KLING
        )

        assert client.predictions.created[0]["input"]["mode"] == mode

    async def test_seedance_sends_the_resolution_it_accepts_directly(self) -> None:
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(model=SEEDANCE, resolution="480p"), model=SEEDANCE
        )

        assert client.predictions.created[0]["input"]["resolution"] == "480p"

    @pytest.mark.parametrize("model", [KLING, SEEDANCE])
    async def test_a_resolution_the_model_does_not_have_is_refused(self, model: str) -> None:
        """Falling back to the default would bill 1080p work at the caller's 360p ask."""
        client = _client(_prediction())

        with pytest.raises(ConfigurationError, match="360p"):
            await ReplicateAdapter(client).submit_video(
                _video_request(model=model, resolution="360p"), model=model
            )

        assert client.predictions.created == []

    async def test_wan_still_refuses_a_duration_it_cannot_express(self) -> None:
        """Only Wan sizes a clip in frames, so only Wan refuses seconds."""
        client = _client(_prediction())

        with pytest.raises(ConfigurationError, match="duration"):
            await ReplicateAdapter(client).submit_video(
                _video_request(model=WAN, duration_seconds=5), model=WAN
            )

    async def test_an_uncatalogued_replicate_model_gets_the_conservative_shape(self) -> None:
        """Unknown means unverified, so only what every video model shares is sent."""
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(
                model="someone/unverified-video", image=ImageInput(url="https://x/a.png")
            ),
            model="someone/unverified-video",
        )

        assert client.predictions.created[0]["input"] == {
            "prompt": "the lion charges and leaps at the gazelle",
            "image": "https://x/a.png",
        }


class TestTheNewVideoModelsAreCatalogued:
    @pytest.mark.parametrize("model", [KLING, SEEDANCE])
    def test_they_are_video_models_priced_by_nothing_this_package_verified(
        self, model: str
    ) -> None:
        from llm_gateway import lookup_model

        info = lookup_model(model)

        assert info is not None
        assert info.provider == "replicate"
        assert info.modality == "video"
        assert info.video_rate is None


class TestTheDefaultResolutionIsTheCheapestEachModelOffers:
    """Saying nothing must not mean "whatever the provider charges most for".

    Every video model here defaults to something dearer than its floor: Wan and
    Seedance to 720p, Kling to `pro` (1080p), and WaveSpeed prices 768p at twice
    480p. A caller who omitted the option gets the cheapest tier the model has,
    and has to ask for anything above it.
    """

    @pytest.mark.parametrize(
        ("model", "field", "expected"),
        [
            (WAN, "resolution", "480p"),
            (KLING, "mode", "standard"),
            (SEEDANCE, "resolution", "480p"),
        ],
    )
    async def test_an_unstated_resolution_becomes_the_models_floor(
        self, model: str, field: str, expected: str
    ) -> None:
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(_video_request(model=model), model=model)

        assert client.predictions.created[0]["input"][field] == expected

    @pytest.mark.parametrize(
        ("model", "field", "asked", "expected"),
        [
            (WAN, "resolution", "720p", "720p"),
            (KLING, "mode", "4k", "4k"),
            (SEEDANCE, "resolution", "1080p", "1080p"),
        ],
    )
    async def test_a_stated_resolution_still_wins(
        self, model: str, field: str, asked: str, expected: str
    ) -> None:
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(model=model, resolution=asked), model=model
        )

        assert client.predictions.created[0]["input"][field] == expected

    async def test_wan_refuses_a_resolution_it_does_not_have(self) -> None:
        """Wan offers 480p and 720p only; 1080p would be a 422 from Replicate."""
        client = _client(_prediction())

        with pytest.raises(ConfigurationError, match="1080p"):
            await ReplicateAdapter(client).submit_video(
                _video_request(model=WAN, resolution="1080p"), model=WAN
            )

    async def test_an_unverified_model_gets_no_invented_default(self) -> None:
        """A floor nobody read from a schema is a guess, and a 422 waiting to happen."""
        client = _client(_prediction())

        await ReplicateAdapter(client).submit_video(
            _video_request(model="someone/unverified-video"), model="someone/unverified-video"
        )

        payload = client.predictions.created[0]["input"]
        assert "resolution" not in payload
        assert "mode" not in payload

    async def test_wavespeed_also_defaults_to_its_cheapest_tier(self) -> None:
        client = FakeWaveSpeedVideoClient({"status": "completed", "outputs": ["https://x/y.mp4"]})

        response = await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video_request(
                model="wavespeed-ai/minimax-h3/image-to-video",
                image=ImageInput(url="https://cdn.test/lion.png"),
            ),
            model="wavespeed-ai/minimax-h3/image-to-video",
        )

        assert client.posted[0][1]["resolution"] == "480p"
        # Reported as used, so the clip is priced at 0.04 rather than UNAVAILABLE.
        assert response.usage.resolution == "480p"

    async def test_wavespeed_still_honours_a_stated_resolution(self) -> None:
        client = FakeWaveSpeedVideoClient({"status": "completed", "outputs": ["https://x/y.mp4"]})

        response = await WaveSpeedAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video_request(
                model="wavespeed-ai/minimax-h3/image-to-video",
                image=ImageInput(url="https://cdn.test/lion.png"),
                resolution="768p",
            ),
            model="wavespeed-ai/minimax-h3/image-to-video",
        )

        assert client.posted[0][1]["resolution"] == "768p"
        assert response.usage.resolution == "768p"

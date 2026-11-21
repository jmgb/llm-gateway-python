"""Venice: Chat Completions over its REST API, and video as queue-and-retrieve."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from llm_gateway import (
    ConfigurationError,
    ImageInput,
    LLMRequest,
    Message,
    ProviderError,
    ResponseFormat,
    VideoRequest,
)
from llm_gateway.errors import InvalidRequestError, ProviderTimeoutError
from llm_gateway.providers.venice import VeniceAdapter

CHILI = "minimax-h3-chili-text-to-video"
DEEPSEEK = "deepseek-v4-1-flash-1017-uncensored"


class FakeVeniceClient:
    """Answers each POST from a queue of canned replies, recording what it was sent."""

    def __init__(self, *replies: Any) -> None:
        self._replies = list(replies)
        self.posted: list[tuple[str, dict[str, Any]]] = []

    async def post(self, path: str, *, json: dict[str, Any]) -> Any:
        self.posted.append((path, json))
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _completion(content: str = "hello", **overrides: Any) -> dict[str, Any]:
    return {
        "model": DEEPSEEK,
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 90,
            "completion_tokens": 21,
            "completion_tokens_details": {"reasoning_tokens": 14},
        },
        **overrides,
    }


def _chat(**kwargs: Any) -> LLMRequest:
    return LLMRequest(
        model=kwargs.pop("model", DEEPSEEK),
        messages=kwargs.pop("messages", (Message(role="user", content="write the post"),)),
        **kwargs,
    )


class Article(BaseModel):
    title: str
    tags: set[str]


class TestVeniceChat:
    async def test_it_sends_chat_completions_without_venices_own_system_prompt(self) -> None:
        """Venice prepends a prompt of its own unless told not to: ~1500 tokens
        the caller never wrote, paid for on every call, steering the answer."""
        client = FakeVeniceClient(_completion())

        response = await VeniceAdapter(client).generate(
            _chat(system_prompt="be brief", temperature=0.4, max_output_tokens=200),
            model=DEEPSEEK,
        )

        path, body = client.posted[0]
        assert path == "/chat/completions"
        assert body["venice_parameters"] == {"include_venice_system_prompt": False}
        assert body["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "write the post"},
        ]
        assert body["temperature"] == 0.4
        assert body["max_tokens"] == 200
        assert response.output_text == "hello"
        assert response.finish_reason == "stop"
        assert response.model_used == DEEPSEEK

    async def test_reasoning_is_reported_inside_the_output_it_was_billed_in(self) -> None:
        client = FakeVeniceClient(_completion())

        response = await VeniceAdapter(client).generate(_chat(), model=DEEPSEEK)

        assert response.usage.input_tokens == 90
        assert response.usage.output_tokens == 21
        assert response.usage.reasoning_tokens == 14
        assert response.usage.visible_output_tokens == 7

    async def test_an_unreported_usage_is_unknown_not_zero(self) -> None:
        reply = _completion()
        del reply["usage"]

        response = await VeniceAdapter(FakeVeniceClient(reply)).generate(_chat(), model=DEEPSEEK)

        assert response.usage.complete is False
        assert response.usage.output_tokens is None

    async def test_a_schema_is_enforced_by_the_api_with_the_keys_venice_refuses_removed(
        self,
    ) -> None:
        """Venice's grammar answers HTTP 400 "Unimplemented keys" for
        ``uniqueItems``, which Pydantic emits for every ``set`` field. The
        gateway still validates the answer against the full model."""
        client = FakeVeniceClient(_completion('{"title": "t", "tags": ["a"]}'))

        await VeniceAdapter(client).generate(
            _chat(response_format=ResponseFormat.JSON_SCHEMA, response_schema=Article),
            model=DEEPSEEK,
        )

        body = client.posted[0][1]
        format_ = body["response_format"]
        assert format_["type"] == "json_schema"
        assert format_["json_schema"]["strict"] is True
        assert format_["json_schema"]["name"] == "Article"
        schema = format_["json_schema"]["schema"]
        assert "uniqueItems" not in str(schema)
        assert schema["additionalProperties"] is False
        # Enforced by the API field, so it is not paid for twice in the prompt.
        assert all(message["role"] != "system" for message in body["messages"])

    async def test_json_mode_asks_for_json_in_the_messages_too(self) -> None:
        client = FakeVeniceClient(_completion("{}"))

        await VeniceAdapter(client).generate(
            _chat(response_format=ResponseFormat.JSON_OBJECT), model=DEEPSEEK
        )

        body = client.posted[0][1]
        assert body["response_format"] == {"type": "json_object"}
        assert "JSON" in body["messages"][0]["content"]

    async def test_visual_inputs_are_refused_before_anything_is_sent(self) -> None:
        client = FakeVeniceClient()
        request = _chat(
            messages=(
                Message(
                    role="user",
                    content="describe",
                    images=(ImageInput(url="https://cdn.test/a.png"),),
                ),
            )
        )

        with pytest.raises(ConfigurationError, match="visual inputs"):
            await VeniceAdapter(client).generate(request, model=DEEPSEEK)

        assert client.posted == []

    async def test_a_reply_without_choices_has_no_output_text(self) -> None:
        client = FakeVeniceClient({"model": DEEPSEEK, "choices": []})

        response = await VeniceAdapter(client).generate(_chat(), model=DEEPSEEK)

        assert response.output_text is None

    async def test_an_http_failure_is_classified_by_status(self) -> None:
        error = RuntimeError("boom")
        error.status_code = 422  # type: ignore[attr-defined]

        with pytest.raises(InvalidRequestError, match="HTTP 422"):
            await VeniceAdapter(FakeVeniceClient(error)).generate(_chat(), model=DEEPSEEK)


def _video(**kwargs: Any) -> VideoRequest:
    return VideoRequest(
        model=kwargs.pop("model", CHILI),
        prompt=kwargs.pop("prompt", "slow aerial shot over a misty valley"),
        **kwargs,
    )


QUEUED = {"model": CHILI, "queue_id": "q-1"}
PROCESSING = {"status": "PROCESSING", "average_execution_time": 145000}


class TestVeniceVideo:
    async def test_it_queues_then_retrieves_the_mp4_bytes(self) -> None:
        client = FakeVeniceClient(QUEUED, PROCESSING, b"\x00\x00mp4")

        response = await VeniceAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video(duration_seconds=10), model=CHILI
        )

        assert client.posted[0] == (
            "/video/queue",
            {
                "model": CHILI,
                "prompt": "slow aerial shot over a misty valley",
                "duration": "10s",
                "resolution": "768P",
            },
        )
        assert client.posted[1] == ("/video/retrieve", {"model": CHILI, "queue_id": "q-1"})
        assert client.posted[2] == client.posted[1]
        assert response.videos[0].data == b"\x00\x00mp4"
        assert response.videos[0].mime_type == "video/mp4"
        assert response.usage.seconds == 10.0
        assert response.usage.resolution == "768p"
        assert response.usage.videos == 1
        assert response.model_used == CHILI

    async def test_an_unstated_duration_is_the_shortest_clip_the_model_offers(self) -> None:
        """Venice requires a duration and bills by it: 4s is USD 0.12, 15s is
        0.60. A caller who said nothing has not authorised the long one."""
        client = FakeVeniceClient(QUEUED, b"mp4")

        response = await VeniceAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video(), model=CHILI
        )

        assert client.posted[0][1]["duration"] == "4s"
        assert response.usage.seconds == 4.0

    async def test_a_resolution_is_matched_whatever_its_case_and_sent_as_venice_spells_it(
        self,
    ) -> None:
        client = FakeVeniceClient(QUEUED, b"mp4")

        await VeniceAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video(resolution="768p"), model=CHILI
        )

        assert client.posted[0][1]["resolution"] == "768P"

    async def test_an_aspect_ratio_the_model_lists_is_sent_as_given(self) -> None:
        """Vertical for social is the reason the field exists; it does not move the price."""
        client = FakeVeniceClient(QUEUED, b"mp4")

        await VeniceAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video(aspect_ratio="9:16"), model=CHILI
        )

        assert client.posted[0][1]["aspect_ratio"] == "9:16"

    async def test_an_unstated_aspect_ratio_is_not_sent(self) -> None:
        client = FakeVeniceClient(QUEUED, b"mp4")

        await VeniceAdapter(client, poll_interval_seconds=0.0).generate_video(_video(), model=CHILI)

        assert "aspect_ratio" not in client.posted[0][1]

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"resolution": "1080p"}, "resolution"),
            ({"aspect_ratio": "2:3"}, "aspect ratio"),
            ({"duration_seconds": 3}, "duration"),
            ({"duration_seconds": 16}, "duration"),
            ({"image": ImageInput(url="https://cdn.test/frame.png")}, "first frame"),
            ({"webhook_url": "https://app.test/hook"}, "webhook"),
        ],
    )
    async def test_an_option_the_model_cannot_honour_raises_before_anything_is_paid(
        self, options: dict[str, Any], message: str
    ) -> None:
        client = FakeVeniceClient()

        with pytest.raises(ConfigurationError, match=message):
            await VeniceAdapter(client).generate_video(_video(**options), model=CHILI)

        assert client.posted == []

    async def test_a_video_model_whose_constraints_nobody_read_is_refused(self) -> None:
        client = FakeVeniceClient()

        with pytest.raises(ConfigurationError, match="no known constraints"):
            await VeniceAdapter(client).generate_video(
                _video(model="venice-unknown-video"), model="venice-unknown-video"
            )

        assert client.posted == []

    async def test_a_clip_announced_by_download_url_is_returned_as_that_url(self) -> None:
        client = FakeVeniceClient(
            {**QUEUED, "download_url": "https://media.venice.test/clip.mp4"},
            {"status": "COMPLETED"},
        )

        response = await VeniceAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video(), model=CHILI
        )

        assert response.videos[0].url == "https://media.venice.test/clip.mp4"
        assert response.videos[0].data is None

    async def test_a_completed_job_with_nothing_to_return_raises(self) -> None:
        client = FakeVeniceClient(QUEUED, {"status": "COMPLETED"})

        with pytest.raises(ProviderError, match="no video"):
            await VeniceAdapter(client, poll_interval_seconds=0.0).generate_video(
                _video(), model=CHILI
            )

    async def test_a_queue_reply_without_an_id_raises_rather_than_polling_nothing(self) -> None:
        client = FakeVeniceClient({"model": CHILI})

        with pytest.raises(ProviderError, match="queue id"):
            await VeniceAdapter(client).generate_video(_video(), model=CHILI)

        assert len(client.posted) == 1

    async def test_a_job_that_never_finishes_times_out(self) -> None:
        client = FakeVeniceClient(QUEUED, PROCESSING, PROCESSING)

        with pytest.raises(ProviderTimeoutError):
            await VeniceAdapter(
                client, max_poll_attempts=2, poll_interval_seconds=0.0
            ).generate_video(_video(), model=CHILI)

    async def test_one_transient_failure_on_a_status_read_does_not_abandon_a_paid_job(
        self,
    ) -> None:
        unavailable = RuntimeError("busy")
        unavailable.status_code = 503  # type: ignore[attr-defined]
        client = FakeVeniceClient(QUEUED, unavailable, b"mp4")

        response = await VeniceAdapter(client, poll_interval_seconds=0.0).generate_video(
            _video(), model=CHILI
        )

        assert response.videos[0].data == b"mp4"

    async def test_a_content_policy_refusal_is_an_invalid_request(self) -> None:
        refused = RuntimeError("policy")
        refused.status_code = 422  # type: ignore[attr-defined]
        client = FakeVeniceClient(QUEUED, refused)

        with pytest.raises(InvalidRequestError):
            await VeniceAdapter(client, poll_interval_seconds=0.0).generate_video(
                _video(), model=CHILI
            )

    async def test_image_generation_is_not_offered(self) -> None:
        assert VeniceAdapter.capabilities.image_generation is False
        assert VeniceAdapter.capabilities.video_generation is True
        assert VeniceAdapter.capabilities.video_from_image is False

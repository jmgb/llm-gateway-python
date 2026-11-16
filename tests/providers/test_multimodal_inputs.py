"""Visual inputs stay attached to their turn and reach the provider intact."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from llm_gateway import (
    AllAttemptsFailed,
    ConfigurationError,
    FallbackPolicy,
    ImageInput,
    LLMGateway,
    LLMRequest,
    Message,
    ReasoningEffort,
    ResponseFormat,
    UsageRecord,
    VideoInput,
)
from llm_gateway.factories import build_registry
from llm_gateway.models import lookup_model
from llm_gateway.providers.gemini import GeminiAdapter
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openai import OpenAIAdapter
from llm_gateway.providers.openrouter import OpenRouterAdapter


class Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"summary":"a scene"}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=150),
            ),
        )


def client(recorder: Recorder) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorder)))


def test_qwen4_max_declares_its_actual_inputs_efforts_and_rates() -> None:
    info = lookup_model("qwen/qwen4-max")
    assert info is not None
    assert info.provider == "openrouter"
    assert info.modality == "text"
    assert info.input_modalities == ("text", "image", "video")
    assert info.reasoning_efforts == ("none", "low", "high", "xhigh")
    assert info.input_usd_per_mtok == Decimal("1.425")
    assert info.output_usd_per_mtok == Decimal("4.275")


@pytest.mark.parametrize("effort", ["none", "low", "high", "xhigh"])
async def test_qwen_reasoning_reaches_openrouter_without_changing_temperature(
    effort: ReasoningEffort,
) -> None:
    recorder = Recorder()
    gateway = LLMGateway(registry=build_registry(openrouter_client=client(recorder)))
    await gateway.generate(
        LLMRequest(model="qwen/qwen4-max", reasoning_effort=effort, temperature=0.2)
    )
    assert recorder.calls[0]["extra_body"]["reasoning"] == {"effort": effort}
    assert recorder.calls[0]["temperature"] == 0.2


async def test_mixed_inputs_preserve_each_conversation_turn_and_structured_accounting() -> None:
    class Summary(BaseModel):
        summary: str

    recorder = Recorder()
    gateway = LLMGateway(registry=build_registry(openrouter_client=client(recorder)))
    request = LLMRequest(
        model="qwen/qwen4-max",
        system_prompt="Analyse the supplied material.",
        messages=(
            Message(
                "user", "Compare this frame.", images=(ImageInput(url="https://example.com/a.png"),)
            ),
            Message("assistant", "Send the clip."),
            Message(
                "user",
                "Describe the scene.",
                images=(ImageInput(data=b"frame", mime_type="image/jpeg"),),
                videos=(VideoInput(data=b"clip", mime_type="video/mp4"),),
            ),
        ),
        response_format=ResponseFormat.JSON_SCHEMA,
        response_schema=Summary,
    )
    result = await gateway.generate(request)
    messages = recorder.calls[0]["messages"]
    assert "summary" in messages[0]["content"]
    assert messages[1]["content"] == [
        {"type": "text", "text": "Compare this frame."},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]
    assert messages[2] == {"role": "assistant", "content": "Send the clip."}
    assert messages[3]["content"] == [
        {"type": "text", "text": "Describe the scene."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZnJhbWU="}},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,Y2xpcA=="}},
    ]
    assert result.output == Summary(summary="a scene")
    assert result.usage.output_tokens == 200
    assert result.usage.visible_output_tokens == 50
    assert result.cost.microusd == 2280


@pytest.mark.parametrize(
    "url", ["https://example.com/clip.webm", "data:video/webm;base64,Y2xpcA=="]
)
async def test_video_urls_are_forwarded_without_fetching_or_reencoding(url: str) -> None:
    recorder = Recorder()
    await OpenRouterAdapter(client(recorder)).generate(
        LLMRequest(
            model="other/video-model",
            messages=(Message("user", "Describe.", videos=(VideoInput(url=url),)),),
        ),
        model="other/video-model",
    )
    assert recorder.calls[0]["messages"][0]["content"][-1] == {
        "type": "video_url",
        "video_url": {"url": url},
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"data": b""},
        {"url": " "},
        {"data": b"x"},
        {"data": b"x", "mime_type": "image/png"},
        {"data": b"x", "url": "https://example.com/a.mp4", "mime_type": "video/mp4"},
    ],
)
def test_invalid_video_sources_fail_before_any_call(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        VideoInput(**kwargs)


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_visual_inputs_require_a_user_turn(role: Any) -> None:
    with pytest.raises(ValueError, match="user"):
        Message(role, "Describe.", images=(ImageInput(url="https://example.com/a.png"),))


@pytest.mark.parametrize("adapter_type", [OpenAIAdapter, GeminiAdapter, GroqAdapter])
@pytest.mark.parametrize("kind", ["images", "videos"])
async def test_adapters_refuse_inputs_they_cannot_translate(adapter_type: Any, kind: str) -> None:
    message = (
        Message("user", "Describe.", images=(ImageInput(url="https://example.com/a.png"),))
        if kind == "images"
        else Message("user", "Describe.", videos=(VideoInput(url="https://example.com/a.mp4"),))
    )
    request = LLMRequest(model="model", messages=(message,))
    with pytest.raises(ConfigurationError, match="visual inputs"):
        await adapter_type(SimpleNamespace()).generate(request, model="model")


class Sinks:
    def __init__(self) -> None:
        self.records: list[UsageRecord] = []
        self.events: list[tuple[str, dict[str, object]]] = []

    def record(self, usage: UsageRecord) -> None:
        self.records.append(usage)

    def emit(self, event: str, fields: dict[str, object]) -> None:
        self.events.append((event, fields))


async def test_visual_content_never_reaches_usage_or_event_sinks() -> None:
    sinks = Sinks()
    gateway = LLMGateway(
        registry=build_registry(openrouter_client=client(Recorder())),
        usage_sink=sinks,
        event_sink=sinks,
    )
    await gateway.generate(
        LLMRequest(
            model="qwen/qwen4-max",
            messages=(
                Message(
                    "user",
                    "private prompt",
                    images=(ImageInput(data=b"private image"),),
                    videos=(VideoInput(url="https://private.example/secret.mp4"),),
                ),
            ),
        )
    )
    telemetry = repr(sinks.records) + repr(sinks.events)
    assert sinks.records and sinks.events
    for content in ("private prompt", "private image", "secret.mp4", "cHJpdmF0ZSBpbWFnZQ=="):
        assert content not in telemetry


async def test_an_unsupported_visual_request_is_a_nonbillable_configuration_failure() -> None:
    recorder = Recorder()
    gateway = LLMGateway(
        registry=build_registry(openai_client=SimpleNamespace(), openrouter_client=client(recorder))
    )
    with pytest.raises(AllAttemptsFailed) as raised:
        await gateway.generate(
            LLMRequest(
                model="gpt-6-luna",
                messages=(
                    Message(
                        "user",
                        "Describe.",
                        videos=(VideoInput(url="https://example.com/a.mp4"),),
                    ),
                ),
            )
        )
    assert raised.value.attempts[0].billable is False
    assert recorder.calls == []


async def test_a_visual_request_can_fall_back_from_an_unsupported_adapter_to_openrouter() -> None:
    recorder = Recorder()
    gateway = LLMGateway(
        registry=build_registry(openai_client=SimpleNamespace(), openrouter_client=client(recorder))
    )
    result = await gateway.generate(
        LLMRequest(
            model="gpt-6-luna",
            messages=(
                Message("user", "Describe.", videos=(VideoInput(url="https://example.com/a.mp4"),)),
            ),
            fallback_policy=FallbackPolicy.models_in_order("qwen/qwen4-max"),
        )
    )
    assert result.execution.fallback_used
    assert result.execution.attempts[0].billable is False
    assert (
        recorder.calls[0]["messages"][0]["content"][-1]["video_url"]["url"]
        == "https://example.com/a.mp4"
    )

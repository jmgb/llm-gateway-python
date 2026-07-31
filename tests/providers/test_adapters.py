"""Adapters translate one call, in both directions, and nothing else.

Clients are injected: the package never constructs one, never reads an
environment variable and never holds a credential. That is what makes these
tests possible without a network and without any extra installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm_gateway import (
    LLMRequest,
    Message,
    RateLimitedError,
    ResponseFormat,
)
from llm_gateway.providers.gemini import GeminiAdapter
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openai import OpenAIAdapter
from llm_gateway.providers.openrouter import OpenRouterAdapter


class Recorder:
    """Captures the kwargs an adapter sends to its SDK."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def _request(**kwargs: Any) -> LLMRequest:
    return LLMRequest(
        model=kwargs.pop("model", "m"),
        messages=(Message("user", "a question"),),
        system_prompt=kwargs.pop("system_prompt", "you are an assistant"),
        **kwargs,
    )


class TestOpenAIAdapter:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(responses=SimpleNamespace(create=recorder))

    async def test_it_returns_the_output_text(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                output_text="an answer",
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
                status="completed",
            )
        )

        response = await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert response.output_text == "an answer"

    async def test_it_maps_token_usage(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                output_text="x",
                usage=SimpleNamespace(
                    input_tokens=11,
                    output_tokens=7,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=3),
                ),
                status="completed",
            )
        )

        response = await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert response.usage.input_tokens == 11
        assert response.usage.output_tokens == 7
        assert response.usage.reasoning_tokens == 3
        assert response.usage.billable_output_tokens == 7, (
            "the Responses API already counts reasoning inside output_tokens"
        )

    async def test_missing_usage_is_unknown_not_zero(self) -> None:
        recorder = Recorder(SimpleNamespace(output_text="x", usage=None, status="completed"))

        response = await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert response.usage.complete is False
        assert response.usage.input_tokens is None

    async def test_the_system_prompt_travels_as_the_first_input_message(self) -> None:
        recorder = Recorder(SimpleNamespace(output_text="x", usage=None, status="completed"))

        await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert recorder.kwargs["input"][0] == {
            "role": "system",
            "content": "you are an assistant",
        }
        assert "instructions" not in recorder.kwargs, (
            "json_object mode only inspects the input for the word 'json'"
        )
        assert recorder.kwargs["model"] == "gpt-x"

    async def test_json_mode_can_be_satisfied_by_the_system_prompt(self) -> None:
        recorder = Recorder(SimpleNamespace(output_text="{}", usage=None, status="completed"))

        await OpenAIAdapter(self._client(recorder)).generate(
            _request(
                system_prompt="Reply with json.",
                response_format=ResponseFormat.JSON_OBJECT,
            ),
            model="gpt-x",
        )

        rendered = " ".join(message["content"] for message in recorder.kwargs["input"])
        assert "json" in rendered.lower()
        assert recorder.kwargs["text"] == {"format": {"type": "json_object"}}

    async def test_a_request_without_a_system_prompt_sends_only_its_messages(self) -> None:
        recorder = Recorder(SimpleNamespace(output_text="x", usage=None, status="completed"))

        await OpenAIAdapter(self._client(recorder)).generate(
            _request(system_prompt=None), model="gpt-x"
        )

        assert [m["role"] for m in recorder.kwargs["input"]] == ["user"]

    async def test_sdk_errors_become_typed_errors(self) -> None:
        error = Exception("rate limited")
        error.status_code = 429  # type: ignore[attr-defined]
        recorder = Recorder(error=error)

        with pytest.raises(RateLimitedError):
            await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

    async def test_temperature_is_omitted_when_unset(self) -> None:
        recorder = Recorder(SimpleNamespace(output_text="x", usage=None, status="completed"))

        await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert "temperature" not in recorder.kwargs

    @pytest.mark.parametrize("effort", ("none", "low", "medium", "high", "xhigh", "max"))
    async def test_it_forwards_each_supported_reasoning_effort(self, effort: str) -> None:
        recorder = Recorder(SimpleNamespace(output_text="x", usage=None, status="completed"))

        await OpenAIAdapter(self._client(recorder)).generate(
            _request(reasoning_effort=effort), model="gpt-5.6-terra"
        )

        assert recorder.kwargs["reasoning"] == {"effort": effort}


class TestGeminiAdapter:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=recorder))
        )

    async def test_it_returns_the_text(self) -> None:
        recorder = Recorder(SimpleNamespace(text="an answer", usage_metadata=None))

        response = await GeminiAdapter(self._client(recorder)).generate(
            _request(), model="gemini-x"
        )

        assert response.output_text == "an answer"

    async def test_it_maps_gemini_usage_names(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                text="x",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=20,
                    candidates_token_count=9,
                    thoughts_token_count=4,
                ),
            )
        )

        response = await GeminiAdapter(self._client(recorder)).generate(
            _request(), model="gemini-x"
        )

        assert response.usage.input_tokens == 20
        assert response.usage.output_tokens == 13, (
            "candidates_token_count excludes thoughts; the neutral contract includes them"
        )
        assert response.usage.reasoning_tokens == 4
        assert response.usage.visible_output_tokens == 9

    async def test_thoughts_alone_do_not_pass_for_a_complete_output_count(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                text="x",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=20,
                    candidates_token_count=None,
                    thoughts_token_count=4,
                ),
            )
        )

        response = await GeminiAdapter(self._client(recorder)).generate(
            _request(), model="gemini-x"
        )

        assert response.usage.output_tokens is None
        assert response.usage.complete is False

    async def test_the_system_prompt_travels_in_the_config(self) -> None:
        recorder = Recorder(SimpleNamespace(text="x", usage_metadata=None))

        await GeminiAdapter(self._client(recorder)).generate(_request(), model="gemini-x")

        assert recorder.kwargs["config"]["system_instruction"] == "you are an assistant"

    async def test_json_mode_is_requested_as_a_mime_type(self) -> None:
        recorder = Recorder(SimpleNamespace(text="{}", usage_metadata=None))

        await GeminiAdapter(self._client(recorder)).generate(
            _request(response_format=ResponseFormat.JSON_OBJECT), model="gemini-x"
        )

        assert recorder.kwargs["config"]["response_mime_type"] == "application/json"

    async def test_it_maps_gemini_3_reasoning_effort_to_a_thinking_level(self) -> None:
        recorder = Recorder(SimpleNamespace(text="x", usage_metadata=None))

        await GeminiAdapter(self._client(recorder)).generate(
            _request(reasoning_effort="medium"), model="gemini-3.6-flash"
        )

        assert recorder.kwargs["config"]["thinking_config"] == {"thinking_level": "medium"}


class TestGroqAdapter:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorder)))

    async def test_it_returns_the_message_content(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="an answer"), finish_reason="stop"
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
            )
        )

        response = await GroqAdapter(self._client(recorder)).generate(_request(), model="llama-x")

        assert response.output_text == "an answer"
        assert response.usage.input_tokens == 5
        assert response.finish_reason == "stop"

    async def test_the_system_prompt_becomes_the_first_message(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="x"), finish_reason=None)],
                usage=None,
            )
        )

        await GroqAdapter(self._client(recorder)).generate(_request(), model="llama-x")

        assert recorder.kwargs["messages"][0] == {
            "role": "system",
            "content": "you are an assistant",
        }

    async def test_an_empty_choice_list_is_an_unknown_output_not_a_crash(self) -> None:
        recorder = Recorder(SimpleNamespace(choices=[], usage=None))

        response = await GroqAdapter(self._client(recorder)).generate(_request(), model="llama-x")

        assert response.output_text is None

    async def test_it_maps_the_reasoning_breakdown_when_one_is_reported(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="x"), finish_reason=None)],
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
                ),
            )
        )

        response = await GroqAdapter(self._client(recorder)).generate(_request(), model="llama-x")

        assert response.usage.output_tokens == 7
        assert response.usage.reasoning_tokens == 3
        assert response.usage.billable_output_tokens == 7

    async def test_a_model_that_does_not_think_reports_no_breakdown(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="x"), finish_reason=None)],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            )
        )

        response = await GroqAdapter(self._client(recorder)).generate(_request(), model="llama-x")

        assert response.usage.reasoning_tokens is None
        assert response.usage.visible_output_tokens == 7

    async def test_it_forwards_reasoning_effort_for_gpt_oss(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="x"), finish_reason=None)],
                usage=None,
            )
        )

        await GroqAdapter(self._client(recorder)).generate(
            _request(reasoning_effort="high"), model="openai/gpt-oss-120b"
        )

        assert recorder.kwargs["reasoning_effort"] == "high"


class TestOpenRouterAdapter:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorder)))

    def _reply(self, **kwargs: Any) -> Any:
        kwargs.setdefault("usage", None)
        kwargs.setdefault(
            "choices",
            [SimpleNamespace(message=SimpleNamespace(content="an answer"), finish_reason="stop")],
        )
        return SimpleNamespace(**kwargs)

    async def test_it_returns_the_message_content(self) -> None:
        recorder = Recorder(
            self._reply(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))
        )

        response = await OpenRouterAdapter(self._client(recorder)).generate(
            _request(), model="deepseek/deepseek-chat-v3.1"
        )

        assert response.output_text == "an answer"
        assert response.usage.input_tokens == 5
        assert response.usage.output_tokens == 2
        assert response.finish_reason == "stop"

    async def test_the_system_prompt_becomes_the_first_message(self) -> None:
        recorder = Recorder(self._reply())

        await OpenRouterAdapter(self._client(recorder)).generate(_request(), model="x/y")

        assert recorder.kwargs["messages"][0] == {
            "role": "system",
            "content": "you are an assistant",
        }

    async def test_a_schema_request_asks_only_for_json(self) -> None:
        """An aggregator cannot promise schema enforcement for every model.

        Asking for JSON is the most it can honestly do; the gateway validates
        the payload afterwards.
        """
        recorder = Recorder(self._reply())

        await OpenRouterAdapter(self._client(recorder)).generate(
            _request(response_format=ResponseFormat.JSON_OBJECT), model="x/y"
        )

        assert recorder.kwargs["response_format"] == {"type": "json_object"}

    async def test_it_reports_the_model_that_actually_served_the_call(self) -> None:
        """`openrouter/auto` picks a model, so the reply names a different one."""
        recorder = Recorder(self._reply(model="deepseek/deepseek-v4-pro"))

        response = await OpenRouterAdapter(self._client(recorder)).generate(
            _request(), model="openrouter/auto"
        )

        assert response.model_used == "deepseek/deepseek-v4-pro"

    async def test_it_maps_cached_and_reasoning_token_details(self) -> None:
        recorder = Recorder(
            self._reply(
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=40,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=64),
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=25),
                )
            )
        )

        response = await OpenRouterAdapter(self._client(recorder)).generate(_request(), model="x/y")

        assert response.usage.cached_input_tokens == 64
        assert response.usage.reasoning_tokens == 25

    async def test_missing_usage_is_unknown_not_zero(self) -> None:
        recorder = Recorder(self._reply())

        response = await OpenRouterAdapter(self._client(recorder)).generate(_request(), model="x/y")

        assert response.usage.input_tokens is None

    async def test_an_empty_choice_list_is_an_unknown_output_not_a_crash(self) -> None:
        recorder = Recorder(SimpleNamespace(choices=[], usage=None))

        response = await OpenRouterAdapter(self._client(recorder)).generate(_request(), model="x/y")

        assert response.output_text is None

    async def test_sdk_errors_become_typed_errors(self) -> None:
        error = Exception("rate limited")
        error.status_code = 429  # type: ignore[attr-defined]
        recorder = Recorder(error=error)

        with pytest.raises(RateLimitedError):
            await OpenRouterAdapter(self._client(recorder)).generate(_request(), model="x/y")


class TestCapabilities:
    def test_each_adapter_declares_what_it_supports(self) -> None:
        assert OpenAIAdapter(SimpleNamespace()).capabilities.structured_outputs is True
        assert GeminiAdapter(SimpleNamespace()).capabilities.reasoning_effort is True
        assert GroqAdapter(SimpleNamespace()).capabilities.reasoning_effort is True

    def test_an_aggregator_does_not_promise_what_its_models_may_not_support(self) -> None:
        capabilities = OpenRouterAdapter(SimpleNamespace()).capabilities

        assert capabilities.structured_outputs is False
        assert capabilities.json_mode is True

    def test_adapters_have_stable_names(self) -> None:
        assert OpenAIAdapter(SimpleNamespace()).name == "openai"
        assert GeminiAdapter(SimpleNamespace()).name == "gemini"
        assert GroqAdapter(SimpleNamespace()).name == "groq"
        assert OpenRouterAdapter(SimpleNamespace()).name == "openrouter"

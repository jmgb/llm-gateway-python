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
        messages=(Message("user", "pregunta"),),
        system_prompt=kwargs.pop("system_prompt", "eres un asistente"),
        **kwargs,
    )


class TestOpenAIAdapter:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(responses=SimpleNamespace(create=recorder))

    async def test_it_returns_the_output_text(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                output_text="respuesta",
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
                status="completed",
            )
        )

        response = await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert response.output_text == "respuesta"

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

    async def test_missing_usage_is_unknown_not_zero(self) -> None:
        recorder = Recorder(SimpleNamespace(output_text="x", usage=None, status="completed"))

        response = await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert response.usage.complete is False
        assert response.usage.input_tokens is None

    async def test_the_system_prompt_travels_as_instructions(self) -> None:
        recorder = Recorder(SimpleNamespace(output_text="x", usage=None, status="completed"))

        await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert recorder.kwargs["instructions"] == "eres un asistente"
        assert recorder.kwargs["model"] == "gpt-x"

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


class TestGeminiAdapter:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=recorder))
        )

    async def test_it_returns_the_text(self) -> None:
        recorder = Recorder(SimpleNamespace(text="respuesta", usage_metadata=None))

        response = await GeminiAdapter(self._client(recorder)).generate(
            _request(), model="gemini-x"
        )

        assert response.output_text == "respuesta"

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
        assert response.usage.output_tokens == 9
        assert response.usage.reasoning_tokens == 4

    async def test_the_system_prompt_travels_in_the_config(self) -> None:
        recorder = Recorder(SimpleNamespace(text="x", usage_metadata=None))

        await GeminiAdapter(self._client(recorder)).generate(_request(), model="gemini-x")

        assert recorder.kwargs["config"]["system_instruction"] == "eres un asistente"

    async def test_json_mode_is_requested_as_a_mime_type(self) -> None:
        recorder = Recorder(SimpleNamespace(text="{}", usage_metadata=None))

        await GeminiAdapter(self._client(recorder)).generate(
            _request(response_format=ResponseFormat.JSON_OBJECT), model="gemini-x"
        )

        assert recorder.kwargs["config"]["response_mime_type"] == "application/json"


class TestGroqAdapter:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorder)))

    async def test_it_returns_the_message_content(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="respuesta"), finish_reason="stop"
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
            )
        )

        response = await GroqAdapter(self._client(recorder)).generate(_request(), model="llama-x")

        assert response.output_text == "respuesta"
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

        assert recorder.kwargs["messages"][0] == {"role": "system", "content": "eres un asistente"}

    async def test_an_empty_choice_list_is_an_unknown_output_not_a_crash(self) -> None:
        recorder = Recorder(SimpleNamespace(choices=[], usage=None))

        response = await GroqAdapter(self._client(recorder)).generate(_request(), model="llama-x")

        assert response.output_text is None


class TestCapabilities:
    def test_each_adapter_declares_what_it_supports(self) -> None:
        assert OpenAIAdapter(SimpleNamespace()).capabilities.structured_outputs is True
        assert GroqAdapter(SimpleNamespace()).capabilities.reasoning_effort is False

    def test_adapters_have_stable_names(self) -> None:
        assert OpenAIAdapter(SimpleNamespace()).name == "openai"
        assert GeminiAdapter(SimpleNamespace()).name == "gemini"
        assert GroqAdapter(SimpleNamespace()).name == "groq"

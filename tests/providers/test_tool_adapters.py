"""The two wire shapes function tools actually have, and who may claim them.

OpenAI's Responses API takes a flat tool and answers with `function_call` items
carrying a `call_id`; Chat Completions nests the tool under `function` and
answers inside `choices[0].message.tool_calls`. A continuation has to put the
model's own call back on the wire *before* the result, in both dialects, or the
provider rejects an output that answers nothing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from llm_gateway import (
    ConfigurationError,
    FunctionTool,
    LLMRequest,
    Message,
    RequiredTool,
    ResponseFormat,
    ToolCall,
    ToolChoice,
    ToolResult,
)
from llm_gateway.providers.gemini import GeminiAdapter
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openai import OpenAIAdapter
from llm_gateway.providers.openrouter import OpenRouterAdapter

WEATHER = FunctionTool(
    name="get_weather",
    description="Current weather for a city",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


class Recorder:
    """Captures the kwargs an adapter sends to its SDK."""

    def __init__(self, response: Any = None) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _request(**kwargs: Any) -> LLMRequest:
    kwargs.setdefault("tools", (WEATHER,))
    return LLMRequest(
        model="m",
        messages=(Message("user", "weather in Madrid?"),),
        system_prompt="you are an assistant",
        **kwargs,
    )


def _responses_reply(*items: Any) -> Any:
    return SimpleNamespace(
        output_text="",
        output=list(items),
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        status="completed",
    )


def _function_call_item(call_id: str, name: str, arguments: Any) -> Any:
    return SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments)


def _chat_reply(*tool_calls: Any, content: str | None = None) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=list(tool_calls) or None),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


def _chat_tool_call(call_id: str, name: str, arguments: str) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TestOpenAISendsTools:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(responses=SimpleNamespace(create=recorder))

    async def test_it_sends_a_flat_function_tool(self) -> None:
        recorder = Recorder(_responses_reply())

        await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert recorder.kwargs["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            }
        ]

    async def test_it_defaults_to_letting_the_model_decide(self) -> None:
        recorder = Recorder(_responses_reply())

        await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert recorder.kwargs["tool_choice"] == "auto"

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            (ToolChoice.AUTO, "auto"),
            (ToolChoice.NONE, "none"),
            (ToolChoice.REQUIRED, "required"),
            (RequiredTool("get_weather"), {"type": "function", "name": "get_weather"}),
        ],
    )
    async def test_it_translates_every_tool_choice(self, choice: Any, expected: Any) -> None:
        recorder = Recorder(_responses_reply())

        await OpenAIAdapter(self._client(recorder)).generate(
            _request(tool_choice=choice), model="gpt-x"
        )

        assert recorder.kwargs["tool_choice"] == expected

    async def test_a_request_without_tools_sends_no_tool_fields(self) -> None:
        """Every call that predates this contract must go out exactly as before."""
        recorder = Recorder(_responses_reply())

        await OpenAIAdapter(self._client(recorder)).generate(_request(tools=()), model="gpt-x")

        assert "tools" not in recorder.kwargs
        assert "tool_choice" not in recorder.kwargs


class TestOpenAIReadsCalls:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(responses=SimpleNamespace(create=recorder))

    async def test_it_reads_the_call_id_name_and_arguments(self) -> None:
        recorder = Recorder(
            _responses_reply(_function_call_item("call_1", "get_weather", '{"city": "Madrid"}'))
        )

        response = await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_1"
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == '{"city": "Madrid"}'

    async def test_it_serialises_arguments_the_sdk_already_decoded(self) -> None:
        """Some SDK versions hand back a dict; the gateway parses one shape."""
        recorder = Recorder(
            _responses_reply(_function_call_item("call_1", "get_weather", {"city": "Madrid"}))
        )

        response = await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert json.loads(response.tool_calls[0].arguments) == {"city": "Madrid"}

    async def test_it_keeps_reasoning_items_out_of_the_calls(self) -> None:
        recorder = Recorder(
            _responses_reply(
                SimpleNamespace(type="reasoning", summary=[]),
                _function_call_item("call_1", "get_weather", "{}"),
            )
        )

        response = await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert [call.name for call in response.tool_calls] == ["get_weather"]

    async def test_an_answer_with_no_calls_reports_none(self) -> None:
        recorder = Recorder(
            SimpleNamespace(
                output_text="Sunny",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                status="completed",
            )
        )

        response = await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert response.tool_calls == ()
        assert response.output_text == "Sunny"

    async def test_it_does_not_invent_a_missing_call_id(self) -> None:
        recorder = Recorder(_responses_reply(_function_call_item("", "get_weather", "{}")))

        response = await OpenAIAdapter(self._client(recorder)).generate(_request(), model="gpt-x")

        assert response.tool_calls[0].id == ""


class TestOpenAIContinues:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(responses=SimpleNamespace(create=recorder))

    async def test_the_call_is_replayed_before_its_output(self) -> None:
        """A `function_call_output` with no `function_call` above it is a 400."""
        recorder = Recorder(_responses_reply())
        call = ToolCall(id="call_1", name="get_weather", arguments={"city": "Madrid"})

        await OpenAIAdapter(self._client(recorder)).generate(
            _request(tool_results=(ToolResult(call, "18C and sunny"),)), model="gpt-x"
        )

        sent = recorder.kwargs["input"]
        assert sent[-2] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Madrid"}',
        }
        assert sent[-1] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "18C and sunny",
        }

    async def test_every_result_keeps_its_own_call_id(self) -> None:
        recorder = Recorder(_responses_reply())
        first = ToolCall(id="call_1", name="get_weather", arguments={"city": "Madrid"})
        second = ToolCall(id="call_2", name="get_weather", arguments={"city": "Oporto"})

        await OpenAIAdapter(self._client(recorder)).generate(
            _request(tool_results=(ToolResult(first, "18C"), ToolResult(second, "21C"))),
            model="gpt-x",
        )

        outputs = [item for item in recorder.kwargs["input"] if isinstance(item, dict)]
        outputs = [item for item in outputs if item.get("type") == "function_call_output"]
        assert [(item["call_id"], item["output"]) for item in outputs] == [
            ("call_1", "18C"),
            ("call_2", "21C"),
        ]


class TestGroqSendsTools:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorder)))

    async def test_it_nests_the_function_the_way_chat_completions_wants(self) -> None:
        recorder = Recorder(_chat_reply(content="x"))

        await GroqAdapter(self._client(recorder)).generate(_request(), model="llama")

        assert recorder.kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            (ToolChoice.AUTO, "auto"),
            (ToolChoice.NONE, "none"),
            (ToolChoice.REQUIRED, "required"),
            (
                RequiredTool("get_weather"),
                {"type": "function", "function": {"name": "get_weather"}},
            ),
        ],
    )
    async def test_it_translates_every_tool_choice(self, choice: Any, expected: Any) -> None:
        recorder = Recorder(_chat_reply(content="x"))

        await GroqAdapter(self._client(recorder)).generate(
            _request(tool_choice=choice), model="llama"
        )

        assert recorder.kwargs["tool_choice"] == expected

    async def test_it_does_not_ask_for_json_while_tools_are_available(self) -> None:
        """Both shipped consumers send tools or the JSON format, never the pair."""
        recorder = Recorder(_chat_reply(content="x"))

        await GroqAdapter(self._client(recorder)).generate(
            _request(response_format=ResponseFormat.JSON_OBJECT), model="llama"
        )

        assert "response_format" not in recorder.kwargs

    async def test_a_json_request_without_tools_is_unchanged(self) -> None:
        recorder = Recorder(_chat_reply(content="{}"))

        await GroqAdapter(self._client(recorder)).generate(
            _request(tools=(), response_format=ResponseFormat.JSON_OBJECT), model="llama"
        )

        assert recorder.kwargs["response_format"] == {"type": "json_object"}


class TestGroqReadsCalls:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorder)))

    async def test_it_reads_the_calls_off_the_message(self) -> None:
        recorder = Recorder(
            _chat_reply(_chat_tool_call("call_1", "get_weather", '{"city": "Madrid"}'))
        )

        response = await GroqAdapter(self._client(recorder)).generate(_request(), model="llama")

        assert response.tool_calls[0].id == "call_1"
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == '{"city": "Madrid"}'

    async def test_a_plain_answer_reports_no_calls(self) -> None:
        recorder = Recorder(_chat_reply(content="Sunny"))

        response = await GroqAdapter(self._client(recorder)).generate(_request(), model="llama")

        assert response.tool_calls == ()
        assert response.output_text == "Sunny"

    async def test_it_does_not_invent_a_missing_call_id(self) -> None:
        recorder = Recorder(_chat_reply(_chat_tool_call("", "get_weather", "{}")))

        response = await GroqAdapter(self._client(recorder)).generate(_request(), model="llama")

        assert response.tool_calls[0].id == ""


class TestGroqContinues:
    def _client(self, recorder: Recorder) -> Any:
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorder)))

    async def test_the_assistant_message_carries_the_calls_it_made(self) -> None:
        recorder = Recorder(_chat_reply(content="Sunny"))
        call = ToolCall(id="call_1", name="get_weather", arguments={"city": "Madrid"})

        await GroqAdapter(self._client(recorder)).generate(
            _request(tool_results=(ToolResult(call, "18C and sunny"),)), model="llama"
        )

        sent = recorder.kwargs["messages"]
        assert sent[-2] == {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Madrid"}',
                    },
                }
            ],
        }
        assert sent[-1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "18C and sunny",
        }

    async def test_several_results_follow_one_assistant_message(self) -> None:
        recorder = Recorder(_chat_reply(content="Sunny"))
        first = ToolCall(id="call_1", name="get_weather", arguments={"city": "Madrid"})
        second = ToolCall(id="call_2", name="get_weather", arguments={"city": "Oporto"})

        await GroqAdapter(self._client(recorder)).generate(
            _request(tool_results=(ToolResult(first, "18C"), ToolResult(second, "21C"))),
            model="llama",
        )

        sent = recorder.kwargs["messages"]
        assert len(sent[-3]["tool_calls"]) == 2
        assert [(m["tool_call_id"], m["content"]) for m in sent[-2:]] == [
            ("call_1", "18C"),
            ("call_2", "21C"),
        ]


class TestWhoMayClaimTools:
    async def test_openai_and_groq_declare_the_capability(self) -> None:
        from llm_gateway.providers.groq import CAPABILITIES as GROQ
        from llm_gateway.providers.openai import CAPABILITIES as OPENAI

        assert OPENAI.function_calling is True
        assert GROQ.function_calling is True

    @pytest.mark.parametrize("adapter_type", [GeminiAdapter, OpenRouterAdapter])
    async def test_an_adapter_without_the_capability_refuses_rather_than_drops(
        self, adapter_type: Any
    ) -> None:
        """Silently dropping tools returns prose where the caller expects a call."""
        with pytest.raises(ConfigurationError, match="tool"):
            await adapter_type(SimpleNamespace()).generate(_request(), model="m")

    @pytest.mark.parametrize("adapter_type", [GeminiAdapter, OpenRouterAdapter])
    async def test_it_refuses_tool_results_too(self, adapter_type: Any) -> None:
        call = ToolCall(id="call_1", name="get_weather", arguments={})

        with pytest.raises(ConfigurationError, match="tool"):
            await adapter_type(SimpleNamespace()).generate(
                _request(tools=(), tool_results=(ToolResult(call, "18C"),)), model="m"
            )

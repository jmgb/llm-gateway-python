"""A tool call is an answer the caller has to act on, not text to be parsed.

The package translates and correlates; it never executes a function and never
runs a business loop. What it owes the caller is that the provider's call id
survives, that arguments it cannot hand over are a failed attempt rather than a
plausible one, and that every billed attempt is still visible.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from llm_gateway import (
    AllAttemptsFailed,
    AttemptOutcome,
    FailurePhase,
    FallbackPolicy,
    FunctionTool,
    LLMGateway,
    LLMRequest,
    Message,
    ModelRate,
    ProviderRegistry,
    ProviderResponse,
    ProviderToolCall,
    RateLimitedError,
    RequiredTool,
    ResponseFormat,
    StaticPriceCatalog,
    TokenUsage,
    ToolCall,
    ToolChoice,
    ToolResult,
)

CATALOG = StaticPriceCatalog(
    version="test-1",
    rates={
        "fast": ModelRate(Decimal("1"), Decimal("2")),
        "slow": ModelRate(Decimal("1"), Decimal("2")),
    },
)

WEATHER = FunctionTool(
    name="get_weather",
    description="Current weather for a city",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
CALENDAR = FunctionTool(name="list_events", parameters={"type": "object", "properties": {}})


class FakeAdapter:
    """Returns queued responses or raises queued errors, in order."""

    name = "fake"

    def __init__(self, *responses: ProviderResponse | Exception) -> None:
        self._queue = list(responses)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        self.requests.append(request)
        item = self._queue.pop(0) if self._queue else _text("default")
        if isinstance(item, Exception):
            raise item
        return item


class RecordingSink:
    """Captures everything the gateway reports, to prove what it does not."""

    def __init__(self) -> None:
        self.records: list[Any] = []
        self.events: list[tuple[str, dict[str, object]]] = []

    def record(self, record: Any) -> None:
        self.records.append(record)

    def emit(self, event: str, fields: dict[str, object]) -> None:
        self.events.append((event, fields))


def _text(text: str) -> ProviderResponse:
    return ProviderResponse(
        output_text=text,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        finish_reason="stop",
    )


def _calls(*calls: ProviderToolCall) -> ProviderResponse:
    """What a provider answers when it wants a function run instead."""
    return ProviderResponse(
        output_text="",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        finish_reason="tool_calls",
        tool_calls=calls,
    )


def _gateway(adapter: FakeAdapter, sink: RecordingSink | None = None) -> LLMGateway:
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=("fast", "slow"))
    return LLMGateway(
        registry=registry,
        price_catalog=CATALOG,
        usage_sink=sink,
        event_sink=sink,
    )


def _request(**kwargs: Any) -> LLMRequest:
    kwargs.setdefault("tools", (WEATHER,))
    return LLMRequest(model="fast", messages=(Message("user", "weather in Madrid?"),), **kwargs)


class TestTheToolContract:
    def test_a_function_needs_a_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            FunctionTool(name="  ")

    def test_two_tools_cannot_share_a_name(self) -> None:
        """The provider picks one by name, so a duplicate makes it ambiguous."""
        with pytest.raises(ValueError, match="get_weather"):
            _request(tools=(WEATHER, WEATHER))

    def test_choosing_a_tool_requires_tools_to_choose_from(self) -> None:
        with pytest.raises(ValueError, match="tool_choice"):
            _request(tools=(), tool_choice=ToolChoice.REQUIRED)

    def test_a_forced_tool_must_be_one_that_was_declared(self) -> None:
        with pytest.raises(ValueError, match="send_email"):
            _request(tools=(WEATHER,), tool_choice=RequiredTool("send_email"))

    def test_two_results_cannot_answer_the_same_call(self) -> None:
        """A duplicate id is a correlation bug the provider would answer 400 to."""
        call = _tool_call("call_1", "get_weather", {"city": "Madrid"})
        with pytest.raises(ValueError, match="call_1"):
            _request(tool_results=(ToolResult(call, "sunny"), ToolResult(call, "rainy")))

    def test_a_result_needs_the_call_it_answers(self) -> None:
        """Correlation by construction: there is no way to supply a loose result."""
        with pytest.raises(TypeError):
            ToolResult(output="sunny")  # type: ignore[call-arg]

    @pytest.mark.parametrize(("call_id", "name"), [("", "get_weather"), ("call_1", " ")])
    def test_a_call_needs_its_provider_id_and_function_name(self, call_id: str, name: str) -> None:
        with pytest.raises(ValueError):
            ToolCall(id=call_id, name=name, arguments={})


class TestReceivingCalls:
    async def test_a_tool_call_is_a_successful_attempt(self) -> None:
        """The provider answered and will invoice it; it is not a failure."""
        adapter = FakeAdapter(_calls(_provider_call("call_1", "get_weather", '{"city": "Madrid"}')))

        result = await _gateway(adapter).generate(_request())

        assert result.execution.attempts[0].outcome is AttemptOutcome.SUCCEEDED
        assert result.usage.input_tokens == 10
        assert result.cost.microusd is not None

    async def test_the_call_id_name_and_arguments_survive(self) -> None:
        adapter = FakeAdapter(
            _calls(_provider_call("call_ab12", "get_weather", '{"city": "Madrid"}'))
        )

        result = await _gateway(adapter).generate(_request())

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_ab12"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Madrid"}

    async def test_several_calls_in_one_answer_are_all_returned(self) -> None:
        adapter = FakeAdapter(
            _calls(
                _provider_call("call_1", "get_weather", '{"city": "Madrid"}'),
                _provider_call("call_2", "list_events", "{}"),
            )
        )

        result = await _gateway(adapter).generate(_request(tools=(WEATHER, CALENDAR)))

        assert [call.id for call in result.tool_calls] == ["call_1", "call_2"]
        assert [call.name for call in result.tool_calls] == ["get_weather", "list_events"]

    async def test_the_output_is_none_rather_than_an_empty_answer(self) -> None:
        """Empty text would read as a model that answered nothing, which is a bug."""
        adapter = FakeAdapter(_calls(_provider_call("call_1", "get_weather", "{}")))

        result = await _gateway(adapter).generate(_request())

        assert result.output is None

    async def test_a_requested_json_format_is_not_parsed_out_of_a_tool_call(self) -> None:
        """There is no JSON to recover: the model chose to call instead of answer."""
        adapter = FakeAdapter(_calls(_provider_call("call_1", "get_weather", "{}")))

        result = await _gateway(adapter).generate(
            _request(response_format=ResponseFormat.JSON_OBJECT)
        )

        assert result.output is None
        assert result.tool_calls[0].name == "get_weather"

    async def test_an_answer_without_calls_still_carries_no_tool_calls(self) -> None:
        result = await _gateway(FakeAdapter(_text("Sunny in Madrid"))).generate(_request())

        assert result.tool_calls == ()
        assert result.output == "Sunny in Madrid"


class TestArgumentsThatCannotBeHandedOver:
    """None of these can be dispatched, so none of them is a successful call."""

    @pytest.mark.parametrize(
        "arguments",
        ["not json at all", "[1, 2]", '"just a string"'],
        ids=["unparseable", "array", "scalar"],
    )
    async def test_arguments_that_are_not_an_object_fail_the_attempt(self, arguments: str) -> None:
        adapter = FakeAdapter(_calls(_provider_call("call_1", "get_weather", arguments)))

        with pytest.raises(AllAttemptsFailed) as raised:
            await _gateway(adapter).generate(_request())

        attempt = raised.value.attempts[0]
        assert attempt.outcome is AttemptOutcome.FAILED
        assert attempt.failure_phase is FailurePhase.OUTPUT_PARSING

    async def test_a_call_to_an_undeclared_tool_fails_the_attempt(self) -> None:
        """The application has no function to run for a name it never offered."""
        adapter = FakeAdapter(_calls(_provider_call("call_1", "wire_money", "{}")))

        with pytest.raises(AllAttemptsFailed) as raised:
            await _gateway(adapter).generate(_request())

        assert raised.value.attempts[0].failure_phase is FailurePhase.OUTPUT_PARSING

    async def test_a_call_without_its_provider_correlation_id_fails_the_attempt(self) -> None:
        adapter = FakeAdapter(_calls(_provider_call("", "get_weather", "{}")))

        with pytest.raises(AllAttemptsFailed) as raised:
            await _gateway(adapter).generate(_request())

        attempt = raised.value.attempts[0]
        assert attempt.failure_phase is FailurePhase.OUTPUT_PARSING
        assert attempt.billable is True

    async def test_the_attempt_is_still_billed(self) -> None:
        """The provider answered. That it answered unusably changes no invoice."""
        adapter = FakeAdapter(_calls(_provider_call("call_1", "get_weather", "nope")))

        with pytest.raises(AllAttemptsFailed) as raised:
            await _gateway(adapter).generate(_request())

        attempt = raised.value.attempts[0]
        assert attempt.billable is True
        assert attempt.usage.input_tokens == 10
        assert attempt.cost.microusd is not None

    async def test_the_fallback_gets_its_turn(self) -> None:
        adapter = FakeAdapter(
            _calls(_provider_call("call_1", "get_weather", "nope")),
            _text("Sunny in Madrid"),
        )

        result = await _gateway(adapter).generate(
            _request(fallback_policy=FallbackPolicy(models=("slow",)))
        )

        assert result.output == "Sunny in Madrid"
        assert result.execution.attempt_count == 2

    async def test_the_failure_never_repeats_the_arguments(self) -> None:
        """Arguments carry whatever the model was told; that is not log material."""
        secret = '{"iban": "ES91 2100 0418 4502 0005 1332"'
        adapter = FakeAdapter(_calls(_provider_call("call_1", "get_weather", secret)))

        with pytest.raises(AllAttemptsFailed) as raised:
            await _gateway(adapter).generate(_request())

        assert "ES91" not in str(raised.value)
        assert "ES91" not in str(raised.value.__cause__)


class TestAccounting:
    async def test_a_provider_failure_before_a_tool_call_is_still_counted(self) -> None:
        adapter = FakeAdapter(
            RateLimitedError("slow down"),
            _calls(_provider_call("call_1", "get_weather", "{}")),
        )

        result = await _gateway(adapter).generate(
            _request(fallback_policy=FallbackPolicy(models=("slow",)))
        )

        assert result.execution.attempt_count == 2
        assert result.execution.attempts[0].outcome is AttemptOutcome.FAILED
        assert result.tool_calls[0].id == "call_1"

    async def test_the_sinks_are_told_nothing_about_the_tools(self) -> None:
        """Definitions, arguments and results may hold PII or credentials."""
        sink = RecordingSink()
        adapter = FakeAdapter(_calls(_provider_call("call_1", "get_weather", '{"city": "Madrid"}')))

        await _gateway(adapter, sink).generate(_request())

        reported = repr(sink.records) + repr(sink.events)
        assert "get_weather" not in reported
        assert "Madrid" not in reported
        assert "call_1" not in reported


class TestContinuing:
    async def test_the_results_reach_the_adapter_with_their_calls(self) -> None:
        """The package correlates and forwards; it never runs the function."""
        call = _tool_call("call_1", "get_weather", {"city": "Madrid"})
        adapter = FakeAdapter(_text("Sunny in Madrid"))

        result = await _gateway(adapter).generate(
            _request(tool_results=(ToolResult(call, "18C and sunny"),))
        )

        sent = adapter.requests[0]
        assert sent.tool_results[0].call.id == "call_1"
        assert sent.tool_results[0].output == "18C and sunny"
        assert result.output == "Sunny in Madrid"


def _provider_call(call_id: str, name: str, arguments: str) -> ProviderToolCall:
    return ProviderToolCall(id=call_id, name=name, arguments=arguments)


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)

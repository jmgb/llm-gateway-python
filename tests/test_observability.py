"""What leaves the package, and what must never leave it."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import BaseModel

from llm_gateway import (
    AllAttemptsFailed,
    CostMeasurement,
    FallbackPolicy,
    LLMGateway,
    LLMRequest,
    Message,
    ModelRate,
    ProviderRegistry,
    ProviderResponse,
    RateLimitedError,
    ResponseFormat,
    RetryPolicy,
    StaticPriceCatalog,
    TokenUsage,
    UsageRecord,
)
from llm_gateway.errors import MAX_ERROR_MESSAGE_CHARS, OutputParsingError
from llm_gateway.json_payload import parse_json_payload

SECRET_PROMPT = "SENSITIVE-PROMPT-CANARY-DO-NOT-LEAK"
CATALOG = StaticPriceCatalog(
    version="v1",
    rates={
        "m": ModelRate(Decimal("1"), Decimal("1")),
        "fallback": ModelRate(Decimal("1"), Decimal("1")),
    },
)


class StructuredAnswer(BaseModel):
    answer: str


class CollectingSinks:
    def __init__(self) -> None:
        self.usage: list[UsageRecord] = []
        self.events: list[tuple[str, dict[str, object]]] = []
        self.alerts: list[tuple[str, dict[str, object]]] = []

    def record(self, usage: UsageRecord) -> None:
        self.usage.append(usage)

    def emit(self, event: str, fields: dict[str, object]) -> None:
        self.events.append((event, fields))

    def alert(self, message: str, fields: dict[str, object]) -> None:
        self.alerts.append((message, fields))


class StubAdapter:
    name = "stub"

    def __init__(self, *items: object) -> None:
        self._queue = list(items)

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, ProviderResponse)
        return item


def _ok() -> ProviderResponse:
    return ProviderResponse(
        output_text=f"an answer about {SECRET_PROMPT}",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        finish_reason="stop",
    )


def _ok_json() -> ProviderResponse:
    return ProviderResponse(
        output_text='{"answer":"ok"}',
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        finish_reason="stop",
    )


def _build(adapter: StubAdapter, sinks: CollectingSinks) -> LLMGateway:
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=("m", "fallback"))
    return LLMGateway(
        registry=registry,
        price_catalog=CATALOG,
        usage_sink=sinks,
        event_sink=sinks,
        alert_sink=sinks,
    )


def _request(**kwargs: object) -> LLMRequest:
    return LLMRequest(
        model="m",
        messages=(Message("user", SECRET_PROMPT),),
        system_prompt=SECRET_PROMPT,
        request_id="req-1",
        source="test-feature",
        **kwargs,  # type: ignore[arg-type]
    )


class TestSinksNeverSeeContent:
    async def test_the_usage_record_carries_no_prompt_or_response(self) -> None:
        sinks = CollectingSinks()

        await _build(StubAdapter(_ok()), sinks).generate(_request())

        assert SECRET_PROMPT not in repr(sinks.usage)

    async def test_events_carry_no_prompt_or_response(self) -> None:
        sinks = CollectingSinks()

        await _build(StubAdapter(_ok()), sinks).generate(_request())

        assert SECRET_PROMPT not in repr(sinks.events)


class TestSinksSeeWhatIsNeededToReconcile:
    async def test_the_usage_record_identifies_the_call(self) -> None:
        sinks = CollectingSinks()

        await _build(StubAdapter(_ok()), sinks).generate(_request())

        record = sinks.usage[0]
        assert record.request_id == "req-1"
        assert record.source == "test-feature"
        assert record.provider == "stub"
        assert record.model_used == "m"
        assert record.cost.microusd == 15
        assert record.succeeded is True

    async def test_a_failed_call_is_still_reported_to_the_usage_sink(self) -> None:
        sinks = CollectingSinks()

        with pytest.raises(AllAttemptsFailed):
            await _build(StubAdapter(RateLimitedError("429")), sinks).generate(_request())

        assert sinks.usage[0].succeeded is False
        assert sinks.usage[0].cost.measurement is CostMeasurement.UNAVAILABLE

    async def test_an_unusable_answer_and_its_fallback_reach_every_sink(self) -> None:
        sinks = CollectingSinks()
        invalid = ProviderResponse(
            output_text="not json",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            finish_reason="stop",
        )
        valid = ProviderResponse(
            output_text='{"answer":"ok"}',
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            finish_reason="stop",
        )

        await _build(StubAdapter(invalid, valid), sinks).generate(
            _request(
                response_format=ResponseFormat.JSON_SCHEMA,
                response_schema=StructuredAnswer,
                fallback_policy=FallbackPolicy.models_in_order("fallback"),
            )
        )

        record = sinks.usage[0]
        assert record.succeeded is True
        assert record.attempts == 2
        assert record.cost.microusd == 30
        assert sinks.events[0][0] == "llm_call_succeeded"
        assert sinks.alerts[0][0] == "llm_fallback_used"

    async def test_the_fallback_alert_carries_the_failure_that_caused_it(self) -> None:
        sinks = CollectingSinks()

        await _build(StubAdapter(RateLimitedError("groq quota exceeded"), _ok()), sinks).generate(
            _request(fallback_policy=FallbackPolicy.models_in_order("fallback"))
        )

        name, fields = sinks.alerts[0]
        assert name == "llm_fallback_used"
        assert fields["requested_model"] == "m"
        assert fields["model_used"] == "fallback"
        assert fields["error_type"] == "RateLimitedError"
        assert fields["error_message"] == "groq quota exceeded"
        assert fields["failure_phase"] == "provider"

    async def test_the_fallback_alert_blames_the_requested_model_not_a_later_one(self) -> None:
        """The cause is the failure that ended the requested model's turn.

        A three-model plan can fail twice. Reporting the last failure would
        name the middle model's problem as the reason the first was left.
        """
        sinks = CollectingSinks()
        registry = ProviderRegistry()
        registry.register(
            StubAdapter(
                RateLimitedError("groq is out of quota"), RateLimitedError("second"), _ok()
            ),
            model_prefixes=("m", "second", "third"),
        )
        gateway = LLMGateway(
            registry=registry,
            price_catalog=StaticPriceCatalog(
                version="v1",
                rates={
                    name: ModelRate(Decimal("1"), Decimal("1")) for name in ("m", "second", "third")
                },
            ),
            usage_sink=sinks,
            event_sink=sinks,
            alert_sink=sinks,
        )

        await gateway.generate(
            _request(fallback_policy=FallbackPolicy.models_in_order("second", "third"))
        )

        assert sinks.alerts[0][1]["error_message"] == "groq is out of quota"

    async def test_an_unusable_answer_is_reported_as_the_fallback_cause(self) -> None:
        sinks = CollectingSinks()
        invalid = ProviderResponse(
            output_text="not json",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            finish_reason="stop",
        )

        await _build(StubAdapter(invalid, _ok_json()), sinks).generate(
            _request(
                response_format=ResponseFormat.JSON_SCHEMA,
                response_schema=StructuredAnswer,
                fallback_policy=FallbackPolicy.models_in_order("fallback"),
            )
        )

        fields = sinks.alerts[0][1]
        assert fields["error_type"] == "OutputParsingError"
        assert fields["failure_phase"] == "output_parsing"
        assert fields["error_message"]

    async def test_the_alert_lists_every_failure_not_only_the_cause(self) -> None:
        """A retry and two models can fail three times for three reasons."""
        sinks = CollectingSinks()
        registry = ProviderRegistry()
        registry.register(
            StubAdapter(
                RateLimitedError("groq 429 first"),
                RateLimitedError("groq 429 retry"),
                RateLimitedError("openai overloaded"),
                _ok(),
            ),
            model_prefixes=("m", "second", "third"),
        )
        gateway = LLMGateway(
            registry=registry,
            price_catalog=StaticPriceCatalog(
                version="v1",
                rates={n: ModelRate(Decimal("1"), Decimal("1")) for n in ("m", "second", "third")},
            ),
            usage_sink=sinks,
            event_sink=sinks,
            alert_sink=sinks,
        )

        await gateway.generate(
            _request(
                retry_policy=RetryPolicy.transient(max_attempts=2),
                fallback_policy=FallbackPolicy.models_in_order("second", "third"),
            )
        )

        failures = sinks.alerts[0][1]["failures"]
        assert isinstance(failures, list)
        assert [f["model"] for f in failures] == ["m", "m", "second"]
        assert [f["error_message"] for f in failures] == [
            "groq 429 first",
            "groq 429 retry",
            "openai overloaded",
        ]
        assert [f["attempt"] for f in failures] == [1, 2, 3]
        # The headline still names what ended the requested model's turn.
        assert sinks.alerts[0][1]["error_message"] == "groq 429 retry"

    async def test_the_listed_failures_exclude_the_attempt_that_succeeded(self) -> None:
        sinks = CollectingSinks()

        await _build(StubAdapter(RateLimitedError("only failure"), _ok()), sinks).generate(
            _request(fallback_policy=FallbackPolicy.models_in_order("fallback"))
        )

        failures = sinks.alerts[0][1]["failures"]
        assert isinstance(failures, list)
        assert len(failures) == 1
        assert failures[0]["model"] == "m"
        assert failures[0]["failure_phase"] == "provider"

    async def test_the_fallback_alert_carries_no_prompt_or_response(self) -> None:
        """The cause is a reason, never an echo of what was sent."""
        sinks = CollectingSinks()

        await _build(StubAdapter(RateLimitedError("quota"), _ok()), sinks).generate(
            _request(fallback_policy=FallbackPolicy.models_in_order("fallback"))
        )

        assert SECRET_PROMPT not in repr(sinks.alerts)

    async def test_a_long_provider_message_is_truncated_before_it_reaches_a_sink(self) -> None:
        sinks = CollectingSinks()

        await _build(StubAdapter(RateLimitedError("x" * 5_000), _ok()), sinks).generate(
            _request(fallback_policy=FallbackPolicy.models_in_order("fallback"))
        )

        message = sinks.alerts[0][1]["error_message"]
        assert isinstance(message, str)
        assert len(message) == MAX_ERROR_MESSAGE_CHARS + 1
        assert message.endswith("…")

    async def test_a_silent_exception_is_recorded_by_its_class_not_as_empty(self) -> None:
        """An empty string would read as "no error", which is a different claim."""
        sinks = CollectingSinks()

        await _build(StubAdapter(RateLimitedError(""), _ok()), sinks).generate(
            _request(fallback_policy=FallbackPolicy.models_in_order("fallback"))
        )

        assert sinks.alerts[0][1]["error_message"] == "RateLimitedError"

    async def test_an_exhausted_call_still_exposes_what_the_last_failure_said(self) -> None:
        sinks = CollectingSinks()

        with pytest.raises(AllAttemptsFailed) as raised:
            await _build(
                StubAdapter(RateLimitedError("first"), RateLimitedError("last")), sinks
            ).generate(_request(fallback_policy=FallbackPolicy.models_in_order("fallback")))

        assert raised.value.last_error == "RateLimitedError"
        assert raised.value.last_error_message == "last"

    async def test_exhausted_unusable_answers_reach_the_failure_sinks_with_full_cost(self) -> None:
        sinks = CollectingSinks()
        invalid = ProviderResponse(
            output_text="not json",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            finish_reason="stop",
        )

        with pytest.raises(AllAttemptsFailed):
            await _build(StubAdapter(invalid, invalid), sinks).generate(
                _request(
                    response_format=ResponseFormat.JSON_SCHEMA,
                    response_schema=StructuredAnswer,
                    fallback_policy=FallbackPolicy.models_in_order("fallback"),
                )
            )

        record = sinks.usage[0]
        assert record.succeeded is False
        assert record.attempts == 2
        assert record.cost.microusd == 30
        assert sinks.events[0][0] == "llm_call_failed"
        assert sinks.alerts == []


class TestJSONRecovery:
    def test_a_fenced_payload_is_recovered(self) -> None:
        assert parse_json_payload('```json\n{"a": 1}\n```') == {"a": 1}

    def test_a_payload_padded_with_prose_is_recovered(self) -> None:
        assert parse_json_payload('Claro:\n{"a": 1}\nEspero que sirva.') == {"a": 1}

    def test_a_plain_payload_is_returned_unchanged(self) -> None:
        assert parse_json_payload('{"a": 1}') == {"a": 1}

    def test_an_array_payload_is_supported(self) -> None:
        assert parse_json_payload("[1, 2]") == [1, 2]

    def test_an_empty_response_is_an_error_not_an_empty_dict(self) -> None:
        with pytest.raises(OutputParsingError):
            parse_json_payload("   ")

    def test_unrecoverable_text_raises_without_echoing_the_payload(self) -> None:
        with pytest.raises(OutputParsingError) as caught:
            parse_json_payload("esto no es json en absoluto")

        assert "esto no es json" not in str(caught.value)

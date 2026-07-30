"""What leaves the package, and what must never leave it."""

from __future__ import annotations

from decimal import Decimal

import pytest

from llm_gateway import (
    AllAttemptsFailed,
    CostMeasurement,
    LLMGateway,
    LLMRequest,
    Message,
    ModelRate,
    ProviderRegistry,
    ProviderResponse,
    RateLimitedError,
    StaticPriceCatalog,
    TokenUsage,
    UsageRecord,
)
from llm_gateway.errors import OutputParsingError
from llm_gateway.json_payload import parse_json_payload

SECRET_PROMPT = "SENSITIVE-PROMPT-CANARY-DO-NOT-LEAK"
CATALOG = StaticPriceCatalog(version="v1", rates={"m": ModelRate(Decimal("1"), Decimal("1"))})


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


def _build(adapter: StubAdapter, sinks: CollectingSinks) -> LLMGateway:
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=("m",))
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

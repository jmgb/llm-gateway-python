"""Orchestration: attempts, retries, fallback, aggregation and typed output.

Every test here uses a fake adapter. The gateway's job is bookkeeping, and
bookkeeping is exactly what must be provable without a network.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import BaseModel

from llm_gateway import (
    AllAttemptsFailed,
    AuthenticationError,
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
    SchemaValidationError,
    StaticPriceCatalog,
    TokenUsage,
    UnknownModelError,
)

CATALOG = StaticPriceCatalog(
    version="test-1",
    rates={
        "fast": ModelRate(Decimal("1"), Decimal("2")),
        "slow": ModelRate(Decimal("1"), Decimal("2")),
    },
)


class FakeAdapter:
    """Returns queued responses or raises queued errors, in order."""

    name = "fake"

    def __init__(self, *responses: ProviderResponse | Exception) -> None:
        self._queue = list(responses)
        self.calls: list[str] = []

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        self.calls.append(model)
        item = self._queue.pop(0) if self._queue else _ok("default")
        if isinstance(item, Exception):
            raise item
        return item


def _ok(text: str, *, usage: TokenUsage | None = None) -> ProviderResponse:
    return ProviderResponse(
        output_text=text,
        usage=usage or TokenUsage(input_tokens=10, output_tokens=5),
        finish_reason="stop",
    )


def _gateway(adapter: FakeAdapter) -> LLMGateway:
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=("fast", "slow"))
    return LLMGateway(registry=registry, price_catalog=CATALOG)


def _request(model: str = "fast", **kwargs: object) -> LLMRequest:
    return LLMRequest(model=model, messages=(Message("user", "hello"),), **kwargs)  # type: ignore[arg-type]


class TestHappyPath:
    async def test_it_returns_the_provider_text(self) -> None:
        result = await _gateway(FakeAdapter(_ok("an answer"))).generate(_request())

        assert result.output == "an answer"

    async def test_it_records_exactly_one_attempt(self) -> None:
        result = await _gateway(FakeAdapter(_ok("x"))).generate(_request())

        assert result.execution.attempt_count == 1
        assert result.execution.model_used == "fast"
        assert result.execution.provider == "fake"

    async def test_it_prices_the_call_with_the_injected_catalog(self) -> None:
        result = await _gateway(FakeAdapter(_ok("x"))).generate(_request())

        assert result.cost.microusd == 20
        assert result.cost.measurement is CostMeasurement.ACTUAL
        assert result.cost.pricing_version == "test-1"

    async def test_no_fallback_means_no_fallback_flag(self) -> None:
        result = await _gateway(FakeAdapter(_ok("x"))).generate(_request())

        assert result.execution.fallback_used is False


class TestRetries:
    async def test_a_transient_error_is_retried_on_the_same_model(self) -> None:
        adapter = FakeAdapter(RateLimitedError("429"), _ok("recovered"))

        result = await _gateway(adapter).generate(
            _request(retry_policy=RetryPolicy.transient(max_attempts=2, base_delay_seconds=0))
        )

        assert result.output == "recovered"
        assert adapter.calls == ["fast", "fast"]

    async def test_a_permanent_error_is_not_retried(self) -> None:
        adapter = FakeAdapter(AuthenticationError("401"), _ok("never reached"))

        with pytest.raises(AllAttemptsFailed):
            await _gateway(adapter).generate(
                _request(retry_policy=RetryPolicy.transient(max_attempts=3, base_delay_seconds=0))
            )

        assert adapter.calls == ["fast"]

    async def test_a_failed_attempt_is_still_recorded(self) -> None:
        adapter = FakeAdapter(RateLimitedError("429"), _ok("ok"))

        result = await _gateway(adapter).generate(
            _request(retry_policy=RetryPolicy.transient(max_attempts=2, base_delay_seconds=0))
        )

        assert result.execution.attempt_count == 2
        assert result.execution.attempts[0].error_type == "RateLimitedError"


class TestFallback:
    async def test_fallback_is_disabled_by_default(self) -> None:
        adapter = FakeAdapter(RateLimitedError("429"))

        with pytest.raises(AllAttemptsFailed):
            await _gateway(adapter).generate(_request())

        assert adapter.calls == ["fast"]

    async def test_an_enabled_fallback_switches_model(self) -> None:
        adapter = FakeAdapter(RateLimitedError("429"), _ok("from slow"))

        result = await _gateway(adapter).generate(
            _request(fallback_policy=FallbackPolicy.models_in_order("slow"))
        )

        assert result.output == "from slow"
        assert adapter.calls == ["fast", "slow"]

    async def test_a_fallback_is_always_visible_in_the_result(self) -> None:
        adapter = FakeAdapter(RateLimitedError("429"), _ok("from slow"))

        result = await _gateway(adapter).generate(
            _request(fallback_policy=FallbackPolicy.models_in_order("slow"))
        )

        assert result.execution.fallback_used is True
        assert result.execution.requested_model == "fast"
        assert result.execution.model_used == "slow"


class TestCostAggregation:
    async def test_the_total_bills_every_attempt_not_just_the_successful_one(self) -> None:
        adapter = FakeAdapter(
            RateLimitedError("429"),
            _ok("ok", usage=TokenUsage(input_tokens=10, output_tokens=5)),
        )

        result = await _gateway(adapter).generate(
            _request(retry_policy=RetryPolicy.transient(max_attempts=2, base_delay_seconds=0))
        )

        assert result.usage.input_tokens == 10, "the failed attempt reported no usage"
        assert result.cost.measurement is CostMeasurement.ESTIMATED, (
            "a failed attempt may have been billed, so the total is a lower bound"
        )


class TestStructuredOutput:
    class Answer(BaseModel):
        veredicto: str

    async def test_json_schema_output_is_validated_into_the_model(self) -> None:
        adapter = FakeAdapter(_ok('{"veredicto": "estimado"}'))

        result = await _gateway(adapter).generate(
            _request(
                response_format=ResponseFormat.JSON_SCHEMA,
                response_schema=TestStructuredOutput.Answer,
            )
        )

        assert isinstance(result.output, TestStructuredOutput.Answer)
        assert result.output.veredicto == "estimado"

    async def test_output_never_carries_technical_metadata(self) -> None:
        adapter = FakeAdapter(_ok('{"veredicto": "estimado"}'))

        result = await _gateway(adapter).generate(
            _request(
                response_format=ResponseFormat.JSON_SCHEMA,
                response_schema=TestStructuredOutput.Answer,
            )
        )

        assert not hasattr(result.output, "tokens_in")
        assert result.usage.input_tokens == 10

    async def test_a_payload_that_violates_the_schema_raises(self) -> None:
        adapter = FakeAdapter(_ok('{"otra_cosa": 1}'))

        with pytest.raises(SchemaValidationError):
            await _gateway(adapter).generate(
                _request(
                    response_format=ResponseFormat.JSON_SCHEMA,
                    response_schema=TestStructuredOutput.Answer,
                )
            )

    async def test_json_object_output_is_returned_as_a_dict(self) -> None:
        adapter = FakeAdapter(_ok('{"a": 1}'))

        result = await _gateway(adapter).generate(
            _request(response_format=ResponseFormat.JSON_OBJECT)
        )

        assert result.output == {"a": 1}


class TestRouting:
    async def test_an_unroutable_model_fails_before_spending_anything(self) -> None:
        gateway = _gateway(FakeAdapter(_ok("x")))

        with pytest.raises(UnknownModelError):
            await gateway.generate(_request(model="unknown-model-xyz"))


class TestFailureAccounting:
    async def test_an_exhausted_call_still_reports_its_attempts(self) -> None:
        adapter = FakeAdapter(RateLimitedError("429"), RateLimitedError("429"))

        with pytest.raises(AllAttemptsFailed) as caught:
            await _gateway(adapter).generate(
                _request(retry_policy=RetryPolicy.transient(max_attempts=2, base_delay_seconds=0))
            )

        assert len(caught.value.attempts) == 2
        assert caught.value.last_error == "RateLimitedError"

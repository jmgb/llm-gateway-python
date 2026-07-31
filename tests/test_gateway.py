"""Orchestration: attempts, retries, fallback, aggregation and typed output.

Every test here uses a fake adapter. The gateway's job is bookkeeping, and
bookkeeping is exactly what must be provable without a network.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel

from llm_gateway import (
    AllAttemptsFailed,
    AttemptOutcome,
    AuthenticationError,
    CostMeasurement,
    FailurePhase,
    FallbackPolicy,
    LLMGateway,
    LLMRequest,
    Message,
    ModelRate,
    OutputParsingError,
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
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        self.calls.append(model)
        self.requests.append(request)
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

    async def test_a_payload_that_violates_the_schema_exhausts_the_call(self) -> None:
        adapter = FakeAdapter(_ok('{"otra_cosa": 1}'))

        with pytest.raises(AllAttemptsFailed) as caught:
            await _gateway(adapter).generate(
                _request(
                    response_format=ResponseFormat.JSON_SCHEMA,
                    response_schema=TestStructuredOutput.Answer,
                )
            )

        assert isinstance(caught.value.__cause__, SchemaValidationError), (
            "the exhausted call must still say why the output was unusable"
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


class TestReasoningEffortRouting:
    @pytest.mark.parametrize("effort", ("low", "medium", "high"))
    async def test_reasoning_effort_is_allowed_for_groq_oss(self, effort: str) -> None:
        adapter = FakeAdapter(_ok("x"))
        adapter.name = "groq"
        registry = ProviderRegistry()
        registry.register(adapter, model_prefixes=("openai/gpt-oss-",))
        gateway = LLMGateway(registry=registry)

        await gateway.generate(_request(model="openai/gpt-oss-120b", reasoning_effort=effort))

        assert adapter.calls == ["openai/gpt-oss-120b"]

    async def test_reasoning_effort_is_allowed_for_gemini_3_flash(self) -> None:
        adapter = FakeAdapter(_ok("x"))
        adapter.name = "gemini"
        registry = ProviderRegistry()
        registry.register(adapter, model_prefixes=("gemini-",))
        gateway = LLMGateway(registry=registry)

        await gateway.generate(_request(model="gemini-3-flash-preview", reasoning_effort="minimal"))

        assert adapter.calls == ["gemini-3-flash-preview"]

    async def test_an_unsupported_effort_is_downgraded_for_gemini_3_pro(self) -> None:
        adapter = FakeAdapter(_ok("x"))
        adapter.name = "gemini"
        registry = ProviderRegistry()
        registry.register(adapter, model_prefixes=("gemini-",))
        gateway = LLMGateway(registry=registry)

        await gateway.generate(_request(model="gemini-3.1-pro-preview", reasoning_effort="max"))

        assert adapter.requests[0].reasoning_effort == "medium"

    async def test_an_unsupported_effort_is_downgraded_before_a_groq_fallback(self) -> None:
        primary = FakeAdapter(RateLimitedError("429"))
        primary.name = "openai"
        fallback = FakeAdapter(_ok("fallback answer"))
        fallback.name = "groq"
        registry = ProviderRegistry()
        registry.register(primary, model_prefixes=("gpt-5.6-",))
        registry.register(fallback, model_prefixes=("openai/gpt-oss-",))
        gateway = LLMGateway(registry=registry)

        result = await gateway.generate(
            _request(
                model="gpt-5.6-luna",
                reasoning_effort="max",
                fallback_policy=FallbackPolicy.models_in_order("openai/gpt-oss-120b"),
            )
        )

        assert result.output == "fallback answer"
        assert primary.requests[0].reasoning_effort == "max"
        assert fallback.requests[0].reasoning_effort == "medium"

    async def test_reasoning_effort_is_removed_for_a_model_without_reasoning_support(self) -> None:
        adapter = FakeAdapter(_ok("x"))
        adapter.name = "openai"
        registry = ProviderRegistry()
        registry.register(adapter, model_prefixes=("gpt-",))
        gateway = LLMGateway(registry=registry)

        await gateway.generate(_request(model="gpt-5.2-2025-12-11", reasoning_effort="medium"))

        assert adapter.requests[0].reasoning_effort is None

    async def test_reasoning_effort_is_allowed_for_openai_56_models(self) -> None:
        adapter = FakeAdapter(_ok("x"))
        adapter.name = "openai"
        registry = ProviderRegistry()
        registry.register(adapter, model_prefixes=("gpt-5.6-",))
        gateway = LLMGateway(registry=registry)

        await gateway.generate(_request(model="gpt-5.6-terra", reasoning_effort="max"))

        assert adapter.calls == ["gpt-5.6-terra"]


class TestUnusableOutput:
    """An attempt that answers with unusable output is a *failed billable* attempt.

    The provider produced tokens, so the money was spent; what came back cannot
    be given to the caller. Both facts have to survive into the accounting.
    """

    class Answer(BaseModel):
        veredicto: str

    def _schema_request(self, **kwargs: Any) -> LLMRequest:
        return _request(
            response_format=ResponseFormat.JSON_SCHEMA,
            response_schema=TestUnusableOutput.Answer,
            **kwargs,
        )

    async def test_unparseable_json_falls_back_to_the_next_model(self) -> None:
        adapter = FakeAdapter(_ok("not json at all"), _ok('{"veredicto": "ok"}'))

        result = await _gateway(adapter).generate(
            self._schema_request(fallback_policy=FallbackPolicy.models_in_order("slow"))
        )

        assert result.output.veredicto == "ok"
        assert adapter.calls == ["fast", "slow"]

    async def test_a_schema_violation_falls_back_to_the_next_model(self) -> None:
        adapter = FakeAdapter(_ok('{"otra_cosa": 1}'), _ok('{"veredicto": "ok"}'))

        result = await _gateway(adapter).generate(
            self._schema_request(fallback_policy=FallbackPolicy.models_in_order("slow"))
        )

        assert result.execution.fallback_used is True
        assert result.execution.model_used == "slow"

    async def test_the_unusable_attempt_is_billed_with_the_tokens_it_spent(self) -> None:
        adapter = FakeAdapter(_ok("not json at all"), _ok('{"veredicto": "ok"}'))

        result = await _gateway(adapter).generate(
            self._schema_request(fallback_policy=FallbackPolicy.models_in_order("slow"))
        )

        rejected = result.execution.attempts[0]
        assert rejected.outcome is AttemptOutcome.FAILED
        assert rejected.billable is True
        assert rejected.usage.input_tokens == 10
        assert rejected.cost.microusd == 20
        assert result.usage.input_tokens == 20, "both attempts consumed input tokens"
        assert result.cost.microusd == 40, "a rejected answer is still on the invoice"

    async def test_the_failure_phase_distinguishes_parsing_from_schema_violation(self) -> None:
        parsing = FakeAdapter(_ok("not json at all"), _ok('{"veredicto": "ok"}'))
        schema = FakeAdapter(_ok('{"otra_cosa": 1}'), _ok('{"veredicto": "ok"}'))
        request = self._schema_request(fallback_policy=FallbackPolicy.models_in_order("slow"))

        from_parsing = await _gateway(parsing).generate(request)
        from_schema = await _gateway(schema).generate(request)

        assert from_parsing.execution.attempts[0].failure_phase is FailurePhase.OUTPUT_PARSING
        assert from_parsing.execution.attempts[0].error_type == "OutputParsingError"
        assert from_schema.execution.attempts[0].failure_phase is FailurePhase.SCHEMA_VALIDATION

    async def test_unusable_output_is_not_retried_on_the_same_model(self) -> None:
        """The same prompt on the same model reproduces the same malformed answer."""
        adapter = FakeAdapter(_ok("not json"), _ok('{"veredicto": "ok"}'))

        with pytest.raises(AllAttemptsFailed):
            await _gateway(adapter).generate(
                self._schema_request(
                    retry_policy=RetryPolicy(max_attempts=3, retry_transient_only=False)
                )
            )

        assert adapter.calls == ["fast"]

    async def test_exhausting_every_model_reports_each_unusable_attempt(self) -> None:
        adapter = FakeAdapter(_ok("not json"), _ok('{"otra_cosa": 1}'))

        with pytest.raises(AllAttemptsFailed) as caught:
            await _gateway(adapter).generate(
                self._schema_request(fallback_policy=FallbackPolicy.models_in_order("slow"))
            )

        attempts = caught.value.attempts
        assert [a.failure_phase for a in attempts] == [
            FailurePhase.OUTPUT_PARSING,
            FailurePhase.SCHEMA_VALIDATION,
        ]
        assert all(a.billable for a in attempts)
        assert isinstance(caught.value.__cause__, SchemaValidationError)

    async def test_a_json_object_request_is_held_to_the_same_rule(self) -> None:
        adapter = FakeAdapter(_ok("prose, not json"), _ok('{"a": 1}'))

        result = await _gateway(adapter).generate(
            _request(
                response_format=ResponseFormat.JSON_OBJECT,
                fallback_policy=FallbackPolicy.models_in_order("slow"),
            )
        )

        assert result.output == {"a": 1}
        assert result.execution.attempts[0].error_type == OutputParsingError.__name__

    async def test_a_provider_failure_is_labelled_as_such(self) -> None:
        adapter = FakeAdapter(RateLimitedError("429"), _ok("ok"))

        result = await _gateway(adapter).generate(
            _request(fallback_policy=FallbackPolicy.models_in_order("slow"))
        )

        assert result.execution.attempts[0].failure_phase is FailurePhase.PROVIDER


class TestRequestNormalisation:
    """Every attempt sends only options the target model accepts."""

    def _openai_gateway(self, adapter: FakeAdapter) -> LLMGateway:
        adapter.name = "openai"
        registry = ProviderRegistry()
        registry.register(adapter, model_prefixes=("gpt-",))
        return LLMGateway(registry=registry)

    async def test_temperature_is_dropped_for_a_model_that_rejects_it(self) -> None:
        adapter = FakeAdapter(_ok("x"))

        await self._openai_gateway(adapter).generate(
            _request(model="gpt-5.6-luna", temperature=0.2)
        )

        assert adapter.requests[0].temperature is None

    async def test_temperature_survives_for_a_model_that_accepts_it(self) -> None:
        adapter = FakeAdapter(_ok("x"))

        await self._openai_gateway(adapter).generate(
            _request(model="gpt-5.2-2025-12-11", temperature=0.2)
        )

        assert adapter.requests[0].temperature == 0.2

    async def test_a_fallback_does_not_inherit_a_temperature_its_model_rejects(self) -> None:
        adapter = FakeAdapter(RateLimitedError("429"), _ok("from the fallback"))

        result = await self._openai_gateway(adapter).generate(
            _request(
                model="gpt-5.2-2025-12-11",
                temperature=0.2,
                fallback_policy=FallbackPolicy.models_in_order("gpt-5.6-terra"),
            )
        )

        assert result.output == "from the fallback"
        assert adapter.requests[0].temperature == 0.2
        assert adapter.requests[1].temperature is None

    async def test_an_uncatalogued_model_keeps_the_temperature_it_was_given(self) -> None:
        """Silence in the catalogue is not evidence that an option is rejected."""
        adapter = FakeAdapter(_ok("x"))

        await self._openai_gateway(adapter).generate(
            _request(model="gpt-brand-new", temperature=0.2)
        )

        assert adapter.requests[0].temperature == 0.2


class TestFailureAccounting:
    async def test_an_exhausted_call_still_reports_its_attempts(self) -> None:
        adapter = FakeAdapter(RateLimitedError("429"), RateLimitedError("429"))

        with pytest.raises(AllAttemptsFailed) as caught:
            await _gateway(adapter).generate(
                _request(retry_policy=RetryPolicy.transient(max_attempts=2, base_delay_seconds=0))
            )

        assert len(caught.value.attempts) == 2
        assert caught.value.last_error == "RateLimitedError"

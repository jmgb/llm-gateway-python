"""Cost must never launder an unknown into a real zero."""

from decimal import Decimal

import pytest

from llm_gateway import (
    CostMeasurement,
    ModelRate,
    NullPriceCatalog,
    StaticPriceCatalog,
    TokenUsage,
)

CATALOG = StaticPriceCatalog(
    version="test-2026-07-30",
    rates={
        "test-model": ModelRate(
            input_microusd_per_token=Decimal("0.3"),
            output_microusd_per_token=Decimal("2.5"),
        )
    },
)


def test_complete_usage_produces_an_actual_measurement() -> None:
    cost = CATALOG.estimate("test-model", TokenUsage(input_tokens=1000, output_tokens=100))

    assert cost.measurement is CostMeasurement.ACTUAL
    assert cost.microusd == 550
    assert cost.amount_usd == Decimal("0.000550")
    assert cost.pricing_version == "test-2026-07-30"


def test_incomplete_usage_produces_a_lower_bound_estimate() -> None:
    cost = CATALOG.estimate("test-model", TokenUsage(input_tokens=1000, output_tokens=None))

    assert cost.measurement is CostMeasurement.ESTIMATED
    assert cost.microusd == 300


def test_unknown_model_never_reports_zero_dollars() -> None:
    cost = CATALOG.estimate("model-with-no-rate", TokenUsage(input_tokens=1000, output_tokens=100))

    assert cost.measurement is CostMeasurement.UNAVAILABLE
    assert cost.amount_usd is None
    assert cost.microusd is None


def test_null_catalog_reports_unavailable_not_free() -> None:
    cost = NullPriceCatalog().estimate("anything", TokenUsage(input_tokens=10, output_tokens=10))

    assert cost.measurement is CostMeasurement.UNAVAILABLE
    assert cost.amount_usd is None


def test_retrieved_documents_are_billed_as_input() -> None:
    usage = TokenUsage(input_tokens=1000, output_tokens=0, retrieved_document_tokens=1000)

    cost = CATALOG.estimate("test-model", usage)

    assert cost.microusd == 600


def test_reasoning_tokens_are_already_inside_the_output_count() -> None:
    thinking = CATALOG.estimate(
        "test-model", TokenUsage(input_tokens=0, output_tokens=100, reasoning_tokens=100)
    )
    without_breakdown = CATALOG.estimate(
        "test-model", TokenUsage(input_tokens=0, output_tokens=100)
    )

    assert thinking.microusd == 250
    assert thinking.microusd == without_breakdown.microusd, "the breakdown must not change the bill"


def test_costs_aggregate_across_attempts() -> None:
    first = CATALOG.estimate("test-model", TokenUsage(input_tokens=1000, output_tokens=100))
    second = CATALOG.estimate("test-model", TokenUsage(input_tokens=1000, output_tokens=100))

    total = first.merge(second)

    assert total.microusd == 1100
    assert total.measurement is CostMeasurement.ACTUAL


def test_aggregating_an_unavailable_cost_downgrades_the_total() -> None:
    known = CATALOG.estimate("test-model", TokenUsage(input_tokens=1000, output_tokens=100))
    unknown = CATALOG.estimate("model-with-no-rate", TokenUsage(input_tokens=1000, output_tokens=1))

    total = known.merge(unknown)

    assert total.measurement is CostMeasurement.ESTIMATED
    assert total.microusd == 550, "the known part survives as a lower bound"


def test_rounding_is_half_up_on_whole_microdollars() -> None:
    catalog = StaticPriceCatalog(
        version="v",
        rates={"m": ModelRate(Decimal("0.5"), Decimal("0"))},
    )

    cost = catalog.estimate("m", TokenUsage(input_tokens=1, output_tokens=0))

    assert cost.microusd == 1


def test_pricing_version_is_required_to_be_non_empty() -> None:
    with pytest.raises(ValueError, match="pricing version"):
        StaticPriceCatalog(version="", rates={})
